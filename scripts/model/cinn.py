import torch
import torch.nn as nn
import torch.optim

import scripts.data.data as data
import config as c
import numpy as np

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import yaml

#Load the default params from the parameter file
model_default_params = yaml.safe_load(open('params.yaml'))['model']

class CondNet(nn.Module):
    '''Conditional Network for preproccessing of the conditional inputs'''
    def __init__(self, params={}):
        super().__init__()
        
        #Use default parameter if not set
        params = dict(model_default_params, **params)
        
        def condnet(ch_in, ch_out):
            """Defition of the conditional neural network"""
            sub = nn.Sequential()
            
            #Input Layer
            sub.add_module("Input_Linear_COND", nn.Linear(ch_in, params["NUM_HIDDEN_NODES_COND"]))
            
            if params["BATCH_NORM_COND"]:
                sub.add_module("Input_BN_COND", nn.BatchNorm1d(params["NUM_HIDDEN_NODES_COND"]))
            
            sub.add_module("Input_ReLU_COND", nn.ReLU())

            #Hidden Layers
            for i in range(params["NUM_HIDDEN_LAYERS_COND"]):
                sub.add_module("Linear_COND_" + str(i), nn.Linear(params["NUM_HIDDEN_NODES_COND"], params["NUM_HIDDEN_NODES_COND"]))
                
                if params["BATCH_NORM_COND"]:
                    sub.add_module("BN_COND_" + str(i), nn.BatchNorm1d(params["NUM_HIDDEN_NODES_COND"]))
                
                sub.add_module("ReLU_COND_" + str(i), nn.ReLU())
                
                #Apply Dropout only between hidden layers
                if i < params["NUM_HIDDEN_LAYERS_COND"] and params["DROPOUT_COND"] > 0.0:
                    sub.add_module("Dropout_COND_" + str(i), nn.Dropout(p=params["DROPOUT_COND"]))
                
            #Output layer
            sub.add_module("Output_Linear_COND", nn.Linear(params["NUM_HIDDEN_NODES_COND"], ch_out))
            
            return sub
            
        self.cond = condnet(data.NUM_COND, params["NUM_COND_NODES"])
        
    def forward(self, c):
        return self.cond(c)


class cINN(nn.Module):
    '''Main cINN definition including the conditional network'''
    def __init__(self, params={}):
        super().__init__()
        
        #Use default parameters if not set
        params = dict(model_default_params, **params)

        self.cinn = self.build_inn(params)
        self.cond_net = CondNet(params)

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        self.trainable_parameters += list(self.cond_net.parameters())
               
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)

        self.optimizer = torch.optim.Adam(self.trainable_parameters,
                                          lr=params["LEARNING_RATE"],
                                          weight_decay=params["L2_DECAY"],
                                          betas=(params["BETA_1"], params["BETA_2"]))
        
    @property
    def num_trainable_parameters(self):
        """Return Number of trainable Parameters"""
        return sum(p.numel() for p in self.trainable_parameters)

    def build_inn(self, params):
        """Defition of cINN"""

        def subnet(ch_in, ch_out):
            """Defition of the non-invertible sub neural networks used within the cINN"""
            sub = nn.Sequential()
            
            #Input Layer
            sub.add_module("Input_Linear", nn.Linear(ch_in, params["NUM_HIDDEN_NODES"]))
            
            if params["BATCH_NORM"]:
                sub.add_module("Input_BN", nn.BatchNorm1d(params["NUM_HIDDEN_NODES"]))
            
            sub.add_module("Input_ReLU", nn.ReLU())

            #Hidden Layers
            for i in range(params["NUM_HIDDEN_LAYERS"]):
                sub.add_module("Linear_" + str(i), nn.Linear(params["NUM_HIDDEN_NODES"], params["NUM_HIDDEN_NODES"]))
                
                if params["BATCH_NORM"]:
                    sub.add_module("BN_" + str(i), nn.BatchNorm1d(params["NUM_HIDDEN_NODES"]))
                
                sub.add_module("ReLU_" + str(i), nn.ReLU())
                
                #Apply Dropout only between hidden layers
                if i < params["NUM_HIDDEN_LAYERS"] and params["DROPOUT"] > 0.0:
                    sub.add_module("Dropout_" + str(i), nn.Dropout(p=params["DROPOUT"]))
                
            #Output layer
            sub.add_module("Output_Linear", nn.Linear(params["NUM_HIDDEN_NODES"], ch_out))
            
            return sub
        
         
        nodes = [Ff.InputNode(data.NUM_DIM, name='input')]
        cond = [Ff.ConditionNode(params["NUM_COND_NODES"], name='Condition')]
        
        for k in range(params["NUM_COUPLING_LAYERS"]):
            nodes.append(Ff.Node(nodes[-1],
                         Fm.GLOWCouplingBlock,
                         {'subnet_constructor':subnet, 'clamp':params["CLAMP"]},
                         conditions=cond,
                         name=F'fully_connected_{k}'))
            nodes.append(Ff.Node(nodes[-1],
                         Fm.PermuteRandom,
                         {'seed':k},
                         name=F'permute_{k}'))
            
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))

        return Ff.GraphINN(nodes + cond)

    def forward(self, x, l):
        z, jac = self.cinn(x, c=self.cond_net(l), rev=False, jac=True)
        return z, jac

    def reverse_sample(self, z, l):
        return self.cinn(z, c=self.cond_net(l), rev=True, jac=False)[0]
 
    
#Further methods to load and evaluate models
#------------------------------------------------------------------------------

class Ensemble():
    '''
    Class for Ensemble Averaging
    
    Can be used to evaluate just like the original model; however it 
    randomly samples from the given list of models instead
    '''
    def __init__(self, models):
        self.__models = models
        
    def forward(self, x, l):
        rand_index = np.random.randint(len(self.__models), size=len(x))
        
        z = torch.empty_like(x)
        
        for i in range(len(self.__models)):
            mask = (i==rand_index)
            z[mask], _ = self.__models[i].forward(x[mask], l[mask])
            
        return z, None
        
    def reverse_sample(self, z, l):
        rand_index = np.random.randint(len(self.__models), size=len(z))
        
        x = torch.empty_like(z)
        
        for i in range(len(self.__models)):
            mask = (i==rand_index)
            x[mask] = self.__models[i].reverse_sample(z[mask], l[mask])
            
        return x
        

def load_model(num_models=None):
    """
    Load model from the model folder

    Return an cINN object if only one model is present in the model folder
    Return an Esemble object of multiple models otherwise   
    
    :param num_models: number of models to sample from; if None all models in /models are used, if 1 a cINN object is returned
    """
    
    #Check if gpu or cpu only is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def load_single(i):
        cinn = cINN()
        cinn.to(device)
        state_dict = {k:v for k,v in torch.load('model/cinn_' + str(i) + '.pt', map_location=device).items() if 'tmp_var' not in k}
        cinn.load_state_dict(state_dict)
        cinn.eval()
        return cinn
        
    #Look for model sin the model path
    import glob
    filelist = glob.glob(c.model_path + "/*.pt")
    
    if num_models is None:
        num_models = len(filelist)
    
    if num_models == 1:
        return load_single(0)
    else:
        assert num_models <= len(filelist), "There are fewer models available as asked for!"
        
        models = []
        for i in range(num_models):
            models.append(load_single(i))
            
        return Ensemble(models)
    
