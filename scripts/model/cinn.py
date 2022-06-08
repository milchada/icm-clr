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
            
        self.cond = condnet(params["RESNET_REPRESENTATION_DIM"], params["NUM_COND_NODES"])
        
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
               
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)
    
    @property
    def trainable_parameters(self):
        """Return trainable parameters"""
        trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        trainable_parameters += list(self.cond_net.parameters())
        
        return trainable_parameters
    
    @property
    def num_trainable_parameters(self):
        """Return number of trainable parameters"""
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
 
    

