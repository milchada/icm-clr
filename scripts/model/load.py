import torch
import numpy as np
import glob
import config as c

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
        

def load_cinn_model(num_models=None, model_index=None):
    """
    Load cinn model from the model folder

    Return an cINN object if only one model is present in the model folder
    Return an Esemble object of multiple models otherwise   
    
    :param num_models: number of models to sample from; if None all models in /models are used, if 1 a cINN object is returned
    :param model_index: return explicitly the model oder Ensemble of models with the given (list of) index; overwrites the num_models option
    """
    
    from scripts.model.combined_model import CombinedModel 
    
    #Check if gpu or cpu only is available
    device = torch.device(c.device)
    
    def load_single(i):
        cinn = CombinedModel()
        cinn.to(device)
        state_dict = {k:v for k,v in torch.load('model/cinn_' + str(i) + '.pt', map_location=device).items() if 'tmp_var' not in k}
        cinn.load_state_dict(state_dict)
        cinn.eval()
        return cinn
        
    #Look for model sin the model path
    filelist = glob.glob(c.model_path + "/cinn_*.pt")
    
    #Check if a specific index is asked for
    if model_index is not None:
        if len(model_index) > 1:
            models = []
            for i in model_index:
                models.append(load_single(i))
            
            return Ensemble(models)
    
        else:
            return load_single(model_index)  
    
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
    
def load_resnet_model():
    """Load resnet model from the model folder"""
    
    from scripts.model.resnet_simclr import ResNetSimCLR 
    
    model = ResNetSimCLR()
    model.to(c.device)
    checkpoint = torch.load(c.resnet_path, map_location=torch.device(c.device))
    model.load_state_dict(checkpoint)
    model.eval()
    return model