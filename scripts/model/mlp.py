import torch
import torch.nn as nn

import scripts.data as data
import config as c
import numpy as np

import yaml

#Load the default params from the parameter file
model_default_params = yaml.safe_load(open('params.yaml'))['model']

class MLP_Gaussian(nn.Module):
    '''Simple MLP with a head for a gaussian model; i.e. the model predicts mean and variance of the target'''
    def __init__(self, params={}):
        super().__init__()
        
        #Use default parameter if not set
        params = dict(model_default_params, **params)
        
        self.mlp = nn.Sequential()
            
        #Input Layer
        self.mlp.add_module("Input_Linear_MLP", nn.Linear(params["RESNET_REPRESENTATION_DIM"], params["NUM_HIDDEN_NODES_MLP"]))
            
        if params["BATCH_NORM_MLP"]:
            self.mlp.add_module("Input_BN_MLLP", nn.BatchNorm1d(params["NUM_HIDDEN_NODES_MLP"]))
            
        self.mlp.add_module("Input_ReLU_MLP", nn.ReLU())

        #Hidden Layers
        for i in range(params["NUM_HIDDEN_LAYERS_MLP"]):
            self.mlp.add_module("Linear_MLP_" + str(i), nn.Linear(params["NUM_HIDDEN_NODES_MLP"], params["NUM_HIDDEN_NODES_MLP"]))
                
            if params["BATCH_NORM_MLP"]:
                self.mlp.add_module("BN_MLP_" + str(i), nn.BatchNorm1d(params["NUM_HIDDEN_NODES_MLP"]))
                
            self.mlp.add_module("ReLU_MLP_" + str(i), nn.ReLU())
                
            #Apply Dropout only between hidden layers
            if i < params["NUM_HIDDEN_LAYERS_MLP"] and params["DROPOUT_MLP"] > 0.0:
                self.mlp.add_module("Dropout_MLP_" + str(i), nn.Dropout(p=params["DROPOUT_MLP"]))
                
        #Output layer
        self.mlp.add_module("Output_Linear_MLP", nn.Linear(params["NUM_HIDDEN_NODES_MLP"], data.NUM_DIM*2))
            
    def forward(self, c):
        return self.mlp(c)
