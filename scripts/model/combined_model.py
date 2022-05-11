import torch.nn as nn
from scripts.model.resnet_simclr import ResNetSimCLR
from scripts.model.cinn import cINN
import torch

import yaml

#Load the default params from the parameter file
model_default_params = yaml.safe_load(open('params.yaml'))['model']

class CombinedModel(nn.Module):
    
    def __init__(self, params={}):
        super().__init__()
        
        #Use default parameters if not set
        params = dict(model_default_params, **params)

        self.resnet = ResNetSimCLR(params)
        self.cinn = cINN(params)

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        self.trainable_parameters += [p for p in self.resnet.parameters() if p.requires_grad]
               
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)
            
    def load_pretrained_resnet(self, path):
        """Load the with simclr pretrained resnet"""
        checkpoint = torch.load(path)
        self.resnet.load_state_dict(checkpoint)
            
    def forward(self, x, l):
        z, jac = self.cinn(x, c=self.resnet(l, projection_head=False), rev=False, jac=True)
        return z, jac

    def reverse_sample(self, z, l):
        return self.cinn(z, c=self.resnet(l, projection_head=False), rev=True, jac=False)[0]
    