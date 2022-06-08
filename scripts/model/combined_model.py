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

        #Load model architectures
        self.resnet = ResNetSimCLR(params)
        self.cinn = cINN(params)
         
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)   

    @property
    def trainable_parameters(self):
        """Return trainable parameters"""
        trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        trainable_parameters += [p for p in self.resnet.parameters() if p.requires_grad]
        
        return trainable_parameters
    
    @property
    def num_trainable_parameters(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.trainable_parameters)
    
    def load_pretrained_resnet(self, path):
        """Load the simclr pretrained resnet"""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.resnet.load_state_dict(checkpoint)
        
    def fix_resnet_weights(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
            
    def unfix_resnet_weights(self):
        for param in self.resnet.parameters():
            param.requires_grad = True
            
    def forward(self, x, l):
        z, jac = self.cinn(x.to(torch.float16), l=self.resnet(l, projection_head=False).to(torch.float16))
        return z, jac

    def reverse_sample(self, z, l):
        return self.cinn.reverse_sample(z.to(torch.float16), l=self.resnet(l, projection_head=False).to(torch.float16)).to(torch.float64)
    