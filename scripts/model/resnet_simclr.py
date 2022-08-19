import torch.nn as nn
#from torchsummary import summary
from scripts.model.resnet import Wide_ResNet

import yaml

#Load the default params from the parameter file
model_default_params = yaml.safe_load(open('params.yaml'))['model']

class ResNetSimCLR(nn.Module):

    def __init__(self, params={}):
        
        super(ResNetSimCLR, self).__init__()
        
        #Use default parameter if not set
        params = dict(model_default_params, **params)
        
        #Get equivariant CNN model
        self.resnet = Wide_ResNet(params["RESNET_DEPTH"],
                                  params["RESNET_WIDTH"],
                                  params["RESNET_DROPOUT"],
                                  num_classes=params["RESNET_REPRESENTATION_DIM"],
                                  num_channels=params["RESNET_NUM_CHANNELS"],
                                  N=params["RESNET_ROTATION_EQUIVARIANCE"],
                                  f=params["RESNET_REFLECTION_EQUIVARIANCE"],
                                  r=params["RESNET_ROTATION_RESTRICTION"],
                                  initial_stride=params["RESNET_INITIAL_STRIDE"])
        
        #Add Linear layer for the representation
        self.resnet.add_module("ResNet_ReLU_Representation_1", nn.ReLU())
        self.resnet.add_module("ResNet_Linear_Representation_1", nn.Linear(params["RESNET_REPRESENTATION_DIM"], params["RESNET_REPRESENTATION_DIM"]))
        self.resnet.add_module("ResNet_ReLU_Representation_2", nn.ReLU())
        self.resnet.add_module("ResNet_Linear_Representation_2", nn.Linear(params["RESNET_REPRESENTATION_DIM"], params["RESNET_REPRESENTATION_DIM"]))
        self.resnet.add_module("ResNet_ReLU_Representation_3", nn.ReLU())
        self.resnet.add_module("ResNet_Linear_Representation_3", nn.Linear(params["RESNET_REPRESENTATION_DIM"], params["RESNET_REPRESENTATION_DIM"]))
        
        # add mlp projection head
        self.projection = nn.Sequential()
        self.projection.add_module("ResNet", self.resnet)
        
        for i in range(params["RESNET_PROJECTION_DEPTH"]):
            self.projection.add_module("ResNet_ReLU_" + str(i), nn.ReLU())
            if i == 0:
                self.projection.add_module("ResNet_Linear_" + str(i), nn.Linear(params["RESNET_REPRESENTATION_DIM"], params["RESNET_PROJECTION_DIM"]))
            else:
                self.projection.add_module("ResNet_Linear_" + str(i), nn.Linear(params["RESNET_PROJECTION_DIM"], params["RESNET_PROJECTION_DIM"]))
        
        #summary(self.projection.cuda(), (params["RESNET_NUM_CHANNELS"], 128, 128))
        
    @property
    def trainable_parameters(self):
        """Return trainable parameters"""
        trainable_parameters = [p for p in self.projection.parameters() if p.requires_grad]
        
        return trainable_parameters
        
    def forward(self, x, projection_head=True):
        
        if projection_head:
            return self.projection(x)
        else:
            return self.resnet(x)
        