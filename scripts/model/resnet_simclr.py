import torch.nn as nn
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
        for i in range(params["RESNET_REPRESENTATION_DEPTH"]):
            self.resnet.add_module("ResNet_ReLU_Representation_" + str(i), nn.ReLU())
            self.resnet.add_module("ResNet_Linear_Representation_" + str(i), nn.Linear(params["RESNET_PROJECTION_DIM"], params["RESNET_PROJECTION_DIM"]))
    
        # We also need an projection head for the contrastive learning
        self.projection = nn.Sequential()
        
        for i in range(params["RESNET_PROJECTION_DEPTH"]):
            self.projection.add_module("ResNet_ReLU_Projection_" + str(i), nn.ReLU())
            if i == 0:
                self.projection.add_module("ResNet_Linear_Projection_" + str(i), nn.Linear(params["RESNET_REPRESENTATION_DIM"], params["RESNET_PROJECTION_DIM"]))
            else:
                self.projection.add_module("ResNet_Linear_Projection_" + str(i), nn.Linear(params["RESNET_PROJECTION_DIM"], params["RESNET_PROJECTION_DIM"]))
        
    @property
    def trainable_parameters(self):
        """Return trainable parameters"""
        trainable_parameters = [p for p in self.resnet.parameters() if p.requires_grad]
        trainable_parameters += [p for p in self.projection.parameters() if p.requires_grad]
        
        return trainable_parameters
        
    def forward(self, x, projection_head=True):
        
        if projection_head:
            return self.projection(self.resnet(x))
        else:
            return self.resnet(x)
        
class ResNetSimCLRVAE(ResNetSimCLR):
    
    def __init__(self, params={}):
        
        super(ResNetSimCLRVAE, self).__init__()
        
        self.linear_mu = nn.Linear(params["RESNET_REPRESENTATION_DIM"], params["RESNET_REPRESENTATION_DIM"])
        self.linear_sigma = nn.Linear(params["RESNET_REPRESENTATION_DIM"], params["RESNET_REPRESENTATION_DIM"])
        self.N = torch.distribution.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
        
    def forward(self, x, projection_head=True):
        
        if projection_head:
            mu = self.linear_mu(self.resnet(x))
            sigma = torch.exp(self.linear_sigma(self.resnet(x))
            z = mu + sigma*self.N.sample(mu.shape)
            self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
            return self.projection(z)
        
        else:
            return self.resnet(x)
        
        
if __name__ == "__main__":
    #from torchinfo import summary
    import torch
    IMAGE_SIZE = yaml.safe_load(open('params.yaml'))['data']['IMAGE_SIZE']
    model = ResNetSimCLR()
    input_size = torch.rand(1, model_default_params["RESNET_NUM_CHANNELS"], IMAGE_SIZE, IMAGE_SIZE)
    #summary(model, input_size=input_size, depth=1, col_names=['num_params'])
    
    #print(model)
    
    #print(model.state_dict())
    
    #out = model(input_size)
    
    
    #for name, layer in model.named_modules():
    #    print(name)
    #    #print(layer)
    #    try:
    #        print(model._modules[name].in_features)
    #        print(model._modules[name].out_features)
    #    except:
    #        pass
    #    print("---------------------------------------------------")
