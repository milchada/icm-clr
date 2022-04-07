import torch.nn as nn
from scripts.model.e2wrn import Wide_ResNet

class ResNetSimCLR(nn.Module):

    def __init__(self, depth,
                       widen_factor,
                       dropout_rate,
                       representation_dim,
                       projection_dim,
                       num_channels):
        
        super(ResNetSimCLR, self).__init__()

        # get main equivariant cnn model
        self.resnet = Wide_ResNet(depth,
                                  widen_factor,
                                  dropout_rate,
                                  num_classes=representation_dim,
                                  num_channels=num_channels)
        

        # add mlp projection head
        self.projection = nn.Sequential(self.resnet,
                                        nn.ReLU(),
                                        nn.Linear(representation_dim, projection_dim))
        


    def forward(self, x, projection_head=True):
        
        if projection_head:
            return self.projection(x)
        else:
            return self.resnet(x)
        