import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms

def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1

class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_transforms(size, random_crop_scale=(0.2, 1.0), gaussian_blur_sigma=(0.1, 1.0)):
    
    random_affine = transforms.RandomAffine((-10,10), translate=(0.1,0.1), scale=(1,3))
    resize = transforms.Resize(size)
    gaussian_blur = transforms.GaussianBlur(round_up_to_odd(size*0.1), sigma=gaussian_blur_sigma)
    std = np.random.uniform(0.0, 0.02)
    gaussian_noise = GaussianNoise(std = std)
    
    data_transforms = transforms.Compose([transforms.ToPILImage(),
                                          random_affine,
                                          resize,
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          gaussian_blur,
                                          transforms.ToTensor(),
                                          gaussian_noise])
    
    return data_transforms

def get_default_transforms(size):
 
    data_transforms = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(size),
                                          transforms.ToTensor()])
    
    return data_transforms