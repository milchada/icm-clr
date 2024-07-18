import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
from scripts.data import data

class RandomGaussianNoise(object):
    def __init__(self, std_limits):
        self.std_limits = std_limits
        
    def __call__(self, tensor):
        std = np.random.uniform(self.std_limits[0], self.std_limits[1])
        return tensor + torch.randn(tensor.size()) * std
    
    def __repr__(self):
        return self.__class__.__name__

class ClipFaintPixels(object):
    """
    Sets all the values below min(random.uniform(), clip_min_ to 0.
    
    Args:
    - image: Input image tensor (assuming it's a PyTorch tensor).
    - clip_min: Threshold value for clipping.
    
    Returns:
    - Clipped image tensor.
    """
    def __init__(self, clip_min):
        self._clip_min = clip_min

    def __call__(self, tensor):
        delta =  min(self._clip_min, np.random.uniform()) 
        print(delta)
        clipped_image = torch.clamp(tensor, min = delta)
        return clipped_image
    
class Augmentation(object):
    def __init__(self, params=None):
        self.transforms_list = []
        self._set_up(params)
        
    def _append_transform(self, x):
        self.transforms_list.append(x)
        
    def _to_image(self):
        self._append_transform(transforms.ToPILImage())
        
    def _to_tensor(self):
        self._append_transform(transforms.ToTensor())
        
    def _resize(self):
        self._append_transform(transforms.Resize(data.IMAGE_SIZE))
        
    def _random_affine(self, rotation, translate, scale):
        '''
        Add a random affine transform to the augmentation
        
        :param rotation: rotate the image around this angle (in + AND - direction)
        :param translate: translate the image as fraction of the total image size
        :param scale: scale into the image with this factor
        '''
        assert rotation >= 0
        assert translate <= 1
        assert translate >= 0
        assert scale >= 1
        self._append_transform(transforms.RandomAffine((-rotation,rotation), translate=(translate,translate), scale=(1,scale)))

    def _random_flip(self, flip_image):
        if flip_image:
            self._append_transform(transforms.RandomHorizontalFlip())
            self._append_transform(transforms.RandomVerticalFlip())
        else:
            print("AUGMENTATION: Flipping switched off")
    
    def _random_noise(self, std_limits):
        assert std_limits[1] >= std_limits[0]
        if std_limits[1] > 0.:
            gaussian_noise = RandomGaussianNoise(std_limits)
            self._append_transform(gaussian_noise)
        else:
            print("AUGMENTATION: Noise switched off")

    def _clip(self, clip_min):
        if clip_min:
            clip = ClipFaintPixels(clip_min)
            self._append_transform(clip)
        
    def _gaussian_blur(self, gaussian_blur_sigma):
    
        assert gaussian_blur_sigma[1] >= gaussian_blur_sigma[0]
    
        def round_up_to_odd(f):
            return np.ceil(f) // 2 * 2 + 1
        
        if gaussian_blur_sigma[1] > 0.:
            gaussian_blur = transforms.GaussianBlur(round_up_to_odd(data.IMAGE_SIZE*0.1), sigma=gaussian_blur_sigma)
            self._append_transform(gaussian_blur)
        else:
            print("AUGMENTATION: Gaussian Blur switched off")
    
    def _set_up(self, params):
        '''Function to set up the augmentation as desired'''
        pass
        
    def get_transforms(self):
        return transforms.Compose(self.transforms_list)
        
class DefaultAugmentation(Augmentation):
    def _set_up(self, params):
        '''Just convert the images into a resized tensor'''
        self._to_image()
        self._resize()
        self._to_tensor()
        
class FlipAugmentation(Augmentation):
    def _set_up(self, params):
        '''Flip image with 50% probability if FLIP:True in params'''
        self._to_image()
        self._resize()
        self._random_flip(params["FLIP"])
        self._to_tensor()

class SimCLRAugmentation(Augmentation):
    def _set_up(self, params):
        '''Augmentations for the simclr training'''
        self._to_image()
        self._random_affine(params["ROTATION"], params["TRANSLATE"], params["SCALE"])
        self._resize()
        self._random_flip(params["FLIP"])
        self._gaussian_blur(params["GAUSSIAN_BLUR_SIGMA"]) #oh this is already implemented
        self._to_tensor()
        self._random_noise(params["NOISE_STD"])
        self._clip(params["CLIP_MIN"])
