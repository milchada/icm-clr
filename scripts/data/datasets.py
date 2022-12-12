import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
import yaml

from scripts.data.augmentations import DefaultAugmentation

class SimClrDataset(Dataset):
    """Dataset to load and transform images with multiple transformed views in a multithreaded and batched way"""

    def __init__(self, image_file, label_file=None, augmentation=None, n_views=1):
        """
        Args:
            image_file (string): Path to the csv file with "image_path" as field.
            label_file (string, optional): Path to the csv file with labels to return (If not set, only images are returned).
            augmentation (Augmentation, optional): use the augmenations (Default: None i.e. Default Augmentation).
            n_views (integer, optional): Number of views to return for each image (Default: 1).
        """
        
        self.df = pd.read_csv(image_file)
        self.image_paths = self.df['image_path']
        
        #Load labels if given
        if label_file is None:
            self.df_label = None
        else:
            self.df_label = pd.read_csv(label_file)
            assert len(self.df) == len(self.df_label), "Error: Number of rows in the image and label csv have to be equal!"
        
        #Load the transforms from the augmentation object
        if augmentation is None or augmentation is False:
            self.transform = DefaultAugmentation().get_transforms()
        else:
            self.transform = augmentation.get_transforms()
        
        #Number of views to return
        self.n_views = n_views
        assert n_views >= 1
        
        #If we want to return labels assure that only one image is returned per label
        if label_file is not None:
            assert n_views == 1

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        '''Return an item with index idx'''

        try:
            image = self._get_single_image(idx)
        except OSError:
            print("Problems reading " + str(self.image_paths[idx]))
            return
        
        samples = [self.transform(image) for i in range(self.n_views)]
            
        if self.df_label is None:
            return samples
        else:
            label = torch.tensor(self.df_label.iloc[idx, :].to_numpy())
            return samples, label
        
    def _get_single_image(self, idx):
        '''Return one single image with the id given as idx'''
        raise NotImplementedError("This function is supposed to be overwritten!")
        
class FitsDataset(SimClrDataset):
    """Dataset to load and transform .fits images with multiple transformed views in a multithreaded and batched way"""
    
    def _get_single_image(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with fits.open(self.image_paths[idx]) as hdul:
            
            G = hdul['G'].data
            R = hdul['R'].data
            I = hdul['I'].data
        
        I, R, G = self._rgbstretch(I,R,G)

        G = np.array(G*(2**8 - 1), dtype=np.uint8)
        R = np.array(R*(2**8 - 1), dtype=np.uint8)
        I = np.array(I*(2**8 - 1), dtype=np.uint8)
        
        return np.concatenate((I[...,np.newaxis],R[...,np.newaxis],G[...,np.newaxis]), axis=2)
    
    def _rgbstretch(self, r, g, b):
        """Stretch rgb channels"""
        raise NotImplementedError("This function is supposed to be overwritten!")

class HSCDataset(FitsDataset):
    
    def _stretch(self, x):
        u_min = -0.05
        u_max = 2. / 3.
        u_a = np.exp(10.)
        x = np.arcsinh(u_a*x) / np.arcsinh(u_a)
        x = (x - u_min) / (u_max - u_min)
        return x
    
    def _rgbstretch(self, r, g, b, ref_mag=26):
        r *= 10**(0.4*(22.5-ref_mag))
        g *= 10**(0.4*(22.5-ref_mag))
        b *= 10**(0.4*(22.5-ref_mag))
        return self._stretch(r), self._stretch(g), self._stretch(b)

        
class TNGDataset(FitsDataset):
    '''Dataset for the HSC Realistic TNG Images'''
    
    def _get_central_crop(self, img, num_pixel=20):
        size = img.shape[0]
        center_coordinate = size//2
        upper = center_coordinate + num_pixel//2
        lower = center_coordinate - num_pixel//2
    
        return img[lower:upper, lower:upper]
    
    def _stretch(self, x):
        """Perform a log stretch on x and normalize"""
        x[x<=0] = np.nan
        x = np.log10(x)
        x[x<-7] = np.nan
        
        a_min = np.nanmedian(x)
        a_max = np.nanquantile(self._get_central_crop(x), 0.99)
        
        x = np.nan_to_num(x, nan=a_min, posinf=a_min, neginf=a_min)
        x = np.clip(x, a_min, a_max)
        
        x -= a_min
        x /= (a_max - a_min)
        
        return x
    
    def _rgbstretch(self, r, g, b):
        """Stretch rgb together to preserve the color"""
        
        r = np.clip(r, 0, None)
        g = np.clip(g, 0, None)
        b = np.clip(b, 0, None)
        
        i = (r + g + b)/3
        
        factor = self._stretch(i)/i
        factor = np.nan_to_num(factor, posinf=0.0)
        
        r *= factor
        g *= factor
        b *= factor
        
        max_value = np.max([r,g,b], axis=0)
        max_mask = max_value > 1
        
        r[max_mask] /= max_value[max_mask]
        g[max_mask] /= max_value[max_mask]
        b[max_mask] /= max_value[max_mask]
        
        return r, g, b
    
class TNGIdealDataset(TNGDataset):
    '''Dataset for the ideal TNG Images'''
    
    def _stretch(self, x):
        """Perform a linear stretch""" 
        x = np.clip(x, -32, None)
        
        a_min = np.min(x)
        a_max = np.max(x)
        
        x -= a_min
        x /= (a_max - a_min)
        
        return x
    
    def _rgbstretch(self, r, g, b):
    
        r = -r
        g = -g
        b = -b
        
        r = self._stretch(r)
        g = self._stretch(g)
        b = self._stretch(b)
        
        return r, g, b