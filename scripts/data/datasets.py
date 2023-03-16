import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
import yaml

data_params = yaml.safe_load(open('params.yaml'))['data']

from scripts.data.augmentations import DefaultAugmentation

class SimClrDataset(Dataset):
    """Dataset to load and transform images with multiple transformed views in a multithreaded and batched way"""

    def __init__(self, image_file, label_file=None, augmentation=None, n_views=1):
        """
        Args:
            image_file (string or list of strings): Path to the csv file with "image_path" as field.
            label_file (string or list of strings, optional): Path to the csv file with labels to return (If not set, only images are returned).
            augmentation (Augmentation, optional): use the augmenations (Default: None i.e. Default Augmentation).
            n_views (integer, optional): Number of views to return for each image (Default: 1).
        """
        
        #Load image paths
        if isinstance(image_file, str):
            self.df = pd.read_csv(image_file)
        elif isinstance(image_file, list):
            df_list = []
            
            for s in image_file:
                if isinstance(s, str):
                    df_list.append(pd.read_csv(s))
                else:
                    raise TypeError()
                    
            self.df = pd.concat(df_list, ignore_index=True)
        else:
            raise TypeError()
        
        self.image_paths = self.df['image_path']
        
        #Load labels if given
        if label_file is None:
            self.df_label = None
        elif isinstance(label_file, str):
            self.df_label = pd.read_csv(label_file)
            assert len(self.df) == len(self.df_label), "Error: Number of rows in the image and label csv have to be equal!"
        elif isinstance(label_file, list):
            df_list = []
            for s in label_file:
                if isinstance(s, str):
                    df_list.append(pd.read_csv(s))
                else:
                    raise TypeError()
            self.df_label = pd.concat(df_list, ignore_index=True)
            assert len(self.df) == len(self.df_label), "Error: Number of rows in the image and label csv have to be equal!"
        else:
            raise TypeError()
        
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
            image = self._get_single_image_index(idx)
        except OSError:
            print("Problems reading " + str(self.image_paths[idx]))
            return
        
        samples = [self.transform(image) for i in range(self.n_views)]
            
        if self.df_label is None:
            return samples
        else:
            label = torch.tensor(self.df_label.iloc[idx, :].to_numpy())
            return samples, label
        
    def _get_single_image_index(self, idx):
        '''Return one single image with the id given as idx'''
        raise NotImplementedError("This function is supposed to be overwritten!")
        
class FitsDataset(SimClrDataset):
    """Dataset to load and transform .fits images with multiple transformed views in a multithreaded and batched way"""
    
    def _get_single_image_index(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self._get_single_image_path(self.image_paths[idx])
            
    def _get_single_image_path(self, path):
            
        with fits.open(path) as hdul:
            
            filter_list = []
            
            for f in data_params['FILTERS']:
                filter_list.append(hdul[f].data)
            
        stretched_filter_list = self._stretch(filter_list)

        stretched_filter_list = [np.array(f*(2**8 - 1), dtype=np.uint8) for f in stretched_filter_list]
        
        return np.concatenate(list(map(lambda x: x[...,np.newaxis], stretched_filter_list)), axis=2)
    
    def _stretch(self, channels):
        """Stretch channels"""
        raise NotImplementedError("This function is supposed to be overwritten!")

class ConnorStretchDataset(FitsDataset):
    
    def _stretch_single_channel(self, x):
        u_min = -0.05
        u_max = 2. / 3.
        u_a = np.exp(10.)
        x = np.arcsinh(u_a*x) / np.arcsinh(u_a)
        x = (x - u_min) / (u_max - u_min)
        return x
    
    def _stretch(self, channels, ref_mag=26):
        channels = [c*10**(0.4*(22.5-ref_mag)) for c in channels]
        return list(map(lambda x: self._stretch_single_channel(x), channels))

        
class SingleStretchDataset(FitsDataset):
    '''Dataset for the HSC Realistic TNG Images'''
    
    def _get_central_crop(self, img, num_pixel=20):
        size = img.shape[0]
        center_coordinate = size//2
        upper = center_coordinate + num_pixel//2
        lower = center_coordinate - num_pixel//2
    
        return img[lower:upper, lower:upper]
    
    def _stretch_single_channel(self, x):
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
    
    def _stretch(self, channels):
        return list(map(lambda x: self._stretch_single_channel(x), channels))
    
        
class MultiStretchDataset(SingleStretchDataset):

    def _stretch(self, channels):
        """Stretch all channels together to preserve the color"""
        
        channels = [np.clip(c, 0, None) for c in channels]
        
        i = (np.sum(channels, axis=0))/len(channels)
        
        factor = self._stretch_single_channel(i)/i
        factor = np.nan_to_num(factor, posinf=0.0)
        
        channels = [c*factor for c in channels]
        
        max_value = np.max(channels, axis=0)
        max_mask = max_value > 1
        
        for c in channels:
            c[max_mask] /= max_value[max_mask]
        
        return channels
