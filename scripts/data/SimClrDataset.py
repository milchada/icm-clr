import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
from .transforms import get_default_transforms, get_transforms
import yaml

class SimClrDataset(Dataset):
    """Dataset to load and transform images given as .fits"""

    def __init__(self, image_file, label_file=None, transform=False, n_views=1):
        """
        Args:
            image_file (string): Path to the csv file with "image_path" as field.
            label_file (string, optional): Path to the csv file with labels to return (If not set, only images are returned).
            transform (boolean, optional): Apply random transformations at each call (Default: False).
            n_views (integer, optional): Number of views to return for each image (Default: 1).
        """
        
        self.df = pd.read_csv(image_file)
        self.image_paths = self.df['image_path']
        
        if label_file is None:
            self.df_label = None
        else:
            self.df_label = pd.read_csv(label_file)
            assert len(self.df) == len(self.df_label), "Error: Number of rows in the image and label csv have to be equal!"
        
        self.transform = transform
        self.n_views = n_views
        
        #Load desired image size from the parameter file (pixel per side)
        data_params = yaml.safe_load(open('params.yaml'))['data']
        self.image_size = data_params["IMAGE_SIZE"]
        
        assert n_views >= 1
        
        #If we want to return labels assure that only one image is returned per label
        if label_file is not None:
            assert n_views == 1

    def __len__(self):
        return len(self.df)
    
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
        
    def _get_single_image(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with fits.open(self.image_paths[idx]) as hdul:
            
            G = hdul['SUBARU_HSC.G'].data
            R = hdul['SUBARU_HSC.R'].data
            I = hdul['SUBARU_HSC.I'].data
        
        I, R, G = self._rgbstretch(I,R,G)

        G = np.array(G*(2**8 - 1), dtype=np.uint8)
        R = np.array(R*(2**8 - 1), dtype=np.uint8)
        I = np.array(I*(2**8 - 1), dtype=np.uint8)
        
        return np.concatenate((I[...,np.newaxis],R[...,np.newaxis],G[...,np.newaxis]),axis=2)
    
    def __getitem__(self, idx):

        image = self._get_single_image(idx)
    
        if self.transform:
            samples = [get_transforms(self.image_size)(image) for i in range(self.n_views)]
        else:
            samples = [get_default_transforms(self.image_size)(image) for i in range(self.n_views)]
            
        if self.df_label is None:
            return samples
        else:
            label = torch.tensor(self.df_label.iloc[idx, :].to_numpy())
            return samples, label
        
    
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from .transforms import get_transforms
    
    n_galaxies=20
    n_views=2
    
    for i in range(n_galaxies):
    
        dataset = SimClrDataset("./dataset/m_train.csv", label_file="./dataset/x_train.csv")
        print(dataset[i])
        image = dataset[i][0][0]
        label = dataset[i][1]
        print(label)
        print(image.shape)
        image = np.swapaxes(image, 0, 2)
        plt.imshow(image)
        plt.savefig('./temp/SimClrDataset_' + str(i) + '.png')

        dataset = SimClrDataset("./dataset/m_train.csv",
                                transform=True,
                                n_views=n_views)

        for j in range(n_views):
            image = dataset[i][j]
            print(image.shape)
            image = np.swapaxes(image, 0, 2)
            plt.imshow(image)
            plt.savefig('./temp/SimClrDataset_' + str(i) + '_' + str(j) + '.png')
    