import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from astropy.io import fits

class SimClrDataset(Dataset):

    def __init__(self, csv_file, transform=None, n_views=1, labels=False):
        """
        Args:
            csv_file (string): Path to the csv file with the image filename as first field.
            transform (callable, optional): Optional transform to be applied on each sample.
            n_views (integer, optional): Number of views to return for each image; only used if transform is given
            labels: Return labels if true
        """
        self.df = pd.read_csv(csv_file)
        self.image_paths = self.df['image_path']
        self.transform = transform
        self.n_views = n_views
        self.labels = labels
        
        assert n_views >= 1

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
        
        label = np.array(self.df.iloc[idx, :].to_numpy())
    
        if self.transform:
            samples = [self.transform(image) for i in range(self.n_views)]
            
            if not self.labels:
                return samples
            else:
                return {'image': samples, 'label': label}
        
        else: 
            
            if not self.labels:
                return [image]
            else:
                return {'image': [image], 'label': label}
    
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from .transforms import get_transforms
    
    n_galaxies=20
    n_views=1
    
    for i in range(n_galaxies):
    
        dataset = SimClrDataset("./dataset/m_train.csv")
        print(dataset[i])
        image = dataset[i][0]
        print(image.shape)
        plt.imshow(image)
        plt.savefig('./temp/SimClrDataset_' + str(i) + '.png')

        dataset = SimClrDataset("./dataset/m_train.csv",
                                transform=get_transforms(128),
                                n_views=n_views)

        for j in range(n_views):
            image = dataset[i][j]
            print(image.shape)
            image = np.swapaxes(image, 0, 2)
            plt.imshow(image)
            plt.savefig('./temp/SimClrDataset_' + str(i) + '_' + str(j) + '.png')
    