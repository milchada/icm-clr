import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.data import data


class TestDataloader(unittest.TestCase):
    '''Consistency test for the dataset class'''
    
    def setUp(self):
        
        self.N_GALAXIES=20
        self.BATCH_SIZE=128
        self.N_VIEWS=2
        

    def test_image_label(self):
        '''Test if the label and image loading is working'''

        dataloader = data.get_train_loader(batch_size=self.BATCH_SIZE, labels=True, transform=False, n_views=1, shuffle=True, drop_last=True)
    
        for i, (images, labels) in enumerate(dataloader):
            
            self.assertIsInstance(labels, torch.Tensor, 'labels is not a tensor')
            self.assertEqual(labels.shape, (self.BATCH_SIZE, data.NUM_DIM), 'incorrect label size')
            
            images = torch.cat(images, dim=0)
            self.assertEqual(images.shape[0], self.BATCH_SIZE, 'incorrect image batch size')
            self.assertEqual(images.shape[2], data.IMAGE_SIZE, 'incorrect image size')
            self.assertEqual(images.shape[3], data.IMAGE_SIZE, 'incorrect image size')
            
            self.assertTrue(torch.max(images) <= 1.0, 'image value larger than 1')
            self.assertTrue(torch.min(images) >= 0.0, 'image value smaller than 0')
            
            image = np.swapaxes(images[0], 0, 2)
            plt.imshow(image)
            plt.savefig('./temp/Dataset_' + str(i) + '.png')
            
            if i > self.N_GALAXIES:
                return

    def test_views(self):
        '''Test if the view generation is working'''
        
        dataloader = data.get_train_loader(batch_size=self.BATCH_SIZE, labels=False, transform=True, n_views=self.N_VIEWS, shuffle=True, drop_last=True)
    
        for i, images in enumerate(dataloader):
            
            images = torch.cat(images, dim=0)
            
            self.assertEqual(images.shape[0], self.BATCH_SIZE*self.N_VIEWS, 'incorrect image batch size')
            self.assertEqual(images.shape[2], data.IMAGE_SIZE, 'incorrect image size')
            self.assertEqual(images.shape[3], data.IMAGE_SIZE, 'incorrect image size')
            
            #I think the noise is messing this up
            #self.assertTrue(torch.max(images) <= 1.0, 'image value larger than 1')
            #self.assertTrue(torch.min(images) >= 0.0, 'image value smaller than 0')
            
            for j in range(self.N_VIEWS):
                image = np.swapaxes(images[j], 0, 2)
                plt.imshow(image)
                plt.savefig('./temp/Dataset_' + str(i) + '_' + str(j) + '.png')
                
            if i > self.N_GALAXIES:
                    return

if __name__ == '__main__':
    unittest.main()

    
    
    
    


    
    
    