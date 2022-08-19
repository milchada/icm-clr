import unittest

import numpy as np
from astropy.io import fits
import glob

from scripts.preprocessing.cropper import ImageCropper, FractionalCropper, PetrosianCropper


class TestCropper(unittest.TestCase):
    '''Consistency test for the cropper classes'''
    
    def setUp(self):
        
        self.filelist = glob.glob("./scripts/preprocessing/test/test_images/**.fits")
        self.images = []
        
        for f in self.filelist:
            with fits.open(f) as hdul:
                self.images.append(hdul['SUBARU_HSC.R'].data)
        
    def test_image_cropper(self):
        target_size = 128
        cropper = ImageCropper(image_target_size=target_size)
        
        cropped_image = cropper(self.images[0])
        self.assertEqual(cropped_image.shape[0], target_size, 'incorrect target size')
        
    def test_petrosian_cropper(self):
        target_size = 128
        cropper = PetrosianCropper(image_target_size=target_size)
        
        for img in self.images:
            median_before_fit = np.nanmedian(img)
            cropper.fit(img)
            median_after_fit = np.nanmedian(img)
            self.assertTrue(median_before_fit == median_after_fit, 'fit changes the image values')
            
            print(cropper.valid)
            print(cropper.r_half_light)
            print(cropper.r_90_light)
            
            cropped_image = cropper(img)
            
            self.assertFalse(np.isnan(cropper.r_half_light), 'half light rad is nan')
            self.assertFalse(np.isnan(cropper.r_90_light), '90 light rad is nan')
            self.assertEqual(cropped_image.shape[0], target_size, 'incorrect target size')

if __name__ == '__main__':
    unittest.main()