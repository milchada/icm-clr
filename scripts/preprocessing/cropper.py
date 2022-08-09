'''

Classes to crop a galaxy image in various ways

'''

import numpy as np

from skimage.transform import resize

from petrofit.segmentation import make_catalog, plot_segments
from astropy.stats import sigma_clipped_stats
from petrofit.photometry import source_photometry
from petrofit.photometry import make_radius_list
from petrofit.petrosian import Petrosian

class ImageCropper(object):
    
    def __init__(self, image_target_size=256):
        self._image_target_size = image_target_size
        
    def _get_center(self, image):
        shape = image.shape
        center_x = shape[0]//2
        center_y = shape[1]//2
        return center_x, center_y
    
    def _get_crop_size(self, image)
        shape = image.shape
        return np.min(shape)
    
    def _get_central_square_crop(image, num_pixel):

        num_pixel_half = num_pixel//2
        
        center_x, center_y = self._get_center(image)
        min_x = center_x - num_pixel_half
        max_x = center_x + num_pixel_half
        min_y = center_y - num_pixel_half
        max_y = center_y + num_pixel_half
        
        return image[min_x:max_x, min_y:max_y]
        
    def _resize(self, image):
        return resize(image, (self._image_target_size, self._image_target_size))
        
    def __call__(self, image)
        crop_size = self._get_crop_size(image)
        image = self._central_square_crop(image, crop_size)
        image = self._resize(image)
        return image
    
class FractionalCropper(ImageCropper):
    
    def __init__(self, crop_fraction=1.0, enlarge_image=False, **kwargs):
        super().__init__(**kwargs)
        self.crop_fraction = crop_fraction
        self._enlarge_image = enlarge_image
        
    @property
    def crop_fraction(self):
        return self._crop_fraction 
    
    @crop_fraction.setter
    def crop_fraction(self, value):
        assert value > 0.
        self._crop_fraction = value
        
    def _get_crop_size(self, image)
        target_size = super()._get_crop_size(image)
        fraction_size = self.crop_fraction * target_size
        
        if self.crop_fraction <= 1.0:
            return fraction_size
        elif self._enlarge_image:
            raise NotImplementedError("The enlargement of images has still to be implemented.")
        else:
            return target_size
    
    
class PetrosianCropper(ImageCropper):

    def __init__(self, petro_multiplier=4, **kwargs):
        super().__init__(**kwargs)
        
        self.petro_multiplier = petro_multiplier
        self._r_half_light = np.nan
        self._r_90_light = np.nan
        
    @property
    def r_half_light(self):
        return self._r_half_light
            
    @property
    def r_90_light(self):
        return self._r_90_light
        
    @property
    def petro_multiplier(self):
        return self._petro_multiplier
    
    @petro_multiplier.setter
    def petro_multiplier(self, value):
        assert value > 0.
        self._petro_multiplier = value
    
    
    def _stretch(self, x):
        """Perform a stretch on x and normalize"""
        x[x<=0] = np.nan

        a_min = np.nanmedian(x)
        a_max = np.nanquantile(self._get_central_square_crop(x, 20), 0.99)

        x = np.nan_to_num(x, nan=a_min, posinf=a_min, neginf=a_min)
        x = np.clip(x, a_min, a_max)

        x -= a_min
        x /= (a_max - a_min)

        return x
    
    def fit(self, image, mode='most_central'):
        
        #Clip and normalize image before fitting
        image = self._stretch(image)
        
        #Get a catalog from the given image (i.e. detect sources)
        image_mean, image_median, image_stddev = sigma_clipped_stats(image, sigma=3)
        
        cat, segm, segm_deblend = make_catalog(image=image,  # Input image
                                               threshold=image_stddev*3,  # Detection threshold
                                               deblend=True,  # Deblend sources?
                                               kernel_size=3,  # Smoothing kernel size in pixels
                                               fwhm=3,  # FWHM in pixels
                                               npixels=4**2  # Minimum number of pixels that make up a source)
        
        #Get as source the object with the highest flux or the most central one
        if mode == 'max_flux':
                                               
            i = np.argmax(cat.kron_flux)
                                               
        elif mode == 'most_central':
        
            source_x = cat.xcentroid
            source_y = cat.ycentroid

            center_x, center_y = self._get_center(image)
            distance = np.sqrt((source_x-center_x)**2 + (source_y-center_y)**2)
            i = np.argmin(distance)
                                               
        else:
            raise ValueError("Unknown fit selection mode")
                                               
        source = cat[i]
        
        #Choose the aperature to look at (choose from source bbox)
        bbox_size = np.min((source.bbox_xmax - source.bbox_xmin,
                            source.bbox_ymax - source.bbox_ymin))                                
                                               
        r_list = make_radius_list(max_pix=bbox_size//2, # Max pixel to go up to
                                  n=bbox_size//2 # the number of radii to produce)

        #Perform Photometry
        flux_arr, area_arr, error_arr = source_photometry(

                                                # Inputs
                                                source, # Source (`photutils.segmentation.catalog.SourceCatalog`)
                                                image, # Image as 2D array
                                                segm_deblend, # Deblended segmentation map of image
                                                r_list, # list of aperture radii

                                                # Options
                                                cutout_size=max(r_list)*2, # Cutout out size, set to double the max radius
                                                bkg_sub=True, # Subtract background
                                                sigma=3, sigma_type='clip' # Fit a 2D plane to pixels within 3 sigma of the mean)
        
        #Do Petrosian stuff
        p = Petrosian(r_list, area_arr, flux_arr)
            
        self._r_half_light = p.r_half_light
        self._r_90_light = p.fraction_flux_to_r(fraction=0.9)
            
        return self._r_half_light, self._r_90_light
            
            
        def _get_crop_size(self, image)
            return self.r_90_light * self.petro_multiplier
