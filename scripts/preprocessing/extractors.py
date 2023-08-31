'''
Extractor objects

As soon as __init__ is called, they should load, prepare and store the respective
dataset (labels and images) into the dataset_raw folder. 

Each dataset type has its own object; depending on the special structure of the data or the source of the data
'''

from scripts.preprocessing.TNG.Catalogue import Catalogue
from scripts.preprocessing.cropper import FractionalCropper, PetrosianCropper
from scripts.util.chunked_pool import ChunkedPool
from scripts.preprocessing.caching import Cache
from scripts.util.logging import logger

import os
from tqdm import tqdm
import glob
from astropy.io import fits
import numpy as np
import pandas as pd
import time

import config as c

import yaml

params = yaml.safe_load(open('params.yaml'))
extract_params = params['extract']
PETRO_RADIUS_FACTOR = extract_params['PETRO_RADIUS_FACTOR']
USE_CACHE = extract_params['USE_CACHE']
NUM_WORKERS = extract_params['NUM_WORKERS']
SIZE_CHUNKS = extract_params['SIZE_CHUNKS']
NUM_PROJECTIONS = extract_params['NUM_PROJECTIONS']
LOAD_IMAGES = extract_params['LOAD_IMAGES']

class DataExtractor(object):
    def __init__(self, dataset, min_mass, max_mass, snapshots, fields=None, image_size=None, filters=None, simulation=None, fraction=1.0):
        self._dataset = dataset
        self._min_mass = min_mass
        self._max_mass = max_mass
        self._snapshots = snapshots
        
        self._label_path = os.path.join(c.dataset_raw_path, self._dataset, "label.csv")
        self._image_path = os.path.join(c.dataset_raw_path, self._dataset, "images/")
        self._cache_path = os.path.join(c.dataset_raw_path, self._dataset, "cache.csv")
        
        self.init_cache()
        
        self._fields = fields
        self._image_size = image_size
        self._filters = filters
        self._filters_keys = self.get_filter_keys(filters)
        self._simulation = simulation
        self._fraction = fraction
        
        self.create_paths()
        self.extract()
        
    def get_extractor(dataset, min_mass, max_mass, snapshots, fields=None, image_size=None, filters=None, fraction=1.0):
        '''Static factory method to get the correct DataExtractor object based on the dataset asked for'''
        
        if dataset in {"TNG50-1", "TNG100-1", "TNG300-1"}:
            return TNGDataExtractor(dataset, min_mass, max_mass, snapshots, fields)
        elif dataset in {"HSC_TNG50", "HSC_TNG50_Ideal"}:
            return TNGHSCExtractor(dataset, min_mass, max_mass, snapshots, fields, image_size, filters, simulation="TNG50-1")
        elif dataset in {"HSC_TNG100"}:
            return TNGHSCExtractor(dataset, min_mass, max_mass, snapshots, fields, image_size, filters, simulation="TNG100-1")
        elif dataset in {"HSC"}:
            return HSCDataExtractor(dataset, min_mass, max_mass, snapshots, fields, image_size, filters, None, fraction)
        else:
            raise NotImplementedError(dataset + " has not been implemented yet!")
    
    
    # Stuff to handle creation of paths and files 
    #--------------------------------------------------------------------------  
    
    def create_paths(self):        
        '''Create paths if not already existing'''
        if not os.path.exists(self._image_path):
            os.makedirs(self._image_path)
            
    def load_labels(self):
        return pd.read_csv(self._label_path)
    
    def save_labels(self, df):
        df["dataset"] = self._dataset
        df.to_csv(self._label_path, index=False)
        
        
    # Caching 
    # During the image handling we do some costly measurements on the images (galaxy radius)
    # So lets keep them if we add more images later on
    #--------------------------------------------------------------------------     
    def init_cache(self):
        self._cache = Cache(self._cache_path, USE_CACHE)
    
    @property
    def use_cache(self):
        return self._cache.use_cache
        
    def push_to_cache(self, df):
        '''If caching is activated, match new data to the one already in the cache'''
        self._cache.push(df)
        
    def pull_from_cache(self):
        return self._cache.pull()
        
    def skip_image(self, path):
        """
        Test if target image is already existing and can be skipped
        
        Return True if the image exists and also in the cache
        Return False if the image is not exsisting and/or not cached
        """
        
        if os.path.exists(path):
            if self._cache.isin(path, 'image_path'):
                return True
            else:
                os.remove(path)
        
        return False
        
        
    # Stuff to handle filter keys/names correctly
    #--------------------------------------------------------------------------    
    @property
    def filter_dict(self):
        '''Return a dictionary with the filter key mapping e.g. G -> HSC_G'''
        raise NotImplementedError("This function is supposed to be overwritten!")
    
    @property
    def inv_filter_dict(self):
        return {v: k for k, v in self.filter_dict.items()}
        
    def get_filter_keys(self, filters):
        '''Return the catalogue/survey specific filter keys'''        
        return [self.filter_dict[f] for f in filters] 
    
    def get_filter(self, filter_key):
        '''Return the filter from catalogue/survey specific filter names'''
        return self.inv_filter_dict[filter_key]
    
    
    # Call which executes the whole loading process
    #---------------------------------------------------------------------------
    def extract(self):
        raise NotImplementedError("This function is supposed to be overwritten!")
    

class TNGDataExtractor(DataExtractor):
    '''Class to load the TNG data'''
    
    def _load_TNG_labels(self, fields):
        
        #Add necessary fields
        fields.append("snapshot_id")
        fields.append("subhalo_id")
        fields.append("projection")

        #Helper method to get fields and measure time
        def get_field(halos, field):
            t = time.time()
            out = getattr(halos, field)
            logger.info('Load ' + field + ' : ' + str(time.time() - t) + ' [s]')
            return out
        
        #If existing, load already cached labels (usefull if additional labels are added)
        if USE_CACHE:
            df_cached = self.load_labels()
        else:
            df_cached = pd.DataFrame(columns=[])
        
        out = []
        
        for snap in tqdm(self._snapshots, total=len(self._snapshots), disable=False):
    
            cat = Catalogue(self._simulation,
                            snap,
                            c.illustris_path,
                            min_stellar_mass=self._min_mass,
                            max_stellar_mass=self._max_mass,
                            random = False)

            halos = cat.get_subhalos()

            for projection in range(NUM_PROJECTIONS):
                halos.projection = projection
                
                labels = []
                for field in fields:
                    if field not in df_cached.head(0):
                        field_values = get_field(halos, field) 
                        labels.append(field_values)
                    else:
                        labels.append([np.nan]*len(halos))
                        
                out.append(np.transpose(labels))
                
            df_loaded = pd.DataFrame(np.concatenate(out), columns=fields)

        return df_loaded.update(df_cached)
        
    def extract(self):
        df = self._load_TNG_labels(self._fields)
        self.save_labels(df)

     
class TNGHSCExtractor(TNGDataExtractor):
    '''Class to load the TNG mocks'''
    
    @property
    def filter_dict(self):
        return {'G': 'SUBARU_HSC.G', 'R': 'SUBARU_HSC.R', 'I': 'SUBARU_HSC.I'}
    
    def _split_filenames(self, filelist):
        #~/simclr/dataset_raw/TNG50-1/images/059/shalo_059-101_v3_HSC_GRIZY.fits
        splitlist = list(map(lambda x: os.path.split(x), filelist))
        #shalo_059-101_v3_HSC_GRIZY.fits
        snap_ids = list(map(lambda x: x[1].split("_")[1], splitlist))
        projections = list(map(lambda x: x[1].split("_")[2][1], splitlist))
        #059-101
        snapnums = list(map(lambda x: x.split("-")[0], snap_ids))
        sub_ids  = list(map(lambda x: x.split("-")[1], snap_ids))

        snapnums = np.array(snapnums, dtype=np.int32)
        sub_ids  = np.array(sub_ids, dtype=np.int32)
        projections  = np.array(projections, dtype=np.int32)
        
        return snapnums, sub_ids, projections
        
    def _add_image_path(self, df):
        '''Add the path to the respective image for each entry in df. Multiply entrys if there are multiple images for the same galaxy and delete if there is no image'''

        #Get list of all available images
        filelist = glob.glob(self._image_path + '**/*.fits', recursive=True) 
        
        if len(filelist) == 0:
            logger.info("No images available, extract images first!")

        #Get snapshot and id
        snapnums, sub_ids, projections = self._split_filenames(filelist)

        logger.info("Assign Images to Labels")
        #Match the images with the data read from the csv (contained in df)
        origin = df.to_numpy()
        target = []
        mask = [] #Mask to sort out images with not avail data

        snapshot_ids = df["snapshot_id"].to_numpy(dtype=np.int32)
        subhalo_ids = df["subhalo_id"].to_numpy(dtype=np.int32)
        projection_ids = df["projection"].to_numpy(dtype=np.int32)

        #Loop over all images
        for j, (snap, i, p, filepath) in enumerate(zip(snapnums, sub_ids, projections, filelist)):
            #Get matched df index for the image
            index = np.argwhere(np.logical_and(np.logical_and(snapshot_ids==snap, subhalo_ids==i), projection_ids==p))
            assert len(index)<=1, "Multiple Data for one Image"

            #Check if there is data in df available for the given image
            if len(index) == 1:
                index = index[0]
                target.append(origin[index])
                mask.append(True)
            else:
                mask.append(False)

        logger.info(str(np.sum(mask)) + " images assigned to simulation data.")
        logger.info(str(np.sum(np.logical_not(mask))) + " images dropped.")
        df_matched = pd.DataFrame(np.array(target)[:,0,:], columns=df.columns)
        df_matched['image_path'] = np.array(filelist)[mask]
        df_matched['projection'] = np.array(projections)[mask]

        return df_matched
    
    def _resized_copy(self, filedir):
        '''Copy a resized version to save space and memory'''
        
        _, filename = os.path.split(filedir)
        new_path = self._image_path + filename

        #Test if file is already existing; either skip and use the cached data or delete and reload 
        if self.skip_image(new_path):
            return ['', np.nan, np.nan]
        
        #Resize images and copy only needed filters
        with fits.open(filedir) as hdul:

            cropper = PetrosianCropper(image_target_size=self._image_size, petro_multiplier=PETRO_RADIUS_FACTOR)
            cropper.fit(hdul["SUBARU_HSC.R"].data)

            hdu_copy = [fits.PrimaryHDU()]

            for fk, f in zip(self._filters_keys, self._filters):
                header = hdul[fk].header
                image = hdul[fk].data                            
                image = cropper(image)
                hdu_copy.append(fits.ImageHDU(image, name=f, header=header))

            hdul_copy = fits.HDUList(hdu_copy)
            hdul_copy.writeto(new_path)
        
        return [new_path, cropper.r_half_light, cropper.r_90_light]
        
    def _resized_copy_exceptions(self, filedirs):
        try:
            return self._resized_copy(filedirs)
        except OSError:
            logger.warning("Faulty fits file: skip image!")
            return ['', np.nan, np.nan]
        except KeyError:
            logger.warning("Filter missing: skip image!")
            return ['', np.nan, np.nan]
        except TypeError:
            logger.warning("Weird type error: skip image!")
            return ['', np.nan, np.nan]
        except ValueError:
            logger.warning("Weird value error: skip image!")
            return ['', np.nan, np.nan]
        
    def _extract_images(self):
        image_path_wildcard = os.path.join(c.image_cache_path, self._dataset) + '/**/*.fits'
        filelist = glob.glob(image_path_wildcard, recursive=True)
        logger.info(str(len(filelist)) + " images found. Start loading...")
        
        def checkpoint(x):
            x = np.array(x)
            df = pd.DataFrame(x, columns=["image_path", "petro_half_light", "petro_90_light"])
            self.push_to_cache(df)
            
        multi_resized_copy = ChunkedPool(self._resized_copy_exceptions, checkpoint, SIZE_CHUNKS, NUM_WORKERS)
        multi_resized_copy(filelist)
    
    def _extract_labels(self):
        logger.info("Load labels...")
        df = self._load_TNG_labels(self._fields)
        df = self._add_image_path(df)
        df_cache = self.pull_from_cache()
        df = pd.merge(df, df_cache, on=["image_path"])
        self.save_labels(df)
        
    def extract(self):
        if LOAD_IMAGES:
            self._extract_images()
            
        self._extract_labels()
    
class HSCDataExtractor(DataExtractor):
    """Class to get the HSC images, because there is no exact snapshot all data is copied"""
    
    @property
    def filter_dict(self):
        return {'G': 'HSC-G', 'R': 'HSC-R', 'I': 'HSC-I'}

    def _get_data_from_fits(self, filelist):
        '''Extract information from fits files filename'''
        vsplit = np.vectorize(lambda x: os.path.split(x)[1])
        filenames = vsplit(filelist)
        id_list = np.array([int(i[:17]) for i in filenames])
        filter_list = np.array([i[28:33] for i in filenames])
        
        return id_list, filter_list
    
    def _group_filters(self, id_list, filter_list, filelist):
        '''
            Choose the filters asked for and group them together according to the id
            I.e. return a list which contains lists with the files for each filter belonging to the 
            same image given in unique_ids
        '''

        unique_ids = np.unique(id_list)
        
        if self._fraction < 1.0:
            unique_ids = np.random.choice(unique_ids, int(len(unique_ids)*self._fraction), replace=False)
        assert len(unique_ids) > 0
        
        filelist_grouped = []
        mask = []
        
        for i in unique_ids:
            index = np.argwhere(id_list == i)[:,0]
        
            file_cutout = filelist[index]
            filter_cutout = filter_list[index]
            
            filter_index_ordered = []
            
            for f in self._filters_keys:
                index = np.argwhere(filter_cutout == f)
                
                if len(index) == 1:
                    filter_index_ordered.append(index[0,0])
                    
            if len(filter_index_ordered) == len(self._filters_keys):
                filelist_grouped.append(file_cutout[filter_index_ordered])
                mask.append(True)
            else:
                logger.warning("Warning: HSC index " + str(i) + " has a filter missing. Galaxy dropped...")   
                mask.append(False)
            
        return unique_ids[mask], filelist_grouped
    
    def _get_new_image_path(self, unique_id):
        '''Get target path'''
        return self._image_path + "%017d.fits" % (unique_id)
    
    def _remove_cached_images(self, unique_ids, filelist_grouped):
        '''Check if image has already been copied and remove those'''
        
        v_get_new_image_path = np.vectorize(self._get_new_image_path)
        image_paths = v_get_new_image_path(unique_ids)
        
        mask = np.isin(image_paths, self.pull_from_cache()['image_path'].to_numpy()) 
        
        return np.array(unique_ids)[~mask], np.array(filelist_grouped)[~mask], 
            
    def _resized_copy(self, filedirs):
        '''Copy a resized version to save space and memory'''
    
        id_list, filter_list = self._get_data_from_fits(filedirs)
        new_path = self._get_new_image_path(id_list[0])
        
        #Init a PetrosianCropper and fit the red channel
        cropper = PetrosianCropper(image_target_size=self._image_size, petro_multiplier=PETRO_RADIUS_FACTOR)
        file_r = filedirs[filter_list == 'HSC-R']
        with fits.open(file_r[0]) as hdul:
            cropper.fit(hdul[1].data)
        
        hdu_copy = [fits.PrimaryHDU()]
        
        for filedir, f, i in zip(filedirs, filter_list, id_list): 

            #Resize images and copy only needed filters
            with fits.open(filedir) as hdul:
                header = hdul[1].header
                image = hdul[1].data

                #Bring the image to AB standard 
                fluxmag0 = hdul[0].header["FLUXMAG0"]
                image *= 1e9 / fluxmag0

                #Retrict to a multiple of the petro90 radius
                image = cropper(image)

                hdu_copy.append(fits.ImageHDU(image, name=self.get_filter(f), header=header))

        hdul_copy = fits.HDUList(hdu_copy)

        #Save
        hdul_copy.writeto(new_path)

        return [new_path, cropper.r_half_light, cropper.r_90_light]
        
    def _resized_copy_exceptions(self, filedirs):
        try:
            return self._resized_copy(filedirs)
        except OSError:
            logger.warning("Faulty fits file: skip image!")
            return ['', np.nan, np.nan]
        except KeyError:
            logger.warning("Filter missing: skip image!")
            return ['', np.nan, np.nan]
        except TypeError:
            logger.warning("Weird type error: skip image!")
            return ['', np.nan, np.nan]
        except ValueError:
            logger.warning("Weird value error: skip image!")
            return ['', np.nan, np.nan]
        
    def _extract_labels(self, unique_ids):
        CSV_PATH = os.path.join(c.image_cache_path, self._dataset, "s20a_hsc-wide_gridet_sdss-dr16_petr20_gswlc2.csv")
        df = pd.read_csv(CSV_PATH)
        
        fields = self._fields + ['object_id']
        
        df_cutout = pd.DataFrame(df[fields], columns=fields)   
        df_unique_ids = pd.DataFrame(unique_ids, columns=['object_id'])
        
        df = pd.merge(df_unique_ids, df_cutout, on=['object_id'])
        
        v_get_new_image_path = np.vectorize(self._get_new_image_path)
        df['image_path'] = v_get_new_image_path(unique_ids)
        
        df_cache = self.pull_from_cache()
        df = pd.merge(df, df_cache, on=["image_path"])
        
        self.save_labels(df)
        
        return df
        
    def extract(self):
        
        filelist = glob.glob(c.image_cache_path + self._dataset + '/**/*.fits', recursive=True)
        filelist = np.array(filelist)
        logger.info("Overall " + str(len(filelist)) + " images found...")
        
        id_list, filter_list = self._get_data_from_fits(filelist)
        unique_ids, filelist_grouped = self._group_filters(id_list, filter_list, filelist)
        logger.info("Load only a fraction of " + str(self._fraction) + " and remove faulty images: Load " + str(len(unique_ids)) + " images...")
        
        unique_ids_removed, filelist_grouped_removed = self._remove_cached_images(unique_ids, filelist_grouped)       
        logger.info("After removing already cached images: Load " + str(len(unique_ids_removed)) + " images...")
        
        def checkpoint(x):
            x = np.array(x)
            df = pd.DataFrame(x, columns=["image_path", "petro_half_light", "petro_90_light"])
            self.push_to_cache(df)
            
        multi_resized_copy = ChunkedPool(self._resized_copy_exceptions, checkpoint, SIZE_CHUNKS, NUM_WORKERS)
        multi_resized_copy(filelist_grouped_removed)
        
        self._extract_labels(unique_ids)
        
