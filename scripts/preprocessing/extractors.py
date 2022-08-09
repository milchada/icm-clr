from scripts.preprocessing.TNG.Catalogue import Catalogue

import os
from multiprocessing import Pool
from tqdm import tqdm
import glob
from astropy.io import fits
import numpy as np
import pandas as pd

import config as c

class DataExtractor(object):
    def __init__(self, dataset, min_mass, max_mass, snapshots, fields=None, image_size=None, filters=None, simulation=None):
        self._dataset = dataset
        self._min_mass = min_mass
        self._max_mass = max_mass
        self._snapshots = snapshots
        
        self._label_path = os.path.join(c.dataset_raw_path, self._dataset, "label.csv")
        self._image_path = os.path.join(c.dataset_raw_path, self._dataset, "images/")
        
        self._fields = fields
        self._image_size = image_size
        self._filters = filters
        self._filters_keys = self.get_filter_keys(filters)
        self._simulation = simulation
        
        self.create_paths()
        
    def get_extractor(dataset, min_mass, max_mass, snapshots, fields=None, image_size=None, filters=None):
        '''Static factory method to get the correct DataExtractor object based on the dataset asked for'''
        
        if dataset in {"TNG50-1", "TNG100-1", "TNG300-1"}:
            return TNGDataExtractor(dataset, min_mass, max_mass, snapshots, fields)
        elif dataset in {"HSC_TNG50", "HSC_TNG50_Ideal"}:
            return TNGHSCExtractor(dataset, min_mass, max_mass, snapshots, fields, image_size, filters, simulation="TNG50-1")
        elif dataset in {"HSC"}:
            return HSCDataExtractor(dataset, min_mass, max_mass, snapshots, fields, image_size, filters)
        else:
            raise NotImplementedError(dataset + " has not been implemented yet!")
    
    def create_paths(self):        
        '''Create paths if not already existing'''
        if not os.path.exists(self._image_path):
            os.makedirs(self._image_path)
            
    def save_labels(self, df):
        df["dataset"] = self._dataset
        df.to_csv(self._label_path, index=False)
            
    def get_filter_keys(self, filters):
        '''Return the catalogue/survey specific filter keys'''
        return None
    
    def extract(self):
        raise NotImplementedError("This function is supposed to be overwritten!")
    

class TNGDataExtractor(DataExtractor):
    '''Class to load the TNG data'''
    
    def _load_TNG_labels(self, fields):
        
        out = []
        
        for snap in tqdm(self._snapshots, total=len(self._snapshots), disable=False):
    
            cat = Catalogue(self._simulation,
                            snap,
                            c.illustris_path,
                            min_stellar_mass=self._min_mass,
                            max_stellar_mass=self._max_mass,
                            random = False)

            halos = cat.get_subhalos()

            labels = []

            for field in fields:
                field_values = getattr(halos, field) 
                labels.append(field_values)

            out.append(np.transpose(labels))
            
        return pd.DataFrame(np.concatenate(out), columns=fields)
        
    def extract(self):
        df = self._load_TNG_labels(self._fields)
        self.save_labels(df)

     
class TNGHSCExtractor(TNGDataExtractor):
    '''Class to load the TNG data incl the HSC Mocks'''
        
    def _add_image_path(self, df):
        '''Add the path to the respective image for each entry in df. Multiply entrys if there are multiple images for the same galaxy and delete if there is no image'''

        #Get list of all available images
        filelist = glob.glob(self._image_path + '**/*.fits', recursive=True) 
        
        if len(filelist) == 0:
            print("No images available, extract images first!")

        #Get snapshot and id
        print("Get image snapshot and ids from filenames")
        #~/simclr/dataset_raw/TNG50-1/images/059/shalo_059-101_v3_HSC_GRIZY.fits
        splitlist = list(map(lambda x: os.path.split(x), filelist))
        #shalo_059-101_v3_HSC_GRIZY.fits
        snap_ids = list(map(lambda x: x[1].split("_")[1], splitlist))
        #059-101
        snapnums = list(map(lambda x: x.split("-")[0], snap_ids))
        sub_ids  = list(map(lambda x: x.split("-")[1], snap_ids))

        snapnums = np.array(snapnums, dtype=np.int32)
        sub_ids  = np.array(sub_ids, dtype=np.int32)


        print("Assign Images to Labels")
        #Match the images with the data read from the csv (contained in df)
        origin = df.to_numpy()
        target = []
        mask = [] #Mask to sort out images with not avail data

        snapshot_ids = df["snapshot_id"].to_numpy(dtype=np.int32)
        subhalo_ids = df["subhalo_id"].to_numpy(dtype=np.int32)

        #Loop over all images
        for j, (snap, i, filepath) in enumerate(zip(snapnums, sub_ids, filelist)):
            #Get matched df index for the image
            index = np.argwhere(np.logical_and(snapshot_ids==snap, subhalo_ids==i))
            assert len(index)<=1, "Multiple Data for one Image"

            #Check if there is data in df available for the given image
            if len(index) == 1:
                index = index[0]
                target.append(origin[index])
                mask.append(True)
            else:
                mask.append(False)

        print(str(np.sum(mask)) + " images assigned to simulation data.")
        print(str(np.sum(np.logical_not(mask))) + " images dropped.")
        df_matched = pd.DataFrame(np.array(target)[:,0,:], columns=df.columns)
        df_matched['image_path'] = np.array(filelist)[mask]

        return df_matched
    
    #Old implementation to also add stuff from the image headers, not needed at the moment.
    '''
    def _add_header(self, image_path):
        """Add aditional information from the fits headers"""
        
        label_dict = {}
        
        with fits.open(image_path) as hdul:
            for hdu in hdul:
                header = hdu.header()
                f = header["FILTER"]
                amag = header["APMAG"]
                label_dict['amag_' + str(f)] = amag
                

    def _multi_add_header(self, df, num_threads=18):
        
        print("Add entrys from fits files to label")
        
        image_paths = df["image_path"]
        dicts = p.map(self._add_header, image_paths)
         
        return pd.concat([df, pd.DataFrame.from_dict(dicts)], axis=1)
    '''
    
    def get_filter_keys(self, filters):
        
        filter_dict = {'G': 'SUBARU_HSC.G',
                       'R': 'SUBARU_HSC.R',
                       'I': 'SUBARU_HSC.I'} 
        
        return [filter_dict[f] for f in filters] 
    
    def _resized_copy(self, filedir):
        '''Copy a resized version to save space and memory'''
        
        _, filename = os.path.split(filedir)
        new_path = self._image_path + filename

        #Test if file is already existing, in that case: skip
        if os.path.exists(new_path):
            return

        #Resize images and copy only needed filters
        with fits.open(filedir) as hdul:
            try:
                
                hdu_copy = [fits.PrimaryHDU()]
                
                for fk, f in zip(self._filters_keys, self._filters):
                    header = hdul[fk].header
                    image = hdul[fk].data
                    resized_image = resize(image, (self._image_size, self._image_size))
                    hdu_copy.append(fits.ImageHDU(resized_image, name=f, header=header))
                    
            except:
                print("Loading failed for " + filedir)
                return

        hdul_copy = fits.HDUList(hdu_copy)

        #Save
        hdul_copy.writeto(new_path)
        
    def _multi_resized_copy(self, filelist, num_threads=18):
        with Pool(num_threads) as p:
            for _ in tqdm(p.imap_unordered(self._resized_copy, filelist), total=len(filelist)):
                pass
        
    def _extract_images(self):
        image_path_wildcard = os.path.join(c.image_cache_path, self._dataset) + '/**/*.fits'
        filelist = glob.glob(image_path_wildcard, recursive=True)
        print(str(len(filelist)) + " images found. Start loading...")
        self._multi_resized_copy(filelist)
    
    def _extract_labels(self):
        print("Load labels...")
        df = self._load_TNG_labels(self._fields)
        df = self._add_image_path(df)
        self.save_labels(df)
        
    def extract(self):
        self._extract_images()
        self._extract_labels()
    
class HSCDataExtractor(DataExtractor):
    """Class to get the HSC images, because there is no exact snapshot all data is copied"""
    
    def get_filter_keys(self, filters):
        
        filter_dict = {'G': 'HSC-G',
                       'R': 'HSC-R',
                       'I': 'HSC-I'} 
        
        return [filter_dict[f] for f in filters] 
    
    def get_filter(self, filter_key):
        filter_dict = {'HSC-G': 'G',
                       'HSC-R': 'R',
                       'HSC-I': 'I'} 
        
        return filter_dict[filter_key]
        

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
                print("Warning: HSC index " + str(i) + " has a filter missing. Galaxy dropped...")   
                mask.append(False)
            
        return unique_ids[mask], filelist_grouped
    
    def _get_new_image_path(self, unique_id):
        return self._image_path + "%017d.fits" % (unique_id)
            
    def _resized_copy(self, filedirs):
        '''Copy a resized version to save space and memory'''
    
        id_list, filter_list = self._get_data_from_fits(filedirs)
        new_path = self._get_new_image_path(id_list[0])

        #Test if file is already existing, in that case: skip
        if os.path.exists(new_path):
            return
        
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
                PETRO_MULTI = 4
                target_size = np.min([PETRO_MULTI*self._petroR90_r[i], 50])
                crop_fraction = target_size/50
                    
                #Get a central crop
                shape = image.shape
                min_edge = int(np.min(shape)*crop_fraction)
                min_edge_half = min_edge//2
                center_x = shape[0]//2
                center_y = shape[0]//2
                min_x = center_x - min_edge_half
                max_x = center_x + min_edge_half
                min_y = center_y - min_edge_half
                max_y = center_y + min_edge_half
                    
                    
                cropped_image = image[min_x:max_x, min_y:max_y]
                resized_image = resize(cropped_image, (self._image_size, self._image_size))
                hdu_copy.append(fits.ImageHDU(resized_image, name=self.get_filter(f), header=header))

        hdul_copy = fits.HDUList(hdu_copy)

        #Save
        hdul_copy.writeto(new_path)
        
    def _multi_resized_copy(self, filelist, df, num_threads=18):
        
        self._petroR90_r = dict(zip(df.object_id, df.petroR90_r))    
        
        with Pool(num_threads) as p:
            for _ in tqdm(p.imap_unordered(self._resized_copy, filelist), total=len(filelist)):
                pass
    
    def _extract_labels(self, unique_ids):
        CSV_PATH = os.path.join(c.image_cache_path, self._dataset, "s20a_hsc-wide_gridet_sdss-dr16_petr20_gswlc2.csv")
        df = pd.read_csv(CSV_PATH)
        
        fields = self._fields + ['object_id']
        
        df_cutout = pd.DataFrame(df[fields], columns=fields)   
        df_unique_ids = pd.DataFrame(unique_ids, columns=['object_id'])
        
        df = pd.merge(df_unique_ids, df_cutout, on=['object_id'])
        
        v_get_new_image_path = np.vectorize(self._get_new_image_path)
        df['image_path'] = v_get_new_image_path(unique_ids)
        
        self.save_labels(df)
        
        return df
        
    def extract(self):
        
        filelist = glob.glob(c.image_cache_path + self._dataset + '/**/*.fits', recursive=True)
        filelist = np.array(filelist)
        
        if len(filelist) == 0:
            print("Error: no images found!")
            return
        
        id_list, filter_list = self._get_data_from_fits(filelist)
        
        unique_ids, filelist_grouped = self._group_filters(id_list, filter_list, filelist)
        
        df = self._extract_labels(unique_ids)
        self._multi_resized_copy(filelist_grouped, df)