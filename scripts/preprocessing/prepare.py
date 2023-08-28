# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:39:14 2020

Prepare the data for the training:
 
1) Load each dataset
2) Apply general data cleaning/preparation tasks
3) Match datasets if necessary (Use the first given set for this)
4) Scale the set to 0 mean and 1 variance (Use the first given set for this)
5) Split and save each set 
    
Example (as in params.yaml):

SETS:
- ["TNG300-1", [0.99, 0., 0.]]  <--- Use this one for the scaling
- ["TNG100-1", [0., 0.95, 0.]]
- ["TNG50-1", [0., 0.5, 0.5]]

The preparation mechanism should support later changes in test fractions.
Example:
 
Changing 
   
SETS:
- ["TNG300-1", [0.99, 0., 0.]]
- ["TNG100-1", [0., 0.95, 0.]]
- ["TNG50-1", [0., 0., 1.]]

to

SETS:
- ["TNG300-1", [0.99, 0., 0.01]]
- ["TNG100-1", [0., 0.95, 0.]]

should not alter the training/validation set. Retraining is therefore
not necessary!

@author: Lukas Eisert
"""

import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from astropy.io import fits
from sklearn.neighbors import KDTree

from scripts.util.redshifts import load_redshift
from astropy.cosmology import Planck15 as cosmo

from pickle import dump

import config as c
from scripts.util.str2None import str2None
from scripts.util.make_dir import make_dir

import yaml

params = yaml.safe_load(open('params.yaml'))
prepare_params = params['prepare']
SPLIT_SEED = prepare_params['SPLIT_SEED']
SETS = prepare_params["SETS"]
OBSERVABLES = map(str2None, prepare_params['OBSERVABLES'])
UNOBSERVABLES = map(str2None, prepare_params['UNOBSERVABLES'])
LABELS = prepare_params['LABELS']
ROOT_DESCENDANT_SPLIT = prepare_params['ROOT_DESCENDANT_SPLIT']
MATCHING_FIELDS = str2None(prepare_params['MATCHING_FIELDS'])
MATCHING_SOURCE_SETS = set(prepare_params['MATCHING_SOURCE_SETS'])
MATCHING_TARGET_SETS = set(prepare_params['MATCHING_TARGET_SETS'])
MATCHING_UNCERTAINTIES = np.array(prepare_params['MATCHING_UNCERTAINTIES'][0])

class DatasetPreparator(object):
    '''Load and perform preparation steps on a given dataset i.e. scaling and splitting operations'''
    def __init__(self, dataset):
        self._dataset = dataset
        self._dataset_path = c.dataset_raw_path + self._dataset + "/label.csv"
        
        self.x_scaler = None
        self.y_scaler = None
        
        #Load and prepare set
        print("Load " + self._dataset + " ...")
        self.df = pd.read_csv(self._dataset_path)
        self.df = self.prepare_df(self.df)
        
    #Function to apply general preparations on the raw df
    def prepare_df(self, df):
        
        def dropna(df, field):
            len_before = len(df)
            df = df.dropna(subset=[field])
            len_after = len(df)
            print("Removing NaNs in column: " + field)
            print(str(len_before-len_after) + " galaxies dropped!")
            return df
        
        #Apply log on the mass
        if "mass_in_rad" in df.head(0):
            df["mass"] = df["mass_in_rad"].apply(np.log10)

        #Crop sfr. It is not really important if it is -2 or -6
        if "sfr" in df.head(0):
            df["sfr"] = np.clip(df["sfr"], -2, None)

        #Calculate lookback_time_last_maj_merger from snap_num_last_maj_merger if avail
        if "snap_num_last_maj_merger" in df.head(0):
            snapnums = df['snap_num_last_maj_merger'].to_numpy(dtype=np.int_)
            z = load_redshift(snapnums)
            lookback_times = np.array(cosmo.age(df['z']) - cosmo.age(z))

            #Check for galaxies which never had a Maj Merger
            #Set them randomly in a range around the valid range of (0,13.7) 
            #ie evenly distributed in [-6,-1] and [15,20]
            mask = (snapnums == -1)
            lookback_times[mask] = 15.0

            df['lookback_time_last_maj_merger'] = lookback_times

        if "mass_last_maj_merger" in df.head(0):
            mass = df['mass_last_maj_merger'].to_numpy()

            mask = (mass < 0.)
            mass[mask] = 1e6
            mass = np.log10(mass)

            df['mass_last_maj_merger'] = mass
            
        #Calulate exsitu fraction
        if "mass_exsitu" in df.head(0) and "mass_in_rad" in df.head(0):
            df["exsitu"] = df["mass_exsitu"]/df["mass_in_rad"]

        #Clip away some bad behaving galaxies
        #if "mean_merger_mass_ratio" in df.head(0):
        #    mask = df["mean_merger_mass_ratio"] > 0
        #    df = df[mask]
            
        #Remove galaxies with bad petro fit or other missing fields
        #df = dropna(df, 'petro_half_light')
        df = dropna(df, 'petro_90_light')
        
        matching_fields = np.unique(MATCHING_FIELDS)
        for m in matching_fields:
            if m in df.head(0):
                df = dropna(df, m)

        return df


    def split(self, x, fractions, root_descendant_ids=None):
        '''
        Split x and y randomly into train, val and test set according to fractions tuple
        e.g (0.7, 0.2, 0.1) If fractions dont sum up to 1: drop the remaining dataset
        => (0.,0.,0.1) uses random 10% of the sample for testing
        '''

        def get_random_fraction_mask(fractions, size):
            '''Create masks (of length 'size') according to the given fractions'''
            
            assert np.sum(fractions) <= 1.0, "Sum of fractions should be <= 1"
            
            rng = np.random.default_rng(SPLIT_SEED)
            u = rng.uniform(0.0, 1.0, size=size)
            cumulative_fractions = np.cumsum([0.] + fractions)
            
            assert np.max(cumulative_fractions) <= 1.0
            
            mask_list = []
            
            for i in range(len(fractions)):
                mask = np.logical_and(u > cumulative_fractions[i], u < cumulative_fractions[i+1])
                mask_list.append(mask)
                    
            assert len(mask_list) == len(fractions)
            
            return mask_list
        
        #Shuffle data
        x = shuffle(x, random_state=SPLIT_SEED)

        #Split according to root descendant ids if given
        if root_descendant_ids is not None and ROOT_DESCENDANT_SPLIT:

            #Shuffle also the rd_ids
            root_descendant_ids = shuffle(root_descendant_ids, random_state=SPLIT_SEED)

            #Get unique rd_ids and shuffle them
            rd = np.unique(root_descendant_ids)
            rd = shuffle(rd, random_state=SPLIT_SEED+1)
            
            #Get random split masks
            fraction_masks = get_random_fraction_mask(fractions, len(rd))
            
            #Transfer from root descendant split to a split of the actual data
            for i, m in enumerate(fraction_masks):
                rd_masked = rd[m]
                fraction_masks[i] = np.isin(root_descendant_ids, rd_masked)
            
        else:
            #Get random split masks
            fraction_masks = get_random_fraction_mask(fractions, len(x))
            
        #Apply masks
        split_x = []
        for m in fraction_masks:
            x_masked = x[m]
            split_x.append(x_masked)

        return split_x

    def scale(self, df, fields, scaler=None):
        '''Scale the input, if avail with a given scaler otherwise create a new one based on the given data'''
        
        if fields is None:
            x = np.empty((len(df)))
            x[:] = np.nan
            return pd.DataFrame(x, columns=["None"]), None
        
        x = df.loc[:, fields].values
                
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(x)
            
        return pd.DataFrame(scaler.transform(x), columns=LABELS), scaler
    
    def scale_split(self, fractions, unobservable, observable):
        '''Select, scale and split the data according to the given fractions'''
        
        #Select fields asked for and scale them
        df_x, self.x_scaler = self.scale(self.df, unobservable, self.x_scaler)
        df_y, self.y_scaler = self.scale(self.df, observable, self.y_scaler)
        
        #Keep all fields also as unscaled metadata
        df_m = self.df
        
        #Check for the rd ids
        if "root_descendant_ids" in df_m.head(0):
            root_descendant_ids = df_m['root_descendant_ids']
        else:
            root_descendant_ids = None
        
        #Split the set
        x_split =  self.split(df_x, fractions, root_descendant_ids)
        y_split =  self.split(df_y, fractions, root_descendant_ids)
        m_split =  self.split(df_m, fractions, root_descendant_ids)
        
        return x_split, y_split, m_split
        

class DatasetMatcher(object):
    '''
    Load multiple dataset. To ensure comparability; match samples according to given fields
    '''
    
    def __init__(self):
        self._sets = SETS
        self._datasets = []
        self._dataset_titles = []
        self._dataset_fractions = []
        self._dataset_observables = []
        self._dataset_unobservables = []
        
        self.load_datasets()
        self.plot_matching_fields("raw")
        self.match_datasets()
        self.plot_matching_fields("matched")
        self.scale_split_save()
            
            
    def load_datasets(self):
        
        for s in self._sets:
            self._dataset_titles.append(s[0])
            self._dataset_fractions.append(s[1])
        
        for title in self._dataset_titles:
            self._datasets.append(DatasetPreparator(title))
            
        for obs in OBSERVABLES:
             self._dataset_observables.append(obs)
        
        for unobs in UNOBSERVABLES:
             self._dataset_unobservables.append(unobs)
            
            
    def __len__(self):
        return len(self._datasets)
       

    # Plotting
    #------------------------------------------------------------------------------------------
    
    def plot_matching_fields(self, title):
        
        plot_base = c.plots_path + "preprocessing/"
        make_dir(plot_base)
        
        for j in range(len(MATCHING_FIELDS[0])):
            path = plot_base + title + "_" + MATCHING_FIELDS[0][j] + ".pdf"
            fields = [r[j] for r in MATCHING_FIELDS]
            self.plot_dataset_hist(fields, path)
            
        for j in range(len(MATCHING_FIELDS[0])-1):
            path = plot_base + title + "_" + MATCHING_FIELDS[0][j] + "_" + MATCHING_FIELDS[0][j+1] + ".pdf"
            fields = [[r[j], r[j+1]] for r in MATCHING_FIELDS]
            self.plot_dataset_scatter2d(fields, path)
        
    
    def plot_dataset_hist(self, field, path, bins=50):
        '''
        Plot the hist of a given field (str) for all datasets
        If the field is a list, plot different fields for each dataset in the same histogram 
        '''
        
        assert isinstance(field, str) or (isinstance(field, list) and len(field)==len(self))
        
        fig, ax = plt.subplots()
        
        for i, (data, title) in enumerate(zip(self._datasets, self._dataset_titles)):
            
            if isinstance(field, str):
                ax.hist(data.df[field], density=True, alpha=0.5, label=title, bins=bins)
            else:
                ax.hist(data.df[field[i]], density=True, alpha=0.5, label=title, bins=bins)
        
        if isinstance(field, str):
            ax.set_xlabel(field)
        else:
            ax.set_xlabel(field[0])
            
        ax.legend()
        fig.savefig(path)
        
    def plot_dataset_scatter2d(self, fields, path):
        '''
        Plot the 2d scatter of two fields
        fields should be a list of list with the str of the header to plot
        with dimensions: (num_simulations, 2)
        '''
        
        assert isinstance(fields, list) and len(fields)==len(self)
        for i in fields:
            assert isinstance(i, list)
            assert len(i)==2
        
        fig, ax = plt.subplots()
        
        for i, (data, title) in enumerate(zip(self._datasets, self._dataset_titles)):
            ax.scatter(data.df[fields[i][0]], data.df[fields[i][1]], s=1, label=title)
            ax.set_xlabel(fields[0][0])
            ax.set_ylabel(fields[0][1])
            
        ax.legend()
        fig.savefig(path)
        
    
        
    # Matching
    #------------------------------------------------------------------------------------------
        
    def match_datasets(self):
        '''
            Match the given datasets according to the Matching Fields.
        '''
        
        def inv_concatenate(x, target):
            ''' 
            We have lists of numpy arrays in this method
            Split the numpy array x such that it has the same shape as the target list of numpy arrays.
            The arrays in the list dont need to be of the same size/shape (otherwise a simple reshape would do it)
            
            E.g.
            
            x = [0,1,2,3,4,5,6,7,8,9]
            target = [[0,0,0],[0,0],[0,0,0,0,0]]
            return [[0,1,2],[3,4],[5,6,7,8,9]]
            
            lenghts_target = [3, 2, 5]
            split_indices = [3, 5]
            x_split = [[0,1,2],[3,4],[5,6,7,8,9]]
            
            '''
            
            length_x = x.shape[0]
            lenghts_target = [i.shape[0] for i in target]
            
            assert length_x == np.sum(lenghts_target), "Error: number of galaxies in x differs from target"
            
            #Get the indices for the splitting of x
            split_indices = np.cumsum(lenghts_target)[:-1]
            
            x_split = np.split(x, split_indices)
            
            #Do some sanity checks on the fly
            assert len(x_split) == len(target)
            assert np.all([i.shape[0]==j.shape[0] for i, j in zip(x_split, target)])
            
            return x_split
        
        source = []
        target = []
        source_label = []
        target_label = []
        
        if MATCHING_FIELDS is None:
            return
        
        assert np.all([len(MATCHING_FIELDS[0]) == len(i) for i in MATCHING_FIELDS]), "Number of matching Fields should be the same for all datsets"
            
        #Get Fields from the datasets
        for i, fields in enumerate(MATCHING_FIELDS):
                
            dataset = self._datasets[i]
            matching_data = dataset.df[fields]
            label_data = dataset.df['dataset']
            
            if self._dataset_titles[i] in MATCHING_SOURCE_SETS:
                source.append(matching_data)
                source_label.append(label_data)
            elif self._dataset_titles[i] in MATCHING_TARGET_SETS:
                target.append(matching_data)
                target_label.append(label_data)
            else:
                raise ValueError("Please specify if dataset is a target or source set in params.yaml")

        #Concat lists
        source_concat = np.concatenate(source, axis=0)
        source_label_concat = np.concatenate(source_label, axis=0)
        target_concat = np.concatenate(target, axis=0)        
        target_label_concat = np.concatenate(target_label, axis=0)
        
        #Match the source to the target
        matched_source_mask, matched_target_mask, matched_source_sets, matched_target_sets = DatasetMatcher.get_matched_masks(source_concat, target_concat, source_label_concat, target_label_concat)
        
        #Split the 2 masks back to the shape of the original sets
        matched_source_mask = inv_concatenate(matched_source_mask, source)
        matched_target_mask = inv_concatenate(matched_target_mask, target)
        matched_source_sets = inv_concatenate(matched_source_sets, source)
        matched_target_sets = inv_concatenate(matched_target_sets, target)
        
        #Messy internal index for iterating separately through the source and target sets
        j = 0
        k = 0
        
        #Now apply the matching to the single datasets
        for i, fields in enumerate(MATCHING_FIELDS):
            
            dataset = self._datasets[i]
            dataset_title = self._dataset_titles[i]
                
            if dataset_title in MATCHING_SOURCE_SETS:
                #Remove unmatched source galaxies
                dataset.df['matched_set'] = matched_source_sets[j]
                dataset.df = dataset.df[matched_source_mask[j]]
                j += 1
            elif dataset_title in MATCHING_TARGET_SETS:
                #Remove target galaxies which have no analogue in the source
                dataset.df['matched_set'] = matched_target_sets[k]
                dataset.df = dataset.df[matched_target_mask[k]]
                k += 1
            else:
                raise ValueError("Please specify if dataset is a target or source set in params.yaml")
 
    @staticmethod
    def get_matched_masks(source, target, source_label, target_label):
        '''
        Match the set given by source to the set given by target. 
        Returns the masks of source and target datasets such that the distribution for the given fields are identical
        '''
        
        #Get random
        rng = np.random.default_rng(SPLIT_SEED)
        
        #Ensure that we are dealing with numpy arrays
        source = np.array(source)
        target = np.array(target)
        
        #Set of already used indexes
        unused_source_mask = np.ones(source.shape[0])
        
        #Output list containing the matched target sets to keep record of the matched pairs
        matched_source_sets = np.array(['']*source.shape[0], dtype=object)
        matched_target_sets = np.array(['']*target.shape[0], dtype=object)
        
        #Output mask to remove targets which have no partner
        matched_source_mask = np.zeros(source.shape[0])
        matched_target_mask = np.zeros(target.shape[0])

        #Prepare indexes to walk randomly through the target set
        random_indexes = np.arange(target.shape[0])
        rng.shuffle(random_indexes)
        
        for i in tqdm(random_indexes):
            
            x = target[i]
            
            within_box = np.all(np.abs(source - x) <= MATCHING_UNCERTAINTIES, axis=1)
            within_box = np.logical_and(unused_source_mask, within_box)
            within_box_index = np.argwhere(within_box)
            
            if len(within_box_index) > 0:
                index = int(rng.choice(within_box_index)[0])
                matched_source_sets[index] = target_label[i]
                matched_target_sets[i] = source_label[index]
                unused_source_mask[index]= False
                matched_source_mask[index] = True 
                matched_target_mask[i] = True
        
        #To boolean numpy array
        matched_source_mask = np.array(matched_source_mask, dtype=bool)
        matched_target_mask = np.array(matched_target_mask, dtype=bool)
        
        #Print numer of matched galaxies
        num_matched = str(np.sum(matched_target_mask))
        num_target = str(len(matched_target_mask))
        print("Number of matched galaxies: " + num_matched + " / " + num_target)
        
        return matched_source_mask, matched_target_mask, matched_source_sets, matched_target_sets
 

    
    #------------------------------------------------------------------------------------------
    
    def scale_split_save(self):
        
        def save(dfs_list, title):
            '''Function to concatenate and save the given lists'''
            
            #Transpose list ([datasets, x/y/m] => [x/y/m, datasets])
            dfs_list = map(list, zip(*dfs_list))
            subset = ['x','y','m']
            
            for dfs, s in zip(dfs_list, subset):
                df = pd.concat(dfs)
                df.to_csv(c.dataset_path + s + "_" + title + ".csv", index=False)
        
        
        #Get scaler first
        first_dataset = self._datasets[0]
        first_dataset.scale_split(self._dataset_fractions[0], self._dataset_unobservables[0], self._dataset_observables[0])
        x_scaler = first_dataset.x_scaler
        y_scaler = first_dataset.y_scaler
        
        #Save scaler
        dump(x_scaler, open(c.dataset_path + "x_scaler.pkl", "wb"))
        dump(y_scaler, open(c.dataset_path + "y_scaler.pkl", "wb"))

        #Split the sets and concat them
        train = []
        domain = []
        val = []
        test = []
        
        for s, frac, obs, unobs in zip(self._datasets, self._dataset_fractions, self._dataset_observables, self._dataset_unobservables):
            s.x_scaler = x_scaler
            s.y_scaler = y_scaler
            x_split, y_split, m_split = s.scale_split(frac, unobs, obs)
            
            #Go through each of the splits for x, y and m
            train.append([x_split[0], y_split[0], m_split[0]])
            domain.append([x_split[1], y_split[1], m_split[1]])
            val.append([x_split[2], y_split[2], m_split[2]])
            test.append([x_split[3], y_split[3], m_split[3]])
            
        save(train, 'train')
        save(domain, 'domain')
        save(val, 'val')
        save(test, 'test')
                      
    
if __name__ == "__main__":
    
    #Create Outputpath if not existing
    if not os.path.exists(c.dataset_path):
        os.makedirs(c.dataset_path)
    
    DatasetMatcher()