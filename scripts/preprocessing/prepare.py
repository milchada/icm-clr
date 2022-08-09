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
- ["TNG300-1", [0.99, 0., 0.]]  <--- Use this one for the scaling and matching
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

import yaml

params = yaml.safe_load(open('params.yaml'))
prepare_params = params['prepare']
SPLIT_SEED = prepare_params['SPLIT_SEED']
SETS = prepare_params["SETS"]
OBSERVABLES = str2None(prepare_params['OBSERVABLES'])
UNOBSERVABLES = prepare_params['UNOBSERVABLES']
ROOT_DESCENDANT_SPLIT = prepare_params['ROOT_DESCENDANT_SPLIT']
MATCHING_MAX_ITER = prepare_params['MATCHING_MAX_ITER']
MATCHING_MAX_DIST = prepare_params['MATCHING_MAX_DIST']
MATCHING_FIELDS = str2None(prepare_params['MATCHING_FIELDS'])

class DatasetPreparator(object):
    '''Load and perform preparation steps on a given dataset i.e. scaling and splitting operations'''
    def __init__(self, dataset):
        self._dataset = dataset
        self._dataset_path = c.dataset_raw_path + self._dataset + "/label.csv"
        
        self.x_scaler = None
        self.y_scaler = None
        
        #Load and prepare set
        self.df = pd.read_csv(self._dataset_path)
        self.df = self.prepare_df(self.df)
        
    #Helper function to log the sfr
    #Replace 0 by minimum
    def log_sfr(self, sfr):
        sfr = np.array(sfr)
        mask = (sfr == 0)
        min_sfr = np.min(sfr[~mask])
        sfr[mask] = min_sfr
        return np.log10(sfr)

    #Function to apply general preparations on the raw df
    def prepare_df(self, df):
        #Apply log on the mass
        if "mass_in_rad" in df.head(0):
            df["mass"] = df["mass_in_rad"].apply(np.log10)

        #Crop sfr. Because it is log it is not really important if it is -2 or -6
        #The MLP however will waste energy into learning this difference
        if "sfr" in df.head(0):
            df["sfr"] = self.log_sfr(df["sfr"])
            df["sfr"] = np.clip(df["sfr"], -2, None)

        #Convert metallicity to solar metallicity
        if "metalicity_star" in df.head(0):
            df["metalicity_star"] = df["metalicity_star"].apply(lambda x: np.log(x/0.0127))

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
        if "mean_merger_mass_ratio" in df.head(0):
            mask = df["mean_merger_mass_ratio"] > 0
            df = df[mask]

        return df


    def train_val_test_split(self, x, y, m, fractions):
        '''
        Split x and y randomly into train, val and test set according to fractions tuple
        e.g (0.7, 0.2, 0.1) If fractions dont sum up to 1: drop the remaining dataset
        => (0.,0.,0.1) uses random 10% of the sample for testing
        '''

        assert np.sum(fractions) <= 1.0, "Sum of fractions should be <= 1"

        #Shuffle data
        x = shuffle(x, random_state=SPLIT_SEED)
        y = shuffle(y, random_state=SPLIT_SEED)
        m = shuffle(m, random_state=SPLIT_SEED)

        #Get masks according to fraction to split the sample:
        if not "root_descendant_ids" in self.df.head(0) or not ROOT_DESCENDANT_SPLIT:

            rng = np.random.default_rng(SPLIT_SEED)
            u = rng.uniform(0.0, 1.0, size=len(x))
            train = u < fractions[0]
            val = np.logical_and(u > fractions[0], u < fractions[0] + fractions[1])
            test = u > (1. - fractions[2]) 

        else: 
            #Get root descendant_ids
            root_descendant_ids = m["root_descendant_id"]

            #Shuffle also the rd_ids
            root_descendant_ids = shuffle(root_descendant_ids, random_state=SPLIT_SEED)

            #Get unique rd_ids and shuffle them
            rd = np.unique(root_descendant_ids)
            rd = shuffle(rd, random_state=SPLIT_SEED+1)

            #Split the unique ids according to fraction
            #(Assume that the number of subhalos for each unique rd differs not too much)
            rng = np.random.default_rng(SPLIT_SEED)
            u = rng.uniform(0.0, 1.0, size=len(rd))
            train = u < fractions[0]
            val = np.logical_and(u > fractions[0], u < fractions[0] + fractions[1])
            test = u > (1. - fractions[2]) 

            rd_train = rd[train]
            rd_val = rd[val]
            rd_test = rd[test]

            #Include all subhalos where the rd is equal
            train = np.isin(root_descendant_ids, rd_train)
            val = np.isin(root_descendant_ids, rd_val)
            test = np.isin(root_descendant_ids, rd_test)

            
        #Apply masks
        x_train = x[train]
        y_train = y[train]
        m_train = m[train]

        x_val = x[val]
        y_val = y[val]
        m_val = m[val]

        x_test = x[test]
        y_test = y[test]
        m_test = m[test]

        return (x_train, y_train, m_train), (x_val, y_val, m_val), (x_test, y_test, m_test)

    def scale(self, df, fields, scaler=None):
        '''Scale the input, if avail with a given scaler otherwise create a new one based on the given data'''
        
        if fields is None:
            x = np.empty((len(df)))
            x[:] = np.nan
            return pd.DataFrame(x, columns=["None"]), None
        
        try:
            x = df.loc[:, fields].values
        except KeyError as e:
            print(e)
            print("Warning: Dataset " + self._dataset + " does not contain all fields.")
            x = np.empty((len(df),len(fields)))
            x[:] = np.nan
            return pd.DataFrame(x, columns=fields), None
                
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(x)
            
        return pd.DataFrame(scaler.transform(x), columns=fields), scaler
    
    def scale_split(self, fractions):
        '''Select, scale and split the data according to the given fractions'''
        
        #Select fields asked for and scale them
        df_x, self.x_scaler = self.scale(self.df, UNOBSERVABLES, self.x_scaler)
        df_y, self.y_scaler = self.scale(self.df, OBSERVABLES, self.y_scaler)
        
        #Keep all fields also as unscaled metadata
        df_m = self.df
        
        #Split this set according to the fractions given by s[1]
        train, val, test =  self.train_val_test_split(df_x, df_y, df_m, fractions)
        
        return train, val, test
        

class DatasetMatcher(object):
    ''''''
    
    def __init__(self):
        self._sets = SETS
        self._datasets = []
        self._dataset_titles = []
        self._dataset_fractions = []
        
        self.load_datasets()
        self.match_datasets()
        self.scale_split_save()
            
            
    def load_datasets(self):
        
        for s in self._sets:
            self._dataset_titles.append(s[0])
            self._dataset_fractions.append(s[1])
        
        for title in self._dataset_titles:
            self._datasets.append(DatasetPreparator(title))
        
    def match_datasets(self):
        
        if MATCHING_FIELDS is not None and len(self._datasets)>1:
            target_dataset = self._datasets[0]
            target_fields = MATCHING_FIELDS[0]
            target = target_dataset.df[target_fields]
            
            for i, fields in enumerate(MATCHING_FIELDS):
                
                if i == 0:
                    continue
                
                source_dataset = self._datasets[i]
                source = source_dataset.df[fields]

                #Match the source to the target
                matched_indexes = self._get_matched_indexes(target, source)
                source_dataset.df = source_dataset.df.iloc[matched_indexes]
        
    
    def _get_matched_indexes(self, target, source):
        '''
        Match the set given by source to the set given by target. 
        Returns the index which will sort the source dataset to match the distribution of the target for the columns given 
        '''
        
        def plot(target, source):
            bins = 50
            for th, sh in zip(target.head(0), source.head(0)):
                fig = plt.Figure()
                plt.hist(target[th], density=True, label="target", bins=bins)
                plt.hist(source[sh], density=True, alpha=0.5, label="source", bins=bins)
                plt.xlabel(th)
                plt.legend()
                fig.savefig(c.plots_path + "matching_" + th + ".pdf")
        
        #Create a KDTree to search for the nearest galaxy in the sample to match
        kdt = KDTree(source, metric='euclidean')
        
        #Set of already used indexes
        index_set = set()
        
        #Output list containing the matched 
        matched_indexes = []

        for x in target.to_numpy():
            for i in range(MATCHING_MAX_ITER): 
                distance, index = kdt.query([x], k=i+1, return_distance=True)
                distance = distance[0,-1]
                index = index[0,-1]
                if index not in index_set or distance>MATCHING_MAX_DIST:
                    break            

            index_set.add(index)
            matched_indexes.append(index)

        ux, counts = np.unique(matched_indexes, return_counts=True)
        print("Number of double matched galaxies: " + str(np.sum(counts > 1)))
        
        #Plot
        plot(target, source.iloc[matched_indexes])
        
        return matched_indexes
    
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
        first_dataset.scale_split(self._dataset_fractions[0])
        x_scaler = first_dataset.x_scaler
        y_scaler = first_dataset.y_scaler
        
        #Save scaler
        dump(x_scaler, open(c.dataset_path + "x_scaler.pkl", "wb"))
        dump(y_scaler, open(c.dataset_path + "y_scaler.pkl", "wb"))

        #Split the sets and concat them
        train = []
        val = []
        test = []
        
        for s, frac in zip(self._datasets, self._dataset_fractions):
            s.x_scaler = x_scaler
            s.y_scaler = x_scaler
            split = s.scale_split(frac)
            train.append(split[0])
            val.append(split[1])
            test.append(split[2])
            
        save(train, 'train')
        save(val, 'val')
        save(test, 'test')
                      
    
if __name__ == "__main__":
    
    #Create Outputpath if not existing
    if not os.path.exists(c.dataset_path):
        os.makedirs(c.dataset_path)
    
    DatasetMatcher()