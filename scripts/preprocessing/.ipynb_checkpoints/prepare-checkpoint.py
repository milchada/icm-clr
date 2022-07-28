# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:39:14 2020

Prepare the data for the training:
 
1) Load each dataset
2) Apply general data cleaning/preparation tasks
3) Scale the set to 0 mean and 1 variance (Use the first given set for this)
4) Split and save each set (Use always the last set where the subset is > 0.0)
    
Example (as in params.yaml):

SETS:
- ["TNG300-1", [0.99, 0., 0.]]  <--- Use this one for the scaling
- ["TNG100-1", [0., 0.95, 0.]]
- ["TNG50-1", [0., 0.5, 0.5]] <--- Use half of this set for validation (overwrites TNG100)

With the last step we ensure that the subsamples are not a mix of multiple simulations
Therefore: One simulation -> multiple subsets
BUT: One subset -> One simulation


This preparation mechanism should support later changes in test fractions.
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

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from astropy.io import fits

from scripts.util.redshifts import load_redshift
from astropy.cosmology import Planck15 as cosmo

from pickle import dump

import config as c

import yaml

params = yaml.safe_load(open('params.yaml'))
prepare_params = params['prepare']
SPLIT_SEED = prepare_params['SPLIT_SEED']
SETS = prepare_params["SETS"]
OBSEVABLES = prepare_params['OBSEVABLES']
UNOBSERVABLES = prepare_params['UNOBSERVABLES']
ROOT_DESCENDANT_SPLIT = prepare_params['ROOT_DESCENDANT_SPLIT']

# General Functions
#-----------------------------------------------------------------------------

#Helper function to log the sfr
#Replace 0 by minimum
def log_sfr(sfr):
    sfr = np.array(sfr)
    mask = (sfr == 0)
    min_sfr = np.min(sfr[~mask])
    sfr[mask] = min_sfr
    return np.log10(sfr)

#Function to apply general preparations on the raw df
def prepare_df(df):
    #Apply log on the mass
    df["mass"] = df["mass_in_rad"].apply(np.log10)
    
    #Crop sfr. Because it is log it is not really important if it is -2 or -6
    #The MLP however will waste energy into learning this difference
    if "sfr" in df.head(0):
        df["sfr"] = log_sfr(df["sfr"])
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
    df["exsitu"] = df["mass_exsitu"]/df["mass_total"]
    
    #Clip away some bad behaving galaxies
    if "mean_merger_mass_ratio" in df.head(0):
        mask = df["mean_merger_mass_ratio"] > 0
        df = df[mask]
    
    return df

#Split x and y randomly into train, val and test set according to fractions tuple
#e.g (0.7, 0.2, 0.1) If fractions dont sum up to 1: drop the remaining dataset
#=> (0.,0.,0.1) uses random 10% of the sample for testing
#Keep it reproducable by setting the seed
#Split according to root_descendant_id if given; by this it is ensured,
#that all progenitor/descendants of a subhalo are in the same subset
def train_val_test_split(x, y, m, fractions, seed=0, root_descendant_ids=None):
    
    assert np.sum(fractions) <= 1.0, "Sum of fractions should be <= 1"
    
    #Shuffle data
    x = shuffle(x, random_state=seed)
    y = shuffle(y, random_state=seed)
    m = shuffle(m, random_state=seed)
    
    #Get masks according to fraction to split the sample:
    if root_descendant_ids is None:
        
        rng = np.random.default_rng(seed)
        u = rng.uniform(0.0, 1.0, size=len(x))
        train = u < fractions[0]
        val = np.logical_and(u > fractions[0], u < fractions[0] + fractions[1])
        test = u > (1. - fractions[2]) 
        
    else:
        
        #Shuffle also the rd_ids
        root_descendant_ids = shuffle(root_descendant_ids, random_state=seed)
        
        #Get unique rd_ids and shuffle them
        rd = np.unique(root_descendant_ids)
        rd = shuffle(rd, random_state=seed+1)
        
        #Split the unique ids according to fraction
        #(Assume that the number of subhalos for each unique rd differs not too much)
        rng = np.random.default_rng(seed)
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
    
#Scale the input, if avail with a given scaler
def scale(df, fields, scaler=None):
    x = df.loc[:, fields].values
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(x)
    return pd.DataFrame(scaler.transform(x), columns=fields), scaler
    
if __name__ == "__main__":
    
    #Create Outputpath if not existing
    if not os.path.exists(c.dataset_path):
        os.makedirs(c.dataset_path)
    
    x_scaler = None
    y_scaler = None
    
    for s in SETS:
        
        #Load and prepare set
        df = pd.read_csv(c.dataset_raw_path + s[0] + "/labels.csv")
        df = prepare_df(df)
        
        #Scale
        df_x, x_scaler = scale(df, UNOBSERVABLES, x_scaler)
        df_y, y_scaler = scale(df, OBSEVABLES, y_scaler)
        
        #Keep all fields also as unscaled metadata
        df_m = df
        
        #Get root_descendant_ids if flag is set
        if ROOT_DESCENDANT_SPLIT:
            root_descendant_ids = df["root_descendant_id"]
        else:
            root_descendant_ids = None
    
        #Split this set according to the fractions given by s[1]
        train, val, test =  train_val_test_split(df_x, df_y, df_m,
                                                 s[1],
                                                 SPLIT_SEED,
                                                 root_descendant_ids)

        #Save Training Set
        if s[1][0] > 0.0:
            
            train[0].to_csv(c.dataset_path + "x_train.csv", index=False)
            train[1].to_csv(c.dataset_path + "y_train.csv", index=False)
            train[2].to_csv(c.dataset_path + "m_train.csv", index=False)
            
        #Save Validation Set
        if s[1][1] > 0.0:

            val[0].to_csv(c.dataset_path + "x_val.csv", index=False)
            val[1].to_csv(c.dataset_path + "y_val.csv", index=False)
            val[2].to_csv(c.dataset_path + "m_val.csv", index=False)
            
        #Save Test Set
        if s[1][2] > 0.0:

            test[0].to_csv(c.dataset_path + "x_test.csv", index=False)
            test[1].to_csv(c.dataset_path + "y_test.csv", index=False)
            test[2].to_csv(c.dataset_path + "m_test.csv", index=False)
            
    #Save scaler
    dump(x_scaler, open(c.dataset_path + "x_scaler.pkl", "wb"))
    dump(y_scaler, open(c.dataset_path + "y_scaler.pkl", "wb"))
