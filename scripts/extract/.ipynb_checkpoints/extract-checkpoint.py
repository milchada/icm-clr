# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:28:57 2020

Get the data from the various files and
store them all in a csv file per snapshot

@author: Lukas Eisert
"""

from scripts.extract.Catalogue import Catalogue
import numpy as np
import pandas as pd
import os, shutil
import config as c
from tqdm import tqdm

import yaml

params = yaml.safe_load(open('params.yaml'))
extract_params = params['extract']
MIN_STELLAR_MASS = float(extract_params["MIN_STELLAR_MASS"])
MAX_STELLAR_MASS = float(extract_params["MAX_STELLAR_MASS"])
SIMULATIONS = extract_params["SIMULATIONS"]

try:
    MIN_SNAPSHOT = extract_params["MIN_SNAPSHOT"]
except KeyError:
    MIN_SNAPSHOT = None

try:
    SNAPSHOTS = extract_params["SNAPSHOTS"]
except KeyError:
    SNAPSHOTS = None


#List of all collumns to extract
columns = ["snapshot_id", "subhalo_id", "root_descendant_id",
           "lookback", "z",
           "stellar_age", "fraction_disk_stars",
           "mass_total", "mass_in_rad", "mass_exsitu",
           "snap_num_last_maj_merger", "mass_last_maj_merger",
           "mean_merger_lookback_time", "mean_merger_mass_ratio"]


# Extract all columns for given halos
def get_labels(halos):
    
    snapshot_id = halos.snapshot_id
    subhalo_id = halos.subhalo_id
    root_descendant_id = halos.root_descendant_id
    
    lookback = halos.lookback
    z = halos.z
    
    stellar_age = halos.stellar_age_2rhalf_lumw

    fraction_disk_stars = halos.fraction_disk_stars

    mass = halos.stellar_subhalo_mass
    mass_in_rad = halos.mass_in_rad
    exsitu = halos.mass_ex_situ
    
    snap_num_last_maj_merger = halos.snap_num_last_maj_merger
    mass_last_maj_merger = halos.mass_last_maj_merger
    
    mean_merger_lookback_time = halos.mean_merger_lookback_time
    mean_merger_mass_ratio = halos.mean_merger_mass_ratio


    return np.array([snapshot_id, subhalo_id, root_descendant_id,
                     lookback, z,
                     stellar_age, fraction_disk_stars,
                     mass, mass_in_rad, exsitu,
                     snap_num_last_maj_merger, mass_last_maj_merger,
                     mean_merger_lookback_time, mean_merger_mass_ratio])


if __name__ == "__main__":
    
    if SNAPSHOTS is not None:
        snapshots = SNAPSHOTS
    elif MIN_SNAPSHOT is not None:
        snapshots = np.arange(MIN_SNAPSHOT,100)
    else:
        snapshots = np.arange(0,100)
    
    for simulation in SIMULATIONS:
        print("Load " + simulation)
        output_path_sim = c.dataset_raw_path + simulation + "/labels/"
        
        if os.path.exists(output_path_sim):
            shutil.rmtree(output_path_sim) 
            
        os.makedirs(output_path_sim)
     
        for snap in tqdm(snapshots, total=len(snapshots), disable=False):
    
            cat = Catalogue(simulation,
                            snap,
                            c.illustris_path,
                            min_stellar_mass=MIN_STELLAR_MASS,
                            max_stellar_mass=MAX_STELLAR_MASS,
                            random = False)
            
            halos = cat.get_subhalos()
            
            labels = get_labels(halos)  
            labels = np.transpose(labels)
        
            df = pd.DataFrame(labels, columns=columns)
            df.to_csv(output_path_sim + "raw_" + str(snap) + ".csv", index=False)


