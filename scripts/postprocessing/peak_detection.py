# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 00:58:07 2021

Calculate posterior MAPs and detect further peaks in the posterior distribution

@author: Lukas Eisert
"""

import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks, peak_prominences

import config as c

from scripts.util import kde
from scripts.data import data

#Load parameters
import yaml
params = yaml.safe_load(open('params.yaml'))
peak_detection_params = params['peak_detection']
EVAL_BINS = peak_detection_params['EVAL_BINS']
MIN_PEAK_DISTANCE = peak_detection_params['MIN_PEAK_DISTANCE']
MIN_PEAK_PROMINENCE = peak_detection_params['MIN_PEAK_PROMINENCE']
MIN_PEAK_HEIGHT = peak_detection_params['MIN_PEAK_HEIGHT']

peak_number_path = c.postprocessing_path + "peak_number.npy"
peak_position_path = c.postprocessing_path + "peak_position.npy"
peak_prominence_path = c.postprocessing_path + "peak_prominence.npy"
map_path = c.postprocessing_path + "map.npy"

def peak_detection(posterior):
    
    def get_peaks(x):
        
        #Filter nans
        mask = np.isnan(x)
        frac_nan = mask.sum()/len(x)
        x = x[~mask]
        
        # Use a gaussian kde
        bin_centre, h = kde(x, N=EVAL_BINS)
        
        #Find Peaks and calculate prominences
        peaks, _ = find_peaks(h, height=MIN_PEAK_HEIGHT, distance=MIN_PEAK_DISTANCE)
        prominences = peak_prominences(h, peaks)[0]
        
        #Enforce a min prominence (but ensure that not all peaks are masked away)
        if np.max(prominences) > MIN_PEAK_PROMINENCE:
            mask = prominences > MIN_PEAK_PROMINENCE
            peaks = peaks[mask]
            prominences = prominences[mask]
        
        #Sort according to prominence large to small
        sort_index = np.flip(np.argsort(prominences))
        peaks = peaks[sort_index]
        prominences = prominences[sort_index]
        
        #Calculate peak positions
        position = bin_centre[peaks]
        
        #If frac of nans larger than 0.5 ad it to the top
        if frac_nan >= 0.5:
            position = np.insert(position, 0, np.nan)
            prominences = np.insert(position, 0, frac_nan)
        elif frac_nan >= 0.:
            position = np.append(position, np.nan)
            prominences = np.append(position, frac_nan)

        return position, prominences
    

    positions = []
    prominences = []
    
    #Loop over all galaxies and target quantities
    for i in tqdm(range(len(data.df_l_test)), total=len(data.df_l_test), disable=False):
        
        pos = []
        pro = []
        
        for j in range(data.NUM_DIM):
            position, prominence = get_peaks(posterior[i,:,j])
            pos.append(position)
            pro.append(prominence)
        
        positions.append(pos)
        prominences.append(pro)
    
    return np.array(positions, dtype=object), np.array(prominences, dtype=object)


if __name__ == "__main__":
    posterior = np.load(c.posterior_path)
    positions, prominences = peak_detection(posterior)

    np.save(peak_position_path, positions, allow_pickle=True)
    np.save(peak_prominence_path, prominences, allow_pickle=True)
    
    #Calculate MAP estimates (thus the postion of the highest peak)
    maps = np.vectorize(lambda x: x[0])(positions)
    np.save(map_path, maps)
    
    #Calculate the number of peaks for each x and galaxy
    peak_number = np.vectorize(lambda x: len(x))(positions)
    np.save(peak_number_path, peak_number)

    metrics = {"frac_multi_peak": str(np.sum(peak_number > 1, axis=0)/peak_number.shape[0])}

    import json
    with open(c.metrics_path + "peak_detection.json", "w") as f:
        json.dump(metrics, f)