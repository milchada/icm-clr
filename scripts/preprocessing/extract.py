# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:28:57 2020

Get the data from the various sources, process and store them in the dataset_raw folder

@author: Lukas Eisert
"""


import numpy as np
import pandas as pd
import config as c
from tqdm import tqdm

from scripts.util.str2None import str2None
from scripts.util.logging import logger
from scripts.preprocessing.extractors import DataExtractor

import yaml

params = yaml.safe_load(open('params.yaml'))
extract_params = params['extract']
MIN_STELLAR_MASS = float(extract_params["MIN_STELLAR_MASS"])
MAX_STELLAR_MASS = float(extract_params["MAX_STELLAR_MASS"])
DATASETS = extract_params["DATASETS"]
FIELDS = [str2None(i) for i in extract_params["FIELDS"]]
IMAGE_SIZE = int(str2None(extract_params["IMAGE_SIZE"]))
FILTERS = extract_params["FILTERS"]
SNAPSHOTS = extract_params["SNAPSHOTS"]
LOAD = extract_params["LOAD"]
FRACTION = extract_params["FRACTION"]

if __name__ == "__main__":
        
    for i, dataset in enumerate(DATASETS):
        if LOAD[i]:
            logger.info("Loading Dataset: " + dataset)
            DataExtractor.get_extractor(dataset, MIN_STELLAR_MASS, MAX_STELLAR_MASS, SNAPSHOTS[i], FIELDS[i], IMAGE_SIZE, FILTERS, FRACTION[i])
            
            

