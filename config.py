# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:48:29 2021

General config

Parameters for scripts which have to be traced by DVC are stored in params.yaml 

@author: Lukas Eisert 
"""

import torch

#Paths
image_cache_path = "./image_cache/"
illustris_path = "/virgotng/universe/IllustrisTNG/"
dataset_raw_path = "./dataset_raw/"
dataset_path = "./dataset/"
model_path = "./model/"
postprocessing_path = "./postprocessing/"
metrics_path = "./metrics/"
plots_path = "./plots/"
resnet_path = model_path + 'resnet.py'
optuna_resnet_path = lambda x: model_path + 'optuna/run_%04d.py' % (x)
representation_path = postprocessing_path + 'representation.npy'
optuna_representation_path = lambda x:  postprocessing_path + 'optuna/run_representation_%04d.npy' % (x)
cinn_path = model_path + 'cinn.py'
optuna_storage = metrics_path + 'optuna_journal.log'

#Device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Device for the nnclr search
device_nn_search = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Label dict (code friendly -> human friendly)
label_dict = {"mgas": "Gas Mass [log $M_\odot$]",
              "mstar": "Stellar Mass [log $M_\odot$]",
              "mbh": "BH Mass [log $M_\odot$]",
              "bh_einj_cum": "BH Cumulative Energy Injection (erg)",
              "bh_accr": "BH instantaneous accretion rate [log $M_\odot/Gyr$]",
              "half_gas_mass_rad": "Half Gas Mass Radius [kpc]",
              "metalicity_star": "Stellar Metallicity [log $Z_\odot$]",
              "stellar_age": "Stellar Ages [Gyr]",
              "mean_merger_mass_ratio": "Mean Merger Mass Ratio",
              "mean_merger_lookback_time": "Mean Merger Time [Gyr ago]",
              "lookback_time_last_maj_merger": "Last Major Merger Time [Gyr ago]",
              "mass_last_maj_merger": "Last Major Merger Mass [log $M_\odot$]"
              }

#Label dict without units
normalized_label_dict = {}
for key in label_dict.keys():
    normalized_label_dict[key] = label_dict[key].split(' [')[0]



