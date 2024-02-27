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
illustris_path = "/virgotng/mpia/TNG-Cluster/" #universe/IllustrisTNG/"
dataset_raw_path = "./dataset_raw/"
dataset_path = "./dataset/"
model_path = "./model/"
postprocessing_path = "./postprocessing/"
metrics_path = "./metrics/"
plots_path = "./plots/"
resnet_path = model_path + 'resnet.pt'
optuna_resnet_path = lambda x: model_path + 'optuna/run_%04d.pt' % (x)
representation_path = postprocessing_path + 'representation.npy'
optuna_representation_path = lambda x:  postprocessing_path + 'optuna/run_representation_%04d.npy' % (x)
cinn_path = model_path + 'cinn.pt'
optuna_storage = metrics_path + 'optuna_journal.log'

#Device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Device for the nnclr search
device_nn_search = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Label dict (code friendly -> human friendly)
label_dict = {
              "stellar_mass"      : "Stellar Mass [log $M_\odot$]", 
              "gas_mass"          : "Gas Mass [log $M_\odot$]",
              "m200c"             : "Halo $M_{200c}$ [log $M_\odot$]", 
              "m500c"             : "Halo $M_{500c}$ [log $M_\odot$]",
              "bh_mass"           : "Black Hole Mass [log $M_\odot$]", 
              "bh_accr"           : "Instantaneous Black Hole Accretion Rate [log $M_\odot/yr$]", 
              "bh_einj_cum"       : "Cumulative Black Hole Energy Injection [log erg]", 
              "bh_kinj_cum"       : "Cumulative Black Hole Kinetic Energy Injection [log erg]", 
              "bh_ting_cum"       : "Cumulative Black Hole Thermal Energy Injection [log erg]",
              "lookback"          : "Lookback Time [Gyr]",
              "mean_merger_lookback_time"    : "Mean Merger Time [Gyr ago]",
              "lookback_time_last_maj_merger": "Last Major Merger Time [Gyr ago]",
              "mass_last_maj_merger"         : "Last Major Merger Mass [log $M_\odot$]"
              }

#Label dict without units
normalized_label_dict = {}
for key in label_dict.keys():
    normalized_label_dict[key] = label_dict[key].split(' [')[0]



