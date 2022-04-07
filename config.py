# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:48:29 2021

General config

Parameters for scripts which have to be traced by DVC are stored in params.yaml 

@author: Lukas Eisert 
"""

#Paths
illustris_path = "../IllustrisTNG/"
dataset_raw_path = "./dataset_raw/"
dataset_path = "./dataset/"
model_path = "./model/"
postprocessing_path = "./postprocessing/"
metrics_path = "./metrics/"
plots_path = "./plots/"

#Label dict (code friendly -> human friendly)
label_dict = {"fraction_disk_stars": "Fraction of Disk Stars [D/T]",
              "mass": "Stellar Mass [log $M_\odot$]",
              "lookback": "Lookback Time [Gyr]",
              "half_light_rad": "Half Light Radius [kpc]",
              "color": "g-r Color [mag]",
              "metalicity_star": "Stellar Metallicity [log $Z_\odot$]",
              "stellar_age": "Stellar Ages [Gyr]",
              "exsitu": "Stellar Ex-Situ Fraction",
              "mean_merger_mass_ratio": "Mean Merger Mass Ratio",
              "mean_merger_lookback_time": "Mean Merger Time [Gyr ago]",
              "lookback_time_last_maj_merger": "Last Major Merger Time [Gyr ago]",
              "mass_last_maj_merger": "Last Major Merger Mass [log $M_\odot$]"
              }

#Label dict without units
normalized_label_dict = {"fraction_disk_stars": "Fraction of Disk Stars",
                         "mass": "Stellar Mass",
                         "lookback": "Lookback Time",
                         "half_light_rad": "Half Light Radius",
                         "color": "g-r Color",
                         "metalicity_star": "Stellar Metallicity",
                         "stellar_age": "Stellar Ages",
                         "exsitu": "Fraction of Ex-Situ Stars",
                         "mean_merger_mass_ratio": "Mean Merger Mass Ratio",
                         "mean_merger_lookback_time": "Mean Merger Lookback Time",
                         "lookback_time_last_maj_merger": "Last Major Merger Lookback Time",
                         "mass_last_maj_merger": "Last Major Merger Mass"
                         }

#List of quantites which are in log but should be not treated as log in some of the plots (especially regarding relative errors)
logged_quantities = ["mass_last_maj_merger"]
