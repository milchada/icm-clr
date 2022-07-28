# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 21:57:17 2019

Class to identify TNG centrals within a certain mass range and load them as Subhalos objects 

@author: Lukas Eisert
"""

import illustris_python as il
from . import Subhalos
import numpy as np


class Catalogue:

    def __init__(self, simulation, snapshot_id, path, min_stellar_mass=None, max_stellar_mass=None, random = False):
        self.__snapshot_id = snapshot_id
        self.__simulation = simulation
        self.__path = path
        self.__base_path = path + simulation + "/output"

        #Hubble Constant
        self.__H = 0.6774

        #Load ids
        self.__centrals = il.groupcat.loadHalos(self.__base_path,
                                                self.__snapshot_id,
                                                fields=["GroupFirstSub"])

        stellar_mass = il.groupcat.loadSubhalos(self.__base_path,
                                                self.__snapshot_id,
                                                fields=["SubhaloMassType"])[:,4]


        flag = il.groupcat.loadSubhalos(self.__base_path,
                                        self.__snapshot_id,
                                        fields=["SubhaloFlag"])

        #Fix units
        stellar_mass *= 1e10 / self.__H

        #Apply criterion
        #Get first a mask for all subhalos which fit to the criterion
        if min_stellar_mass is not None and max_stellar_mass is not None:
            mass_mask = np.logical_and(stellar_mass >= min_stellar_mass,
                                       stellar_mass <= max_stellar_mass)
        elif min_stellar_mass is not None:
            mass_mask = stellar_mass >= min_stellar_mass
        elif max_stellar_mass is not None:
            mass_mask = stellar_mass <= max_stellar_mass
        else:
            mass_mask = np.ones_like(stellar_mass)

        mask = np.logical_and(mass_mask, flag)

        #Get index of subhalos which are fine
        mask_index = np.nonzero(mask)

        #Get all subhalo ids which are centrals and fullfill requirements
        self.__centrals = np.intersect1d(self.__centrals, mask_index)

        if random is True:
            np.random.shuffle(self.__centrals)

        self.__num_centrals = len(self.__centrals)

    @property
    def num_centrals(self):
        return self.__num_centrals
    
    def __len__(self):
        return self.num_centrals

    def get_subhalos(self):
        return Subhalos.Subhalos(self.__centrals,
                                 self.__snapshot_id,
                                 self.__simulation,
                                 path = self.__path)


