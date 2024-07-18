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

    def __init__(self, simulation, snapshot_id, path, min_mass=None, max_mass=None, centrals_only=False, random=False, grouptype = 'FoF'):
        self._grouptype = grouptype
        assert self._grouptype in ['FoF','SubFind'], "Invalid group type! Must be either FoF or SubFind"
        
        self._snapshot_id = snapshot_id
        self._simulation = simulation
        self._path = path
        self._base_path = path + simulation + "/output"

        #Hubble Constant
        self._H = 0.6774

        #Load centrals ids
        if centrals_only:
            self._centrals = il.groupcat.loadHalos(self._base_path,
                                                   self._snapshot_id,
                                                   fields=["GroupFirstSub"])
            
        if self._grouptype == 'FoF':
            mass = il.groupcat.loadHalos(self._base_path, self._snapshot_id, fields=['Group_M_Crit200'])
        else:
            mass = il.groupcat.loadSubhalos(self._base_path,
                                            self._snapshot_id,
                                            fields=["SubhaloMassType"])[:,4]

            flag = il.groupcat.loadSubhalos(self._base_path,
                                            self._snapshot_id,
                                            fields=["SubhaloFlag"])

        #Fix units
        mass *= 1e10 / self._H

        #Apply criterion
        #Get first a mask for all subhalos which fit to the criterion
        if min_mass is not None and max_mass is not None:
            mass_mask = np.logical_and(mass >= min_mass,
                                       mass <= max_mass)
        elif min_mass is not None:
            mass_mask = mass >= min_mass
        elif max_mass is not None:
            mass_mask = mass <= max_mass
        else:
            mass_mask = np.ones_like(mass)

        if self._grouptype == 'SubFind':
            mask = np.logical_and(mass_mask, flag)
        else:
            mask = mass_mask # add a mask for group['GroupPrimaryZoomTarget'] == 1 
                             # else there are some massive groups that are not zoomed into 

        #Get index of subhalos which are fine
        mask_index = np.nonzero(mask)

        #Get all subhalo ids which are centrals and fullfill requirements
        if self._grouptype == 'SubFind':
            if centrals_only:
                self._galaxies = np.intersect1d(self._centrals, mask_index)
            else:
                self._galaxies = mask_index[0]        
        else:
            self._galaxies = il.groupcat.loadHalos(path+self._simulation+'/output/', 
                                                   self._snapshot_id,
                                                   fields = ['GroupFirstSub']) [mask_index[0]]
        
        if random is True:
            np.random.shuffle(self._galaxies)

        self._num_galaxies = len(self._galaxies)

    @property
    def num_galaxies(self):
        return self._num_galaxies
    
    def __len__(self):
        return self.num_galaxies

    def get_subhalos(self):
        sub = Subhalos.get_subhalo_object(self._simulation)
        return sub(self._galaxies, self._snapshot_id, self._simulation, self._path)
