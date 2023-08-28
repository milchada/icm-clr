# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:28:32 2020

Short help script to get the redshift from the snap num using a file 
containing the list of redshifts for each snapshot

@author: Lukas Eisert
"""

import numpy as np

TNG_PATH = "/virgo/simulations/IllustrisTNG/TNG100-1/output"
REDSHIFT_PATH = "./scripts/util/RedshiftsIllustrisTNG.txt"


def save_redshift():
    '''Get Redshift from simulation header and write it in a txt file'''
    z = np.zeros(100)
    
    import illustris_python as il
    
    for i in range(100):
        z[i] = il.groupcat.loadHeader(TNG_PATH, i)["Redshift"]
    
    np.savetxt(REDSHIFT_PATH, z, fmt="%.6f")
    
    
def load_redshift(snapnum=None):
    '''Return z for all z (snapnum=None) or a list of snapnums'''
    z = np.loadtxt(REDSHIFT_PATH)
    
    if snapnum is None:
        return z
    else:
        #Set negative napnum to 0
        snapnum = np.clip(snapnum, 0, None)
        return z[snapnum]
