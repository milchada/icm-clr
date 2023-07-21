# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 20:14:53 2019

Class to load the data (snapshot-wise) from the catalogues
Return data of the subhalos specified by the subhalo_ids

@author: Lukas Eisert
"""
import illustris_python as il
from scripts.util.redshifts import load_redshift

from functools import partial

import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
import h5py
import glob

def get_subhalo_object(simulation):
    '''
    Factory method to get the correct Simulation object based on the simulation asked for
    
    Use this for simulation dependent adjustments for the code
    '''
        
    if simulation in {"TNG100-1"}:
        return SubhalosTNG100
    else:
        return Subhalos


class Subhalos(object):
    
    def __init__(self, subhalo_ids, snapshot_id, simulation, path, projection=0):
        
        sim_dict = {"TNG50-1": "L35n2160TNG",
                    "TNG100-1": "L75n1820TNG",
                    "TNG300-1": "L205n2500TNG"}

        self._subhalo_ids = subhalo_ids
        self._snapshot_id = snapshot_id
        self._projection = projection
        self._simulation = simulation
        self._full_qualifier = sim_dict[simulation]
        
        #Paths to TNG simulation results
        self._base_path = path + simulation
        self._groupcat_path = self._base_path + "/output"
        self._tree_path = self._base_path + "/postprocessing/"
        self._assembly_path = self._base_path + "/postprocessing/StellarAssembly/galaxies_%03d.hdf5" % (snapshot_id)
        self._circularity_path = self._base_path + "/postprocessing/circularities/circularities_aligned_10Re_" + self._full_qualifier + "%03d.hdf5" % (snapshot_id)
        self._stellar_ages_path = self._base_path + "/postprocessing/stellar_ages/stellar_ages_%03d.hdf5" % (snapshot_id)
        self._stellar_phot_path = self._base_path + "/postprocessing/stellar_light/Subhalo_StellarPhot_p07c_cf00dust_res_conv_z_30pkpc_%03d.hdf5" % (snapshot_id)
        
        #Load additional paths to data which is not available in the basepath
        self.define_merger_paths()
        self.define_auxcat_paths()
        self.define_statmorph_paths()
        
        #Hubble Constant
        self._H = 0.6774
        
        
    #Path definitions
    #-------------------------------------------------------------------------
    def define_merger_paths(self):
        '''Define alternative paths until the cleaned mergercats are in the general repo'''
        clean_merger_base_path = '/u/leisert/mergerhistorycleanup/output/' + self._simulation
        
        self._history_path = clean_merger_base_path + "/postprocessing/MergerHistory/merger_history_cleaned_%03d.hdf5" % (self._snapshot_id)
        self._history_addons_path = clean_merger_base_path + "/postprocessing/MergerHistory/merger_history_addons_cleaned_%03d.hdf5" % (self._snapshot_id)
        self._history_bonus_path = clean_merger_base_path + "/postprocessing/MergerHistory/merger_history_bonus_cleaned_%03d.hdf5" % (self._snapshot_id)
    
    def define_auxcat_paths(self):
        '''Add here additional paths which are not in the base path of the simulation'''
        
        auxcat_path = self._base_path + "/postprocessing/dnelson/auxCat"
        self._stellar_phot_dust_path = self._base_path + "/postprocessing/stellar_light/Subhalo_StellarPhot_p07c_cf00dust_res_conv_z_30pkpc_%03d.hdf5" % (self._snapshot_id)
        self._stellar_halfrad_path = auxcat_path + "/Subhalo_HalfLightRad_p07c_cf00dust_z_%03d.hdf5" % (self._snapshot_id)
        self._stellar_metalicity_path = auxcat_path + "/Subhalo_StellarZ_2rhalf_rBandLumWt_%03d.hdf5" % (self._snapshot_id)
        
    def define_statmorph_paths(self):
        statmorph_path = "/vera/ptmp/gc/vrg/HSC_morph/IllustrisTNG/" +  self._simulation + "/snapnum_%03d" % (self._snapshot_id)
        self._morph_path = lambda x:  statmorph_path + "/morphs_v" + str(x) + ".hdf5"
        
        
    #Basic properties of class
    #-------------------------------------------------------------------------
    @property
    def subhalo_id(self):
        return self._subhalo_ids
    
    @property
    def snapshot_id(self):
        return np.array([self._snapshot_id]*len(self._subhalo_ids))

    @property
    def simulation(self):
        return [self._simulation]*len(self._subhalo_ids)
    
    @property
    def projection(self):
        return np.array([self._projection]*len(self._subhalo_ids))

    #Choose projection 
    #-------------------------------------------------------------------------
    @projection.setter
    def projection(self, projection):
        assert isinstance(projection, int)
        self._projection = projection

    #Functions to load data from the various Catalogues
    #-------------------------------------------------------------------------
    def load_groupcat(self, field):
        return il.groupcat.loadSubhalos(self._groupcat_path,
                                        self._snapshot_id,
                                        fields=[field])[self._subhalo_ids]
    
    def load_auxcat(self, path, field):
        
        #As sometimes the empty fields are np.nan and sometimes completely missing
        #first try it directly and if an "out of range" error occurs use the subhalo ID field
        try:
            with h5py.File(path, 'r') as f:
                result = f[field][self._subhalo_ids]
                
        except ValueError:
            print("WARNING: Value error occured during loading of field: " + field)
            with h5py.File(path, 'r') as f:
                ids = f["subhaloIDs"][:]
                index = np.where(np.isin(ids, self._subhalo_ids))
                result = f[field][index]

                #Not sure if they are always sorted so fix that by "entsorting" the result
                sort_index = np.argsort(self._subhalo_ids)
                result = result[np.argsort(sort_index)]
                
        return result  
    
    def load_statmorph(self, field, band="HSC_R"):  
        
        with h5py.File(self._morph_path(self._projection), 'r') as f:
            #Match ids
            ids = f["subfind_id"][:]
            
            #Care for subhalos which have no avail data
            avail_mask = np.isin(self._subhalo_ids, ids)
            result = np.zeros(len(avail_mask))
            result[~avail_mask] = np.nan
            
            #Now get the data from the file
            index = np.where(np.isin(ids, self._subhalo_ids))[0] 
            assert len(index) == np.sum(avail_mask), "If this is not equal I made an coding error"
            #remove bad fits
            mask = f[band]['flag'][index] >= 2
            result_t = f[band][field][index]
            result_t[mask] = np.nan          
            
            #Now insert the date for the available subhalos
            result[avail_mask] = result_t
            
        return result
    
    def load_merger_history(self, field):
        with h5py.File(self._history_path, 'r') as f:
            result = f[field][self._subhalo_ids]
        return result
    
    def load_merger_history_addons(self, field):
        with h5py.File(self._history_addons_path, 'r') as f:
            result = f[field][self._subhalo_ids]
        return result
    
    def load_merger_history_bonus(self, field):
        with h5py.File(self._history_bonus_path, 'r') as f:
            result = f[field][self._subhalo_ids]
        return result

    def load_stellar_assembly(self, field):
        with h5py.File(self._assembly_path, 'r') as f:
            result = f[field][self._subhalo_ids]
        return result

    def load_circularity(self, field):
        with h5py.File(self._circularity_path, 'r') as f:
            result = f[field][self._subhalo_ids]
        return result
    
    def load_stellar_ages(self, field):
        with h5py.File(self._stellar_ages_path, 'r') as f:
            result = f[field][self._subhalo_ids]
        return result
    
    def load_stellar_phot(self, field):
        with h5py.File(self._stellar_phot_path, 'r') as f:
            result = f[field][self._subhalo_ids]
        return result
    
    #Merger and Assembly Data
    #-------------------------------------------------------------------------
    @property
    def mass_exsitu(self):
        return self.load_stellar_assembly("StellarMassExSitu") * 1e10 / self._H

    @property
    def mass_merger(self):
        return self.load_stellar_assembly("StellarMassFromCompletedMergers") * 1e10 / self._H

    @property
    def mass_merger_major(self):
        return self.load_stellar_assembly("StellarMassFromCompletedMergersMajor") * 1e10 / self._H

    @property
    def mass_merger_major_minor(self):
        return self.load_stellar_assembly("StellarMassFromCompletedMergersMajorMinor") * 1e10 / self._H

    @property
    def mass_stripped(self):
        return (self.load_stellar_assembly("StellarMassFromOngoingMergers") +
                self.load_stellar_assembly("StellarMassFromFlybys")) * 1e10 / self._H

    @property
    def mass_stripped_major(self):
        return (self.load_stellar_assembly("StellarMassFromOngoingMergersMajor") +
                self.load_stellar_assembly("StellarMassFromFlybysMajor")) * 1e10 / self._H

    @property
    def mass_stripped_major_minor(self):
        return (self.load_stellar_assembly("StellarMassFromOngoingMergersMajorMinor") +
                self.load_stellar_assembly("StellarMassFromFlybysMajorMinor")) * 1e10 / self._H

    @property
    def num_major_mergers_last_250myr(self):
        return self.load_merger_history("NumMajorMergersLast250Myr")

    @property
    def num_major_mergers_last_500myr(self):
        return self.load_merger_history("NumMajorMergersLast500Myr")

    @property
    def num_major_mergers_last_gyr(self):
        return self.load_merger_history("NumMajorMergersLastGyr")

    @property
    def num_major_mergers_last_2gyr(self):
        return self.load_merger_history("NumMajorMergersLast2Gyr")

    @property
    def num_major_mergers_last_5gyr(self):
        return self.load_merger_history("NumMajorMergersLast5Gyr")

    @property
    def num_minor_mergers_last_250myr(self):
        return self.load_merger_history("NumMinorMergersLast250Myr")

    @property
    def num_minor_mergers_last_500myr(self):
        return self.load_merger_history("NumMinorMergersLast500Myr")

    @property
    def num_minor_mergers_last_gyr(self):
        return self.load_merger_history("NumMinorMergersLastGyr")

    @property
    def num_minor_mergers_last_2gyr(self):
        return self.load_merger_history("NumMinorMergersLast2Gyr")

    @property
    def num_minor_mergers_last_5gyr(self):
        return self.load_merger_history("NumMinorMergersLast5Gyr")

    @property
    def num_mergers_last_250myr(self):
        return self.load_merger_history("NumMergersLast250Myr")

    @property
    def num_mergers_last_500myr(self):
        return self.load_merger_history("NumMergersLast500Myr")

    @property
    def num_mergers_last_gyr(self):
        return self.load_merger_history("NumMergersLastGyr")

    @property
    def num_mergers_last_2gyr(self):
        return self.load_merger_history("NumMergersLast2Gyr")

    @property
    def num_mergers_last_5gyr(self):
        return self.load_merger_history("NumMergersLast5Gyr")

    #Note: Lookback time is relative to redshift of galaxy in this case!
    #Old implementation, moved the calculation of the lookbacktime to the prepare script
    @property
    def lookback_time_last_maj_merger(self):
        snap_num = self.load_merger_history("SnapNumLastMajorMerger")
        z = load_redshift(snap_num)
        return np.array(cosmo.age(self.z) - cosmo.age(z))
        
    @property
    def snap_num_last_maj_merger(self):
        return self.load_merger_history("SnapNumLastMajorMerger")
    
    @property
    def mass_last_maj_merger(self):
        return self.load_merger_history("MassLastMajorMerger") * 1e10 / self._H
        
    @property
    def mean_merger_gas_fraction(self):
        return self.load_merger_history_bonus("MeanGasFraction")
    
    @property
    def mean_merger_lookback_time(self):
        return self.load_merger_history_bonus("MeanLookbackTime")

    @property
    def mean_merger_mass_ratio(self):
        return self.load_merger_history_bonus("MeanMassRatio")


    #RootDescendantID
    #For spliting according to Branches
    #-------------------------------------------------------------------------
    @property
    def root_descendant_id(self):
        treeName = "SubLink"
        
        search_path = il.sublink.treePath(self._tree_path, treeName, '*')
        numTreeFiles = len(glob.glob(search_path))
        
        offsetFile = il.sublink.offsetPath(self._tree_path, self._snapshot_id)
        prefix = 'Subhalo/' + treeName + '/'

        with h5py.File(offsetFile, 'r') as f:
            # load the merger tree offsets of this subgroup
            RowNum = f[prefix+'RowNum'][self._subhalo_ids]
            
        #Get offsets
        offsets = il.sublink.subLinkOffsets(self._tree_path, treeName)
    
        # find the tree file chunk containing this row
        rowOffsets = RowNum[:,np.newaxis] - offsets
        
        def get_fileNum(exception_value, x):
            try:
                index = np.where(x>=0)
                return np.max(index)
            except ValueError:
                #Set to an index which is never reached by the loop, i.e. they stay nan
                print("Weird Value Error when loading the root_descendant_id")
                return exception_value
        
        fileNum_Off = np.apply_along_axis(partial(get_fileNum, 0), 1, rowOffsets)
        fileNum = np.apply_along_axis(partial(get_fileNum, numTreeFiles), 1, rowOffsets)

        #Calculate the index within the files
        fileOff = rowOffsets[range(len(rowOffsets)), fileNum_Off]
        
        #Now go through the files
        result = np.full_like(self._subhalo_ids, np.nan, dtype=int)
        
        for i in range(numTreeFiles):
            
            #Look only at subhalos which are contained in this file
            file_index = np.where(i==fileNum)
            
            with h5py.File(il.sublink.treePath(self._tree_path, treeName, i),'r') as f:
                
                #Get data (sort because h5py insists on an increasing order)
                file_offset = fileOff[file_index]
                j = np.argsort(file_offset)
                sort = file_offset[j]
                file_rd_id = f['RootDescendantID'][sort]
                
                #Now reverse the sorting
                file_rd_id = file_rd_id[np.argsort(j)]
                
            result[file_index] = file_rd_id
            
        return result
    

    #Integrated properties of Subhalo
    #-------------------------------------------------------------------------
    @property
    def z(self):
        return load_redshift(self.snapshot_id)

    @property
    def a(self):
        return 1.0/(1.0 + self.z)
    
    @property
    def luminosity_distance(self):
        return cosmo.luminosity_distance(self.z).to(u.pc).value
    
    @property
    def distance_modulus(self):
        return 5*np.log10(self.luminosity_distance/10.0)
    
    @property
    def lookback(self):
        zeros = [0]*len(self._subhalo_ids)
        return np.array(cosmo.age(zeros) - cosmo.age(self.z))
    
    @property
    def num_dm_particles(self):
        return self.load_groupcat("SubhaloLenType")[:,1]
    
    @property
    def stellar_mass(self):
        return self.load_groupcat("SubhaloMassType")[:,4] * 1e10 / self._H

    @property
    def i_band_mag_dust_apparent(self):
        return self.load_auxcat(self._stellar_phot_dust_path, "Subhalo_StellarPhot_p07c_cf00dust_res_conv_z_30pkpc")[:,3]
    
    @property
    def r_band_mag_dust_apparent(self):
        return self.load_auxcat(self._stellar_phot_dust_path, "Subhalo_StellarPhot_p07c_cf00dust_res_conv_z_30pkpc")[:,2]

    @property
    def g_band_mag_dust_apparent(self):
        return self.load_auxcat(self._stellar_phot_dust_path, "Subhalo_StellarPhot_p07c_cf00dust_res_conv_z_30pkpc")[:,1]
    
    @property
    def i_band_mag_dust(self):
        return self.i_band_mag_dust_apparent - self.distance_modulus
    
    @property
    def i_band_mag_apparent(self):
        return self.i_band_mag + self.distance_modulus
    
    @property
    def color_dust(self):
        return self.g_band_mag_dust_apparent - self.r_band_mag_dust_apparent

    @property
    def z_band_mag(self):
        return self.load_groupcat("SubhaloStellarPhotometrics")[:,7]
    
    @property
    def i_band_mag(self):
        return self.load_groupcat("SubhaloStellarPhotometrics")[:,6]
    
    @property
    def r_band_mag(self):
        return self.load_groupcat("SubhaloStellarPhotometrics")[:,5]

    @property
    def g_band_mag(self):
        return self.load_groupcat("SubhaloStellarPhotometrics")[:,4]
    
    @property
    def color(self):
        return self.g_band_mag - self.r_band_mag

    @property
    def metallicity_star(self):
        return self.load_auxcat(self._stellar_metalicity_path, "Subhalo_StellarZ_2rhalf_rBandLumWt")
    
    @property
    def metallicity_gas(self):
        return self.load_groupcat("SubhaloStarMetallicityHalfRad")

    @property
    def sfr(self):
        return self.load_groupcat("SubhaloSFRinHalfRad")
    
    @property
    def log_sfr(self):
        '''Log the sfr, replace 0 by minimum'''
        sfr = np.array(self.sfr)
        mask = (sfr == 0)
        min_sfr = np.min(sfr[~mask])
        sfr[mask] = min_sfr
        return np.log10(sfr)
    
    @property
    def velocity_dispersion(self):
        return self.load_groupcat("SubhaloVelDisp")

    @property
    def half_mass_rad_physical(self):
        return self.load_groupcat("SubhaloHalfmassRadType")[:,4] * self.a / self._H
    
    @property
    def half_light_rad(self):
        return self.load_auxcat(self._stellar_halfrad_path, "Subhalo_HalfLightRad_p07c_cf00dust_z")[:,2] * self.a / self._H

    @property
    def mass_in_rad(self):
        return self.load_groupcat("SubhaloMassInRadType")[:,4] * 1e10 / self._H
    
    @property
    def stellar_age_2rhalf_lumw(self):
        return self.load_stellar_ages("Subhalo_StellarAge_2rhalf_rBandLumWt")

    @property
    def fraction_disk_stars(self):
        return self.load_circularity("CircAbove07Frac")[:,0]
 

    #Statmorph parameters
    #-------------------------------------------------------------------------
    @property
    def asymmetry(self):
        return self.load_statmorph('asymmetry')
    
    @property
    def concentration(self):
        return self.load_statmorph('concentration')
    
    @property
    def deviation(self):
        return self.load_statmorph('deviation')
    
    @property
    def gini(self):
        return self.load_statmorph('gini')
    
    @property
    def m20(self):
        return self.load_statmorph('m20')
    
    @property
    def multimode(self):
        return self.load_statmorph('multimode')
    
    @property
    def sersic_n(self):
        return self.load_statmorph('sersic_n')
    
    @property
    def sersic_rhalf(self):
        return self.load_statmorph('sersic_rhalf')
    
    @property
    def sersic_ellip(self):
        return self.load_statmorph('sersic_ellip')
    
    @property
    def smoothness(self):
        return self.load_statmorph('smoothness')
    
    
# Here one can add special treatment of the different simulations. However, try to keep the code below this line as simple as possible!
# In an ideal world there should be nothing here at all
#-----------------------------------------------------------------------------------------------------------------------------
        
class SubhalosTNG100(Subhalos):
    
    def define_auxcat_paths(self):
        auxcat_path = "/ptmp/leisert/image_cache/TNG100-1/auxCat"
        self._stellar_phot_dust_path = auxcat_path + "/Subhalo_StellarPhot_p07c_cf00dust_res_conv_z_30pkpc_%03d.hdf5" % (self._snapshot_id)
        self._stellar_halfrad_path = auxcat_path + "/Subhalo_HalfLightRad_p07c_cf00dust_z_%03d.hdf5" % (self._snapshot_id)
        self._stellar_metalicity_path = auxcat_path + "/Subhalo_StellarZ_2rhalf_rBandLumWt_%03d.hdf5" % (self._snapshot_id)