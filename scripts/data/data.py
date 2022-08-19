"""
Helper Class to load the data from the dataset folder

x: unobservable
l: observable
m: metadata; i.e. everything that was extracted from the simulation in physical quantities

Note that an import of this file returns an instance of the class Data

@author: Lukas Eisert
"""

import torch
from torch.utils.data import DataLoader

import pandas as pd
from pickle import load
import numpy as np

import yaml
import sys

import config as c


class Data():
    
    def __init__(self):
        
        #Load Default Params
        data_params = yaml.safe_load(open('params.yaml'))['data']
        self.__valid_range = data_params['VALID_RANGE']
        self.__num_workers = data_params['NUM_WORKERS']
        self.__image_size = data_params['IMAGE_SIZE']
        self.__dataset_classname = data_params['DATASET'] 
    
    @property
    def valid_range(self):
        '''Get valid range in phyiscal units'''
        return self.__transform_valid_range(self.__valid_range, self.x_scaler)[0]
    
    @property
    def valid_range_scaled(self):
        '''Get valid range in scaled units'''
        return self.__transform_valid_range(self.__valid_range, self.x_scaler)[1]

        
    def __transform_valid_range(self, vr, scaler):
        ''' Scale the valid range to model scale and replace the yaml-string-None with a python-None'''

        out = []
        scaled_out = []

        for i, r in enumerate(vr):

            if r == 'None':
                out.append(None)
                scaled_out.append(None)

            else:
                assert len(r) == 2, 'ERROR: VALID_RANGE has to be either None or a list with a len of 2'
                tmp = np.zeros([2, len(vr)])
                tmp[:, i] = r
                tmp = scaler.transform(tmp)[:, i]

                out.append(r)
                scaled_out.append(tmp)

        return out, scaled_out


    #Load Scalers physical -> scaled
    #-----------------------------------------------------
    @property
    def x_scaler(self):
        return load(open(c.dataset_path + "x_scaler.pkl", 'rb'))
    
    @property
    def l_scaler(self):
        return load(open(c.dataset_path + "y_scaler.pkl", 'rb'))
    
    
    #Load Dataframes
    # x = Scaled Unobservables
    # l = Scaled Observables
    # m = Non-Scaled (Physical) Metadata
    #-----------------------------------------------------
    @property
    def df_x_test(self):
        return pd.read_csv(c.dataset_path + 'x_test.csv')
    
    @property
    def df_l_test(self):
        return pd.read_csv(c.dataset_path + 'y_test.csv')
    
    @property
    def df_m_test(self):
        return pd.read_csv(c.dataset_path + 'm_test.csv')
    
    @property
    def df_x_val(self):
        return pd.read_csv(c.dataset_path + 'x_val.csv')
    
    @property
    def df_l_val(self):
        return pd.read_csv(c.dataset_path + 'y_val.csv')
    
    @property
    def df_m_val(self):
        return pd.read_csv(c.dataset_path + 'm_val.csv')
    
    @property
    def df_x_train(self):
        return pd.read_csv(c.dataset_path + 'x_train.csv')
    
    @property
    def df_l_train(self):
        return pd.read_csv(c.dataset_path + 'y_train.csv')
    
    @property
    def df_m_train(self):
        return pd.read_csv(c.dataset_path + 'm_train.csv')
    
    
    #Overall sets (for general population studies)
    @property
    def df_x(self):
        return pd.concat([self.df_x_train, self.df_x_val, self.df_x_test])
    
    @property
    def df_l(self):
        return pd.concat([self.df_l_train, self.df_l_val, self.df_l_test])
    
    @property
    def df_m(self):
        return pd.concat([self.df_m_train, self.df_m_val, self.df_m_test])
    
    
    #Test set Ground Truth for model evaluation
    #ie scaled to physical values and replaced non valid datapoints with np.nan
    @property
    def gt(self):
        gt = self.x_scaler.inverse_transform(self.df_x_test)

        for i, vr in enumerate(self.valid_range):
            if vr is not None:
                mask = np.logical_or(gt[:,i] < vr[0], gt[:,i] > vr[1])
                gt[:,i][mask] = np.nan
                
        return gt
    
    #Ground Truth for model evaluation for all galaxies in the 3 subsets
    @property
    def gt_all(self):
        gt = self.x_scaler.inverse_transform(self.df_x)

        for i, vr in enumerate(self.valid_range):
            if vr is not None:
                mask = np.logical_or(gt[:,i] < vr[0], gt[:,i] > vr[1])
                gt[:,i][mask] = np.nan
                
        return gt

    #Load Headers
    #-----------------------------------------------------
    @property
    def x_header(self):
        return list(self.df_x_test.columns)
    
    @property
    def l_header(self):
        return list(self.df_l_test.columns)
    
    @property
    def m_header(self):
        return list(self.df_m_test.columns)

    #Load Label (i.e. human friendly for plots)
    #-----------------------------------------------------
    @property
    def x_label(self):
        return list(map(c.label_dict.get, self.x_header))
    
    @property
    def l_label(self):
        return list(map(c.label_dict.get, self.l_header))
    
    @property
    def m_label(self):
        return list(map(c.label_dict.get, self.m_header))
    
    #Dimensionality of the problem/data and therefore model
    #-----------------------------------------------------
    @property
    def NUM_DIM(self):
        return len(self.x_header)
    
    @property
    def NUM_COND(self):
        return len(self.l_header)
    
    @property
    def IMAGE_SIZE(self):
        return self.__image_size
    

    #Pytorch data loader for batch-wise multithreaded loading of images
    #-----------------------------------------------------
    def get_loader(self, batch_size, labels, augmentation, n_views, shuffle, drop_last, meta_path, label_path):
        
        module = __import__("scripts.data.datasets", globals(), locals(), self.__dataset_classname)
        dataset_object = getattr(module, self.__dataset_classname)
        
        if labels:
            dataset = dataset_object(meta_path, label_file=label_path, augmentation=augmentation, n_views=n_views)
        else:
            dataset = dataset_object(meta_path, augmentation=augmentation, n_views=n_views)
    
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                      num_workers=self.__num_workers, pin_memory=True, drop_last=drop_last)
        
        return loader
        
    def get_train_loader(self, batch_size, labels, augmentation, n_views, shuffle, drop_last):
        return self.get_loader(batch_size, labels, augmentation, n_views, shuffle, drop_last, c.dataset_path + 'm_train.csv', c.dataset_path + 'x_train.csv')
    
    def get_domain_loader(self, batch_size, labels, augmentation, n_views, shuffle, drop_last):
        return self.get_loader(batch_size, labels, augmentation, n_views, shuffle, drop_last, c.dataset_path + 'm_domain.csv', c.dataset_path + 'x_domain.csv')
    
    def get_val_loader(self, batch_size, labels, augmentation, n_views, shuffle, drop_last):
        return self.get_loader(batch_size, labels, augmentation, n_views, shuffle, drop_last, c.dataset_path + 'm_val.csv', c.dataset_path + 'x_val.csv')
    
    def get_test_loader(self, batch_size, labels, augmentation, n_views, shuffle, drop_last):
        return self.get_loader(batch_size, labels, augmentation, n_views, shuffle, drop_last, c.dataset_path + 'm_test.csv', c.dataset_path + 'x_test.csv')

#When importing this module, the module is replaced by an instance of the class Data
# => the class is transparent to the outside 
sys.modules[__name__] = Data()