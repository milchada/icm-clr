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
    

    #Pytorch data loader
    #-----------------------------------------------------
    def get_train_loader(self, batch_size):
        x_train = torch.Tensor(self.df_x_train.to_numpy())
        l_train = torch.Tensor(self.df_l_train.to_numpy())
        
        train_data = torch.utils.data.TensorDataset(x_train, l_train)
        
        train_loader  = DataLoader(train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   drop_last=True)
        
        return train_loader
    
    def get_val_loader(self, batch_size):
        x_val = torch.Tensor(self.df_x_val.to_numpy())
        l_val = torch.Tensor(self.df_l_val.to_numpy())
        
        val_data = torch.utils.data.TensorDataset(x_val, l_val)
        
        val_loader  = DataLoader(val_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 drop_last=True)
        
        return val_loader
    
    def get_test_loader(self, batch_size):
        x_test = torch.Tensor(self.df_x_test.to_numpy())
        l_test = torch.Tensor(self.df_l_test.to_numpy())
        
        test_data = torch.utils.data.TensorDataset(x_test, l_test)
        
        test_loader  = DataLoader(test_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        
        return test_loader
        

#When importing this module, the module is replaced by an instance of the class Data
# => the class is transparent to the outside 
sys.modules[__name__] = Data()