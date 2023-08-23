# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:13:43 2021

Sample the cINN posterior

@author: Lukas Eisert
"""

import torch
import numpy as np
from tqdm import tqdm
import os

import config as c

from scripts.model.load import load_cinn_model
import scripts.data.data as data
from torch.cuda.amp import autocast
  

# Method(s) to sample the posterior
#----------------------------------------------------------------------------
def sample_posterior(l, N, cinn):
    ''' Return N physical posterior samples of a single galaxy given the observables l and the model cinn '''
    
    #Prepare model for evaluation
    cinn.to(c.device)
    cinn.eval()
    
    l = np.array(l)
    
    l = np.array([l,]*N)
    
    l = torch.FloatTensor(l).to(c.device)
    z = torch.randn(N, data.NUM_DIM).to(c.device)

    with torch.no_grad(), autocast(enabled=True):
        samples = cinn.reverse_sample(z, l).cpu().numpy()
    
    #Scale back to physical quantities
    samples = data.x_scaler.inverse_transform(samples)
    
    #Replace samples outside the valid range with nans
    for i, vr in enumerate(data.valid_range):
        if vr is not None:
            mask = np.logical_or(samples[:,i] < vr[0], samples[:,i] > vr[1])
            samples[:,i][mask] = np.nan
        
    return samples

def sample_all_posteriors(N, cinn):
    ''' Return N physical posterior samples for all galaxies in the test set given the model cinn '''
    
    #Prepare model for evaluation
    cinn.to(c.device)
    cinn.eval()
    
    #Get dataloader
    BATCHSIZE = 1
    test_loader = data.get_test_loader(BATCHSIZE, labels=False, augmentation=None, n_views=1, shuffle=False, drop_last=False) 
    posterior = np.zeros([len(test_loader), N, data.NUM_DIM])
    
    with torch.no_grad(), autocast(enabled=True):
        #Loop over images
        for j, image in tqdm(enumerate(test_loader)):
            #Get N samples
            z = torch.randn(N, data.NUM_DIM).to(c.device)
            image = image[0].to(c.device)
            image = torch.cat(N*[image])
            posterior[j, :, :] = cinn.reverse_sample(z, image).cpu().numpy()

            #Scale back to physical quantities
            posterior[j, :, :] = data.x_scaler.inverse_transform(posterior[j, :, :])
    
    #Replace samples outside the valid range with nans
    for i, vr in enumerate(data.valid_range):
        if vr is not None:
            mask = np.logical_or(posterior[:,:,i] < vr[0], posterior[:,:,i] > vr[1])
            posterior[:,:,i][mask] = np.nan
            
    return posterior

    
if __name__ == "__main__":
    
    #Load parameters
    import yaml
    params = yaml.safe_load(open('params.yaml'))
    sample_posterior_params = params['sample_posterior']
    NUM_SAMPLES = sample_posterior_params['NUM_SAMPLES']
    
    if not os.path.exists(c.postprocessing_path):
        os.makedirs(c.postprocessing_path)
    
    #Load Model
    cinn = load_cinn_model()   
    
    #Sample posteriors
    post = sample_all_posteriors(NUM_SAMPLES, cinn)
    
    #Save them 
    np.save(c.posterior_path, post)




