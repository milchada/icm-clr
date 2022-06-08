# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 20:41:08 2021

Check if the latent space z is a gaussian when propagating the test set through the model
Furthermore write the norm in latenspace for each galaxy into a file

@author: lukas
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy

from torch.cuda.amp import autocast
from scripts.model.load import load_cinn_model
from scripts.util import make_dir

from scripts.data import data
import config as c

#Define path to save posterior
z_norm_path = c.postprocessing_path + "z_norm.npy"  

#Range of X to plot
X_RANGE = 4

def forward():
    """forward (x,c->z) evaluation of the cinn model"""
    
    #Load cinn model
    cinn = load_cinn_model()
    cinn.eval()
    
    z_list = []
    
    #Get test loader
    test_loader = data.get_test_loader(256, labels=True, transform=False, n_views=1, shuffle=False, drop_last=False)
    
    for i, (image, x) in enumerate(test_loader):
        image = torch.cat(image, dim=0)
        image = image.to(c.device)
        x = x.to(c.device)
    
        with torch.no_grad(), autocast(enabled=True):
            z, _ = cinn.forward(x, image)
            z_list.append(z.cpu().numpy())
    
    return np.concatenate(z_list, axis=0)

def get_norm():
    
    z = forward()
    
    return np.linalg.norm(z, 2, axis=1)

def test_z():
    '''Get and plot points in the latent space for the whole test sample'''
    
    z = forward()
    
    #Cov matrix
    cov = np.cov(z.transpose())
    
    fig_cov, axes = plt.subplots(1, 1)
    mat = axes.matshow(cov)
    fig_cov.colorbar(mat)


    #Calculate residual (norm of cov - diag) => should be 0
    residual = np.linalg.norm(cov - np.diag([1]*cov.shape[0]))

    #Plot z    
    num_dim = len(data.x_header)
    #fig_z, axes = plt.subplots(1, num_dim, sharey=True, sharex=True)
    fig_z, ax = plt.subplots(1, 1)
    
    mean = np.mean(z,axis=0)
    std = np.std(z,axis=0)
    
    ylim = 0.5
    
    #Plot ideal line
    x = np.linspace(-X_RANGE, X_RANGE, 100)
    ax.plot(x, scipy.stats.norm.pdf(x), 'k--', label = "Gaussian")
    
    
    for i in range(num_dim):

        #Plot gaussian fit
        mu, st = scipy.stats.norm.fit(z[:,i], loc=0)
        ax.plot(x, scipy.stats.norm.pdf(x, mu, st), label = "Gaussian Fit $z^" + str(i) + "$")
        
    plt.legend()
    plt.grid() 
    plt.show()
    
    figures = (fig_z, fig_cov)
    metrics = {"z_covariance_residual": float(residual),
               "z_max_abs_mean": float(np.max(np.abs(mean))),
               "z_max_std": float(np.max(std))}
    
    return figures, metrics

def plot_norm(norm):
    '''Plot the distribution in space of observables in bins of the latent space norm (the limits are given by norm_bins)'''
    
    header_l = data.l_header
    header = [c.label_dict[h] for h in header_l]
    
    gt = data.l_scaler.inverse_transform(data.df_l_test)
    gt = pd.DataFrame(gt, dtype=np.float32, columns=header)
    
    norm_bins = [0.0, 1.0, 2.0, 100.0]
    gt["norm"] = np.digitize(norm, norm_bins)
    
    g = sns.pairplot(gt, hue="norm", kind="kde", plot_kws=dict(levels=6), corner=True)
    g._legend.remove()
        
    return g


if __name__ == "__main__":
    
    #Create Outputpath if not existing
    make_dir(c.plots_path + "test_z")
    
    figures, metrics = test_z()
    figures[0].savefig(c.plots_path + "test_z/z.pdf")
    figures[1].savefig(c.plots_path + "test_z/z_cov.pdf")
    
    norm = get_norm()
    np.save(z_norm_path, norm)
    
    import json
    with open(c.metrics_path + "test_z.json", "w") as f:
        json.dump(metrics, f)
    
    