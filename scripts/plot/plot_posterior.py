# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 20:08:14 2021

@author: lukas
"""

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from scripts.postprocessing.sample_posterior import sample_posterior
from scripts.model.load import load_cinn_model
from scripts.util import kde

from scripts.data import data 
import config as c


#Fix fondsize
font = {'family' : 'Roboto',
        'sans-serif': ['Roboto-Light'],
        'size'   : 7}
plt.rc('font', **font)


def plot_posteriors(l, x_gt=None, id_label=None, num_samples=200, flat_prior=False):
    """
    Plot posteriors

    :param l:    Observables of galaxies to predict posteriors for (first dim => galaxies; second dim => unobservables); in SCALED units
    :param x_gt: Ground truth unobservables; if None no Ground Truth is plotted; in PHYSICAL units
    :param id_label: additional information to plot on the right edge of each plot; tuple (subfind_id, redshift)
    :param num_samples: number of posterior samples to draw for each galaxy
    :param flat_prior: EXPERIMENTAL; use a flat prior i.e. devide by the overall TNG prior
    
    """ 
    
    #Number of gridpoints used for the kde
    NUM_KDE_GRIDPOINTS = 200

    #Number of galaxies to plot and ensure that other inputs have also the correct dimension
    n_plots = l.shape[0]
    
    if x_gt is not None:
        assert x_gt.shape[0] == n_plots, "Number of samples for observables ({}) and unobservables ({}) differ!".format(n_plots, x_gt.shape[0])
        
    if id_label is not None:
        sub_id = id_label[0]
        z = id_label[1]
        
        assert len(sub_id) == n_plots
        assert len(z) == n_plots
    
    #Get model
    model = load_cinn_model()
    
    #Get physical prior
    prior = data.df_x_test.to_numpy()
    prior_physical = data.x_scaler.inverse_transform(prior)
    
    #Init plot
    fig, axes = plt.subplots(n_plots, data.NUM_DIM, sharey=True, sharex='col', figsize=(16, n_plots), gridspec_kw = {'wspace':0.2, 'hspace':0.3})

    #Loop through all #n_plots galaxies given
    for i in tqdm(range(n_plots), total=n_plots, disable=False):
        
        #Samle the posterior distribution 
        posteriors = sample_posterior(l[i], num_samples, model)
        
        #Plot each single field
        for j, label in enumerate(data.x_label):
            if n_plots > 1:
                ax = axes[i][j]
            else:
                ax = axes[j]
            
            #Deal with the valid range if applicable
            #Ie plot a bar at the right edge as additional bin for the non valid points
            if data.valid_range[j] is not None:
                axins = ax.inset_axes([1, 0.0, 0.1, 1])
                vr = data.valid_range[j]
                
                #Posteriors
                mask = np.isnan(posteriors[:,j])
                valid_posteriors = posteriors[:,j][~mask]
                frac_non_valid_post = mask.sum()/len(posteriors[:,j]) #Note: Fraction of non valid posterior samples
                axins.fill_between([0,1],[frac_non_valid_post, frac_non_valid_post], color="blue")
                
                #Priors
                mask = np.bitwise_or(prior_physical[:, j] < vr[0], prior_physical[:, j] > vr[1])
                valid_prior = prior_physical[:, j][~mask]
                frac_non_valid_pri = 1 - len(valid_prior)/len(prior_physical[:, j])
                axins.plot([0,1], [frac_non_valid_pri, frac_non_valid_pri], color='grey', lw=2)
                
                #If number of points outsite the valid range => its the MAP
                if frac_non_valid_post >= 0.5:
                    axins.plot([0.66,0.66], [0., 1.], color='orange', lw=2)
                
                #If gt is outsite the valid range => its the gt
                if x_gt is not None:
                    if x_gt[i,j] == np.nan:
                        axins.plot([0.33,0.33], [0., 1.], color='red', lw=2)
                
                axins.set_ylim(0,1)
                axins.set_yticklabels([])
                axins.set_xticklabels([])
            
            #Plot Prior
            if data.valid_range[j] is not None:
                prior_xs, prior_dxs = kde(valid_prior, N=NUM_KDE_GRIDPOINTS)
            else:
                prior_xs, prior_dxs = kde(prior_physical[:, j], N=NUM_KDE_GRIDPOINTS)
            ax.plot(prior_xs, prior_dxs, color='grey', lw=2)
            
            #Get Posterior
            if data.valid_range[j] is not None:
                post_xs, post_dxs = kde(valid_posteriors, N=NUM_KDE_GRIDPOINTS)
            else:
                post_xs, post_dxs = kde(posteriors[:,j], N=NUM_KDE_GRIDPOINTS)
            
            #Calculate the posterior for a flat prior (i.e. divide by the original TNG prior)
            if flat_prior:
                from scipy.interpolate import interp1d
                #Interpolate posterior and prior on common grid
                prior = interp1d(prior_xs, prior_dxs, bounds_error=False, fill_value=0)
                post = interp1d(post_xs, post_dxs, bounds_error=False, fill_value=0)
                post_xs = np.linspace(np.min([post_xs, prior_xs]), np.max([post_xs, prior_xs]), NUM_KDE_GRIDPOINTS)
                prior = prior(post_xs)
                post = post(post_xs)
                #Divide posterior by prior and renormalize
                post_dxs = np.divide(post, prior, out=np.zeros_like(post), where=prior!=0)
                post_dxs /= np.max(post_dxs)
                
            #Plot Posterior
            ax.plot(post_xs, post_dxs, color='blue', lw=2)
            
            #Plot Ground Truth
            if x_gt is not None:
                ax.plot([x_gt[i,j], x_gt[i,j]], [0,1], color='red', lw=2)
            
            #Plot MAP
            if data.valid_range[j] is None or frac_non_valid_post < 0.5:
                index = np.argmax(post_dxs, axis=0)
                peak_position = post_xs[index]
                ax.plot([peak_position, peak_position], [0,1], color='orange', lw=2)
    
            #Enforce xlim
            if data.valid_range[j] is None:
                ax.set_xlim([np.min(prior_physical[:,j]), np.max(prior_physical[:,j])])
            else:
                ax.set_xlim(data.valid_range[j])
                
            #Enforce ylim
            ax.set_ylim(0,1)
            ax.get_yaxis().set_visible(False)
                
            #Add xlabel, adjust the hight in y direction to avoid overlap of the labels
            if i == n_plots - 1:
                ax.set_xlabel(label)

            #Add the subfind id to the right
            if j == 0 and id_label is not None:
                txt = "z={:.2f} \n {:n}".format(z[i], sub_id[i])
                ax.text(-0.10, 0.5, txt, color='k',
                                         horizontalalignment='center',
                                         verticalalignment='center',
                                         transform=ax.transAxes,
                                         rotation='vertical',
                                         size=12)
                
    plt.show()
    return fig


def plot_random_posteriors(n_plots):
    """
    Plot posterior examples for random galaxies out of the test set (i.e. the first ones)

    :param n_plots: Number of random galaxies to plot
    
    """ 

    #Get ground truth 
    x_gt = data.gt
    l_gt_loader = data.get_test_loader(n_plots, labels=False, augmentation=False, n_views=1, shuffle=False, drop_last=False)
    sub_id = data.df_m_test["subhalo_id"]
    z = data.df_m_test["z"]
    
    #Restrict to first #n_plots galaxies
    x_gt = x_gt[:n_plots]
    l_gt = next(iter(l_gt_loader))[0]
    sub_id = sub_id[:n_plots]
    z = z[:n_plots]

    return plot_posteriors(l_gt, x_gt=x_gt, id_label=(sub_id, z))


if __name__ == "__main__":

    fig = plot_random_posteriors(n_plots=25)
    fig.savefig(c.plots_path + "posterior_example.pdf")
