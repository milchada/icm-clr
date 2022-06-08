# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:19:51 2021

@author: Lukas Eisert
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from scipy import stats

import config as c 
from config import label_dict
from scripts.util import make_dir
from scripts.data import data

from scripts.postprocessing.peak_detection import map_path, peak_number_path, peak_position_path
from scripts.postprocessing.sample_posterior import posterior_path

font = {'size'   : 26,
        'family' : 'Roboto'}
        #'sans-serif': ['Roboto-Light']
plt.rc('font', **font)

#Number of samples for the posterior plot
NUM_SAMPLES = 400

#Bin number for histograms 
BINS = 40

#Bin number for statistic evaluation
STATS_BIN = 10

#Parameter for the Plotting
FIG_SIZE = 10
LINE_WIDTH = 14
CMAP = "viridis"
MEAN_C = "k"
STD_C = "k"

#Output path
output_path = c.plots_path + "map/"


def plot_map(gt, pre, header, sigma=None, vmin=0.1, vmax=10, sample_mode=False, error_mode=False, sigma_mode=False, sigma_error_mode=False):
    """
    Function to plot MAPs, Errors and Sigmas
    This function is able to handle NaNs by plotting them into extra bins

    :param gt: ground truth
    :param pre: predicted model MAPs or posterior samples
    :param header: header with title for each field
    :param sigma: standard Deviation, same shape as gt without NaNs (use np.nanstd)
    :param vmin: min num galaxies per bin
    :param vmax: max num galaxies per bin
    :param sample_mode: Plot cinn samples instead of maps i.e. pre expects the posteriors
    :param error_mode: Plot the error on the y-axis instead of the MAP
    :param sigma_mode: Plot the posterior std on the y-axis instead of the MAP (sigma is needed as input)
    :param sigma_error_mode: Plot the posterior std on the y-axis and the MAP error on the x-axis (sigma is needed as input)
    """ 
    
    # Ensure that the colorbar is not messed up if we pass samples instead of maps
    if sample_mode:
        vmin *= NUM_SAMPLES
        vmax *= NUM_SAMPLES

    fig, axes = plt.subplots(1, len(header),
                             figsize=(FIG_SIZE*len(header),FIG_SIZE))
    
    plt.subplots_adjust(wspace=0.05)
    
    #Init log norm
    log_norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    
    for i, out_label in enumerate(header):
        
        #Further Plot Parameters
        DIAGONAL = False #Diagonal line in plot, i.e. the x and y axis have the same dimension
        DOUBLE_NAN = True #Plot two NaN columns if True, otherwise only one for the gt

        #Define quanties to plot
        if error_mode:
            error = pre[:,i] - gt[:,i]
            X = gt[:,i]
            Y = error
        elif sigma_error_mode:
            error = np.abs(pre[:,i] - gt[:,i])
            X = error
            Y = sigma[:,i]
            DOUBLE_NAN = False
        elif sigma_mode:
            X = gt[:,i]
            Y = sigma[:,i]
            DOUBLE_NAN = False
        else:
            X = gt[:,i]
            Y = pre[:,i]
            DIAGONAL = True
        
        #Get mask for NaN Galaxies and a flag which is true if there are NaNs
        nan_x_mask = np.isnan(X)
        nan_y_mask = np.isnan(Y)
        nan_mask = np.logical_or(nan_x_mask, nan_y_mask)
        NAN_FLAG = (nan_mask.sum() > 0)
            
        #Define limits
        xlim = (np.nanquantile(X, 0.001), np.nanquantile(X, 0.999))
        if DIAGONAL:
            ylim = xlim
        else:
            ylim = (np.nanquantile(Y, 0.001), np.nanquantile(Y, 0.999))     

        #Get the axis for that plot
        ax = axes[i]
        
        #Construct bins
        xbins = np.linspace(*xlim, BINS)
        ybins = np.linspace(*ylim, BINS)
            
        #Replace the nans (if avail) with a number to plot them into the same hist
        if NAN_FLAG:
            
            #Padding between the outermost datapoints in the 2d hist and the additional bins
            PADDING = 2 * np.abs(xlim[1] - xlim[0])/BINS
            #Size of the additional bins in fraction of the overall plot size
            BIN_SIZE = 0.12
            
            extra_x_bin_size = BIN_SIZE * np.abs(xlim[1] - xlim[0])
            extra_y_bin_size = BIN_SIZE * np.abs(ylim[1] - ylim[0])
            
            extra_x_bin = xlim[1] + extra_x_bin_size/2. + PADDING
            extra_y_bin = ylim[1] + extra_y_bin_size/2. + PADDING
            
            lower_x_edge = extra_x_bin - extra_x_bin_size/2
            upper_x_edge = extra_x_bin + extra_x_bin_size/2
            lower_y_edge = extra_y_bin - extra_y_bin_size/2
            upper_y_edge = extra_y_bin + extra_y_bin_size/2
            
            xbins = np.append(xbins, lower_x_edge)
            xbins = np.append(xbins, upper_x_edge)
            
            if DOUBLE_NAN:
                ybins = np.append(ybins, lower_y_edge)
                ybins = np.append(ybins, upper_y_edge)
            
            X = np.nan_to_num(X, nan=extra_x_bin)
            Y = np.nan_to_num(Y, nan=extra_y_bin)
            
            NUM_TICK_STEP = 6
            
            #Add NMM labels, ticks and fix the plot lims to include the new bin
            TICK_STEP = np.abs(xlim[1] - xlim[0])/NUM_TICK_STEP
            xticks = np.arange(*xlim, step=TICK_STEP)
            
            if TICK_STEP > 2:
                xticks = xticks.astype(int)
                xticklabels = xticks.astype(int)
            else:
                xticklabels = ["%.1f" % i for i in xticks]
                
            xticks = np.append(xticks, extra_x_bin)
            xticklabels = np.append(xticklabels, "NMM")
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            xlim = (xlim[0], upper_x_edge)

            
            #Add the y axis extra bin only if the FLAG is set
            if DOUBLE_NAN:
                TICK_STEP = np.abs(ylim[1] - ylim[0])/NUM_TICK_STEP
                yticks = np.arange(*ylim, step=TICK_STEP)
                
                if TICK_STEP > 2:
                    yticks = yticks.astype(int)
                    yticklabels = yticks.astype(int)
                else:
                    yticklabels = ["%.1f" % i for i in yticks]
                    
                yticks = np.append(yticks, extra_y_bin)
                yticklabels = np.append(yticklabels, "NMM")
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
                ylim = (ylim[0], upper_y_edge)
        
        
        #Plot histogram
        _, _, _, h = ax.hist2d(X, Y,
                               bins=[xbins, ybins],
                               norm=log_norm,
                               cmap=CMAP,
                               cmin=vmin)

        
        
        # Further stuff for the nan plots
        if NAN_FLAG:
            
            #Add some black lines to make it look nicer
            ax.plot([lower_x_edge,lower_x_edge], [ylim[0], upper_y_edge], lw=LINE_WIDTH//2, c='k')
            
            if DOUBLE_NAN:
                ax.plot([xlim[0], upper_x_edge], [lower_y_edge,lower_y_edge], lw=LINE_WIDTH//2, c='k')
            
            #Write fractions of points which fall correctly and wrongly into the nan bins
            if DOUBLE_NAN:
                frac_corr_nan = np.logical_and(nan_x_mask, nan_y_mask).sum() / len(gt[:,i]) 
                frac_gt_nan = np.logical_and(nan_x_mask, ~nan_y_mask).sum() / len(gt[:,i]) 
                frac_pre_nan = np.logical_and(~nan_x_mask, nan_y_mask).sum() / len(gt[:,i])
                frac_corr_non_nan = np.logical_and(~nan_x_mask, ~nan_y_mask).sum() / len(gt[:,i])
                
                ax.text(0.1, 0.8, "{:.0f}%".format(frac_corr_non_nan*100), color='k', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.text(extra_x_bin, extra_y_bin, "{:.0f}%".format(frac_corr_nan*100), color='k', horizontalalignment='center', verticalalignment='center')
                ax.text(extra_x_bin, (ylim[1]+ylim[0])/2, "{:.0f}%".format(frac_gt_nan*100), color='k', horizontalalignment='center', verticalalignment='center', rotation='vertical')
                ax.text((xlim[1]+xlim[0])/2, extra_y_bin, "{:.0f}%".format(frac_pre_nan*100),color='k', horizontalalignment='center', verticalalignment='center')
            else:
                frac_gt_nan = nan_x_mask.sum() / len(gt[:,i])
                ax.text(extra_x_bin, (ylim[1]+ylim[0])/2, "{:.0f}%".format(frac_gt_nan*100), color='k', horizontalalignment='center', verticalalignment='center', rotation='vertical')
            
            
        #Add further lines for median and certain enclosuring fractions 
        if sigma_error_mode:
            
            #Plot 1 sigma line
            bin_68, bin_edges, _ = stats.binned_statistic(Y[~nan_mask], X[~nan_mask], statistic=lambda x: np.quantile(x, 0.68), bins=STATS_BIN, range=ylim)
            ax.plot(bin_68, (bin_edges[:-1] + bin_edges[1:])/2, lw=LINE_WIDTH, c=STD_C)

            #Plot 2 sigma line
            bin_95, bin_edges, _ = stats.binned_statistic(Y[~nan_mask], X[~nan_mask], statistic=lambda x: np.quantile(x, 0.95), bins=STATS_BIN, range=ylim)
            ax.plot(bin_95, (bin_edges[:-1] + bin_edges[1:])/2, lw=LINE_WIDTH, c=STD_C)

            #Plot best case sigma lines (assuming gaussian distributed error which is not allways true)
            ax.plot([xlim[0], xlim[1]],[xlim[0], xlim[1]], lw=LINE_WIDTH//2, c=STD_C, linestyle='dashed')
            ax.plot([xlim[0], xlim[1]*2],[xlim[0], xlim[1]], lw=LINE_WIDTH//2, c=STD_C, linestyle='dashed')
            
        elif error_mode:
            
            #Area containting 50%, 1 sigma, 2 sigma
            bin_95, bin_edges, binnumber = stats.binned_statistic(X[~nan_mask], Y[~nan_mask], statistic=lambda x: np.quantile(x, 0.978), bins=STATS_BIN, range=xlim)
            ax.plot((bin_edges[:-1] + bin_edges[1:])/2, bin_95, lw=LINE_WIDTH//2, c=STD_C, linestyle='dashdot')
            bin_68, bin_edges, binnumber = stats.binned_statistic(X[~nan_mask], Y[~nan_mask], statistic=lambda x: np.quantile(x, 0.842), bins=STATS_BIN, range=xlim)
            ax.plot((bin_edges[:-1] + bin_edges[1:])/2, bin_68, lw=LINE_WIDTH//2, c=STD_C, linestyle='dashed')
            bin_50, bin_edges, binnumber = stats.binned_statistic(X[~nan_mask], Y[~nan_mask], statistic=lambda x: np.quantile(x, 0.50), bins=STATS_BIN, range=xlim)
            ax.plot((bin_edges[:-1] + bin_edges[1:])/2, bin_50, lw=LINE_WIDTH//2, c=STD_C, linestyle='solid')
            bin_32, bin_edges, binnumber = stats.binned_statistic(X[~nan_mask], Y[~nan_mask], statistic=lambda x: np.quantile(x, 0.158), bins=STATS_BIN, range=xlim)
            ax.plot((bin_edges[:-1] + bin_edges[1:])/2, bin_32, lw=LINE_WIDTH//2, c=STD_C, linestyle='dashed')
            bin_5, bin_edges, binnumber = stats.binned_statistic(X[~nan_mask], Y[~nan_mask], statistic=lambda x: np.quantile(x, 0.022), bins=STATS_BIN, range=xlim)
            ax.plot((bin_edges[:-1] + bin_edges[1:])/2, bin_5, lw=LINE_WIDTH//2, c=STD_C, linestyle='dashdot')
            
            #And a line for zero error
            ax.plot(xlim, [0, 0], lw=LINE_WIDTH//2, c="r", linestyle='dotted')
        
        else:

            #Median + Ideal line
            bin_median, bin_edges, binnumber = stats.binned_statistic(X[~nan_mask], Y[~nan_mask], statistic='median', bins=STATS_BIN, range=xlim)
            ax.plot((bin_edges[:-1] + bin_edges[1:])/2, bin_median, lw=LINE_WIDTH, c=MEAN_C)
            
            #2 Lines enclosing area containing 80% of the points
            bin_90, bin_edges, binnumber = stats.binned_statistic(X[~nan_mask], Y[~nan_mask], statistic=lambda x: np.quantile(x, 0.90), bins=STATS_BIN, range=xlim)
            ax.plot((bin_edges[:-1] + bin_edges[1:])/2, bin_90, lw=LINE_WIDTH, c=STD_C, linestyle='dashed')
            bin_10, bin_edges, binnumber = stats.binned_statistic(X[~nan_mask], Y[~nan_mask], statistic=lambda x: np.quantile(x, 0.10), bins=STATS_BIN, range=xlim)
            ax.plot((bin_edges[:-1] + bin_edges[1:])/2, bin_10, lw=LINE_WIDTH, c=STD_C, linestyle='dashed')
            
        
        #Plot a diagonal line
        if DIAGONAL:
            if NAN_FLAG:
                ax.plot([xlim[0], lower_x_edge],[xlim[0], lower_x_edge], lw=LINE_WIDTH//2, c=MEAN_C, linestyle='dashed')
            else:
                ax.plot([xlim[0], xlim[1]],[xlim[0], xlim[1]], lw=LINE_WIDTH//2, c=MEAN_C, linestyle='dashed')

        
        #Set grid
        ax.grid()
        
        #Set limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        #Set x label
        if sigma_error_mode:
            ax.set_xlabel('|MAP - Ground Truth|')
        else:
            ax.set_xlabel(label_dict[out_label])
        
        #First ax gets also a y label
        if i == 0:
            if sample_mode:
                if error_mode:
                    ax.set_ylabel("Abs Posterior Estimate Error")
                else:
                    ax.set_ylabel("Posterior Estimate")
            else:
                if error_mode:
                    ax.set_ylabel("MAP - Ground Truth")
                elif sigma_error_mode or sigma_mode:
                    ax.set_ylabel("Posterior Standard Deviation")
                else:
                    ax.set_ylabel("MAP Estimate")    
                    
        #Add title
        if sigma_error_mode :
            ax.set_title(label_dict[out_label])
            
    #Add a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    if not sample_mode:
        fig.colorbar(h, label='# Galaxies per Bin', cax=cax)
    else:
        fig.colorbar(h, label='# Samples per Bin', cax=cax)
    
    plt.tight_layout()
    plt.show()
    return fig


    
#Choose the peak which is closest to the gt
#To handle NaNs, replace them with a very large number
def get_best_peak(gt, peak_pos):
    
    def best_peak(x):
        temp_0 = np.nan_to_num(x[0], nan=999999999999.)
        temp_1 = np.nan_to_num(x[1], nan=999999999999.)
        i = np.argmin(np.abs(temp_0 - temp_1))
        return x[1][i]
    
    temp = np.stack([gt, peak_pos])
    return np.apply_along_axis(best_peak, 0, temp)
        
        
if __name__ == "__main__":  
    
    #Create Outputpath if not existing
    make_dir(output_path + "mass_redshift/")
    
    #Get ground truth, maps and other peaks 
    gt = data.gt
    maps = np.load(map_path)
    peak_number = np.load(peak_number_path)
    peak_pos = np.load(peak_position_path, allow_pickle=True)
    
    #Plot map
    fig = plot_map(gt, maps, data.x_header)
    fig.savefig(output_path + "map.pdf")
    
    #Plot map error
    fig = plot_map(gt, maps, data.x_header, error_mode=True)
    fig.savefig(output_path + "map_error.pdf")
    
    #Plot best guess
    best = get_best_peak(gt, peak_pos)
    fig = plot_map(gt, best, data.x_header)
    fig.savefig(output_path + "map_best_peak.pdf")
    
    #Plot total posterior
    posterior = np.load(posterior_path)
    posterior_subsample = np.concatenate(posterior[:,:NUM_SAMPLES,:])
    gt_ext = np.repeat(gt, NUM_SAMPLES, axis=0)
    fig = plot_map(gt_ext, posterior_subsample, data.x_header, sample_mode=True)
    fig.savefig(output_path + "map_total_posterior.pdf")
    
    #Sigma vs Abs Error
    sigma = np.nanstd(posterior, axis=1)
    fig = plot_map(gt, maps, data.x_header, sigma=sigma, sigma_error_mode=True)
    fig.savefig(output_path + "map_sigma_error.pdf")
    
    #Sigma vs gt
    sigma = np.nanstd(posterior, axis=1)
    fig = plot_map(gt, None, data.x_header, sigma=sigma, sigma_mode=True)
    fig.savefig(output_path + "map_sigma_gt.pdf")
    
    #Plot error for various bins
    z_bins = [0.0, 0.2, 0.5, 1.0]
    mass_bins = [10.0, 10.7, 12.0]
    
    z = data.df_m_test["z"]
    mass = data.df_m_test["mass"]
    
    z_ids = np.digitize(z, z_bins)
    mass_ids = np.digitize(mass, mass_bins)   
    
    for i in range(3):
        for j in range(2):
            #mask = np.logical_and(z_ids == i+1, mass_ids == j+1)
            mask = mass_ids == j+1
            masked_gt = gt[mask]
            masked_maps = maps[mask]
            fig = plot_map(masked_gt, masked_maps, data.x_header)
            fig.savefig(output_path + "mass_redshift/map_" + str(i) + str(j) + ".svg")
    
    #Save MAE as Metric
    metrics={"MAP_MAE": str(np.mean(np.abs(gt-maps), axis=0))}
    
    import json
    with open(c.metrics_path + "plot_map.json", "w") as f:
        json.dump(metrics, f)
    
    
