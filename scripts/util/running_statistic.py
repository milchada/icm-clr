"""
Created on Wed Oct 18 2023

Calculate running statistic using a fixed window

@author: Lukas Eisert
"""

import numpy as np


def running_statistic(x: np.ndarray, y: np.ndarray, statistic, bins: int, window_size: float = None):
    """
    Calculate running statistic using a fixed window in linear spaced bins

    :param x: array with x coordinates which is binned
    :param y: array with y coordinates the statistic is calculated from 
    :param statistic: function taking a list as argument and returning a float
    :param bins: number of bins used to calculate the running statistic in
    :param windows_size: window sized to calculate the statistic in. Defaults to 5 times the bin size
    :return: running statistics, bin edges, not used(for compatibility with scipy)
    """ 
    assert len(x) == len(y)
    
    bin_edges = np.linspace(np.min(x), np.max(x), bins+1, endpoint=True)
        
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    
    if window_size is None:
        window_size = 5*bin_width
    
    bin_statistic = []
    
    for i, center in enumerate(bin_centers):
        upper = center + window_size/2
        lower = center - window_size/2
        
        mask = np.logical_and(x >= lower, x <= upper)
        
        if sum(mask) == 0:
            bin_statistic.append(np.nan)
        else:
            bin_statistic.append(statistic(y[mask]))
    
    return np.array(bin_statistic), bin_edges, None