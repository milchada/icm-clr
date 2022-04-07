import numpy as np
from scipy.stats import gaussian_kde

def kde(x, N):
    """
    Perform a gaussian kde using the silverman rule  

    :param x: 1-Dim samples to perform the kde on
    :param N: Number of gridpoints to use (between the minimum and maximum contained in x)
    
    :return: gridpoint coordinates and densities
    """ 
    
    #Handle empty input
    if len(x) == 0:
        return [0.], [0.]
        
    density = gaussian_kde(x, 'silverman')
    xs = np.linspace(np.min(x), np.max(x), N)
    density._compute_covariance()
    dxs = density(xs)
    dxs /= np.max(dxs)
    
    return xs, dxs