import numpy as np
import matplotlib.pyplot as plt
import glob

import umap

import pandas as pd
from sklearn.utils import shuffle
import config as c

#Set seed
SEED = 0

#Plot a single UMAP of x with color c
def create_umap(x, n_neighbors=50, min_dist=0.1, n_components=2, metric='euclidean'):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        densmap=False,
        random_state=SEED
    )
    u = fit.fit_transform(x);
    
    return np.array(u)

    
with open(c.postprocessing_path + 'representation.npy', 'rb') as f:
    rep = np.load(f)
        
u = create_umap(rep)

with open(c.postprocessing_path + 'umap.npy', 'wb') as f:
    np.save(f, u)