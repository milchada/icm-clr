import numpy as np
import umap

#Set seed
SEED = 0

def fit_umap(x, n_neighbors: int = 50, min_dist: float = 0.1, n_components: int = 2, metric: str = 'euclidean'):
    
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        densmap=False,
        random_state=SEED
    )
    model = fit.fit(x)
    
    return model

def fit_transform_umap(x, n_neighbors: int = 50, min_dist: float = 0.1, n_components: int = 2, metric: str = 'euclidean'):
    model = fit_umap(x, n_neighbors, min_dist, n_components, metric)
    u = model.transform(x)
    return np.array(u)