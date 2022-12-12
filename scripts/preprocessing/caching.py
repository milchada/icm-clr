import pandas as pd
import os

class Cache(object):
    def __init__(self, path, use_cache=True):
        
        self._path = path
        self._use_cache = use_cache
    
        self.load()
    
    def load(self):
        if os.path.exists(self._path) and self._use_cache:
            self._cache = pd.read_csv(self._path)
        else:
            self._cache = None
        
    def save(self):
        if self.use_cache:
            self._cache.to_csv(self._path, index=False)
    
    def isin(self, x, column):
        if self.use_cache:
            return x in self._cache[column].values
        else:
            return False

    def push(self, df):
        if self.use_cache:
            self._cache = pd.concat([self._cache, df])
            self._cache = self._cache.drop_duplicates()
        else:
            self._cache = df
            
        self.save()
        
    def pull(self):
        return self._cache
    
    @property
    def use_cache(self):
        return self._use_cache and self._cache is not None
    
    @property
    def path(self):
        return self._path
    
    
#Short test
if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm
    cache = Cache("../../temp/cache.csv")
    for i in tqdm(range(100)):
        a = pd.DataFrame(np.random.rand(10000, 3), columns=['a','b','c'])
        cache.push(a)
        
        for i in tqdm(range(10000)):
            cache.isin(2, 'a')