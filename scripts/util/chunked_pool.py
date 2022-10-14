from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

class ChunkedPool(object):
    '''
    
    Helper class to work multithreaded on unordered data including a callback (=checkpoints)
    
    '''
    
    def __init__(self, f, callback, size_chunks, num_workers):
        
        self._f = f
        self._callback = callback
        self._size_chunks = size_chunks
        self._num_workers = num_workers

    def __call__(self, x):
        
        num_chunks = int(len(x)/self._size_chunks)
        x_chunked = np.array_split(np.array(x), num_chunks)
        
        with Pool(self._num_workers) as p:
            
            for x_c in tqdm(x_chunked, total=num_chunks):
                
                output = []
                
                for i in p.imap_unordered(self._f, x_c):
                    output.append(i)
                    
                self._callback(output)