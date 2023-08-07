import torch
import config as c
from multiprocessing import Pool
from numba import jit
#from torch.multiprocessing import Pool
import torch.multiprocessing as mp
import numpy as np
from functools import partial

mp.set_sharing_strategy('file_system')
assert mp.get_sharing_strategy() == 'file_system'

def init_batch_queue(model, dataloader, batch_queue_size):
    img_batch_list = []
    rep_batch_list = []
    
    for batch in dataloader:
        
        img = torch.cat(batch, dim=0)[0::2]
        
        with torch.no_grad():
            rep = model(img.to(c.device))
        
        img_batch_list.append(img.to(c.device_nn_search))
        rep_batch_list.append(rep.to(c.device_nn_search))
        
        if len(img_batch_list) == batch_queue_size:
            break
            
    return BatchQueue(img_batch_list, rep_batch_list)

class BatchQueue(object):
    def __init__(self, img_batch_list, rep_batch_list):
        assert len(img_batch_list) == len(rep_batch_list)
        assert len(img_batch_list[0]) == len(rep_batch_list[0])
        self._img_batch_list = img_batch_list
        self._rep_batch_list = rep_batch_list
        self._calc_sample()
    
    def _calc_sample(self):
        self._img_sample = torch.cat(self._img_batch_list, dim=0)
        self._rep_sample = torch.cat(self._rep_batch_list, dim=0) 
    
    def __len__(self):
        return len(self._rep_batch_list)
    
    @property
    def batch_size(self):
        return len(self._rep_batch_list[0])
    
    def push(self, img, rep):
        assert len(img) == self.batch_size
        assert len(rep) == self.batch_size
        
        self._img_batch_list.append(img.to(c.device_nn_search))
        self._img_batch_list.pop(1)
        
        rep = rep.detach()
        rep = torch.nn.functional.normalize(rep, dim=1)
        self._rep_batch_list.append(rep.to(c.device_nn_search))
        self._rep_batch_list.pop(1)
        
        self._calc_sample() 
    
    def nn_search(self, x):
        dist = torch.norm(self._rep_sample - x, dim=1, p=1)
        knn = dist.topk(1, largest=False)
        return self._img_sample[knn.indices]
    
    def _multi_nn_search_torch(self, x):
        x = torch.nn.functional.normalize(x, dim=1)
        return torch.cat([self.nn_search(x_i) for x_i in torch.unbind(x, dim=0)], dim=0).to(c.device)
    
    @jit(nopython=True)
    def _multi_nn_search_core(x, rep_sample):
        diff = np.abs(rep_sample - x)
        dist = np.sum(diff, axis=1)
        return np.argmin(dist)
    
    def _multi_nn_search_numba(self, x):
        
        x = torch.unbind(x, dim=0)
        x = [i.numpy() for i in x]
        rep_sample = self._rep_sample.numpy()
        
        nn_search_results = [BatchQueue._multi_nn_search_core(i, rep_sample) for i in x]
        
        img_out = []
        for i in nn_search_results:
            img_out.append(self._img_sample[i])
        
        return torch.stack(img_out).to(c.device)
    
    def multi_nn_search(self, x, method='torch'):
        '''

        Get the images which are closest in rep space to x (batch of reps)

        Numba seems to slower than torch

        '''
    
        x = x.to(c.device_nn_search)
        
        if method == 'numba':
            return self._multi_nn_search_numba(x)
        elif method == 'torch':
            return self._multi_nn_search_torch(x)
    
#Short test
if __name__ == "__main__":
    import time
    BATCH_SIZE = 64
    NUM_DIM = 128
    IMG_DIM = 128
    QUEUE_LENGTH = 512
    rep_batch_list = [torch.randn(BATCH_SIZE, NUM_DIM) for i in range(QUEUE_LENGTH)]
    img_batch_list = [torch.randn(BATCH_SIZE, IMG_DIM, IMG_DIM) for i in range(QUEUE_LENGTH)]
    
    bq = BatchQueue(img_batch_list, rep_batch_list)
    assert len(bq) == QUEUE_LENGTH
    
    test = torch.randn(1, NUM_DIM)
    out = bq.nn_search(test)
    print(out.size())
    
    
    test = torch.randn(BATCH_SIZE, NUM_DIM)
    
    print('Using serial torch')
    st = time.time()
    out = bq.multi_nn_search(test, method='torch')
    print(time.time() - st)
    print(out.size())
    
    print('Using serial numba (inc. compilation)')
    st = time.time()
    outn = bq.multi_nn_search(test, method='numba')
    print(time.time() - st)
    print(outn.size())
    
    print('Using serial numba')
    st = time.time()
    outn2 = bq.multi_nn_search(test, method='numba')
    print(time.time() - st)
    
    assert torch.all(torch.eq(out, outn2))

    bq.push(torch.randn(BATCH_SIZE, IMG_DIM, IMG_DIM), torch.randn(BATCH_SIZE, NUM_DIM))
    assert len(bq) == QUEUE_LENGTH
