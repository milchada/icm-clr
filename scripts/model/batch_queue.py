import torch
import config as c
from multiprocessing import Pool
from functools import partial

def init_batch_queue(model, dataloader, batch_queue_size):
    img_batch_list = []
    rep_batch_list = []
    
    for batch in dataloader:
        
        img = torch.cat(batch, dim=0)[0::2]
        
        with torch.no_grad():
            rep = model(img.to(c.device))
        
        img_batch_list.append(img.to('cpu'))
        rep_batch_list.append(rep.to('cpu'))
        
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
        
        NUM_CORES = 16
        self._pool = Pool(NUM_CORES)
        
    def __del__(self):
        self._pool.close()
    
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
        
        self._img_batch_list.append(img.to('cpu'))
        self._img_batch_list.pop(1)
        
        rep = rep.detach()
        self._rep_batch_list.append(rep.to('cpu'))
        self._rep_batch_list.pop(1)
        
        self._calc_sample() 
    
    def _nn_search(self, rep_sample, x):
        dist = torch.norm(rep_sample - x.to('cpu'), dim=1, p=1)
        knn = dist.topk(1, largest=False)
        return self._img_sample[knn.indices]
    
    def multi_nn_search(self, x):
        f = partial(self._nn_search, self._rep_sample)
        nn_search_results = self._pool.map(f, torch.unbind(x.to('cpu'), dim=0))    
        return torch.cat(nn_search_results, dim=0).to(c.device)
    
#Short test
if __name__ == "__main__":
    BATCH_SIZE = 32
    NUM_DIM = 128
    QUEUE_LENGTH = 64
    rep_batch_list = [torch.randn(BATCH_SIZE, NUM_DIM) for i in range(QUEUE_LENGTH)]
    img_batch_list = [torch.randn(BATCH_SIZE, NUM_DIM, NUM_DIM) for i in range(QUEUE_LENGTH)]
    
    bq = BatchQueue(img_batch_list, rep_batch_list)
    assert len(bq) == QUEUE_LENGTH
    
    test = torch.randn(1, NUM_DIM)
    out = bq.nn_search(test)
    print(out.size())
    print(out)
    
    test = torch.randn(4, NUM_DIM)
    out = bq.multi_nn_search(test)
    print(out.size())
    print(out)
    
    bq.push(torch.randn(BATCH_SIZE, NUM_DIM, NUM_DIM), torch.randn(BATCH_SIZE, NUM_DIM))
    assert len(bq) == QUEUE_LENGTH
