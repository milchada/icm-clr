from scipy.spatial import cKDTree
import numpy as np

class DistributionTest(object):
    def __init__(self, x):
        self._x = x
        
    def reshape_input(x, y):
        '''Function to reshape the inputs by selecting random rows from the larger of the two input arrays'''
        rng = np.random.default_rng()
        
        assert x.ndim == 2
        assert y.ndim == 2
        
        assert x.shape[1] == y.shape[1], "Second dimensions of arrays should be equal! Got " + str(x.shape[1]) + " and " + str(y.shape[1])
        
        num_x = x.shape[0]
        num_y = y.shape[0]
        
        if num_x == num_y:
            return x, y
        elif num_x > num_y:
            rand_index = rng.choice(np.arange(num_x), size=num_y, replace=False)
            return x[rand_index], y
        elif num_x < num_y:
            rand_index = rng.choice(np.arange(num_y), size=num_x, replace=False)
            return x, y[rand_index]
        else:
            raise
            
    def _check_input(self, x, y):
        assert x.ndim == 2
        assert y.ndim == 2
        
        assert x.shape == y.shape, "Shapes of arrays should be equal! Got " + str(x.shape) + " and " + str(y.shape)
        
    def __call__(self, y):
        raise NotImplementedError("This function is supposed to be overwritten!")
        
class MeanNeighborDistanceDeviation(DistributionTest):
    def __init__(self, x, n_neighbor=64, p=1):
        super().__init__(x)
        
        self._n_neighbor = n_neighbor
        self._p = p
        
    def _build_tree(self, data):
        return cKDTree(data, compact_nodes=False, balanced_tree=False)
        
    def _get_distances(self, y, k):
        '''Get the distance for each y to the k_th neigbor in x'''
        assert k >= 1
        
        tree = self._build_tree(y)
        distance, _ = tree.query(self._x, k=[k], p=self._p, workers=-1)
        
        return distance.reshape((self._x.shape[0],))
    
    def get_distances(self, y):
        return self._get_distances(y, self._n_neighbor)
        
    def get_self_distances(self):
        return self._get_distances(self._x, self._n_neighbor+1)
    
    def get_normalization_distances(self, y):
        concat_array = np.concatenate([self._x , y], axis=0)
        return self._get_distances(concat_array, 2*self._n_neighbor+1)
        
    def get_deviations(self, y):
        self._check_input(self._x, y)
        return (self.get_distances(y) - self.get_self_distances())/self.get_normalization_distances(y)
        
    def __call__(self, y):
        '''Get the mean neighbor distance deviation between y and x'''
        return np.mean(np.abs(self.get_deviations(y)))
    
    
class PermutationTest(object):
    
    def __init__(self, x, y, distribution_test):
        assert x.shape == y.shape
        #assert isinstance(distribution_test, DistributionTest)
        
        self._x = x
        self._y = y 
        self._test = distribution_test
        
        self._T_list = [self.get_test_statistic()]
        
        self._rng = np.random.default_rng()
    
    def perform_permutation_step(self, num_swaps):
        '''
        Permutate one datapoint between x and y
        
        Shuffle both sets and then swap the first datapoint in each set with each other
        
        '''
        
        self._rng.shuffle(self._x, axis=0)
        self._rng.shuffle(self._y, axis=0)
        
        for i in range(num_swaps):
            tmp = self._x[i]
            self._x[i] = self._y[i]
            self._y[i] = tmp
        
    def get_test_statistic(self):
        test = self._test(self._x)
        return test(self._y)
    
    def perform_step(self, num_swaps_per_step):
        self.perform_permutation_step(num_swaps_per_step)
        T = self.get_test_statistic()
        self._T_list.append(T)
        
    @property
    def T_list(self):
        return self._T_list
        
    @property
    def T(self):
        return self.T_list[0]
    
    @property
    def T_last(self):
        return self.T_list[-1]
        
    @property
    def p_value(self):
        '''That should only work if the number of swaps is high enough; but not sure if it will be meaningfull anyways'''
        T_array = np.array(self._T_list)
        return np.sum(T_array > self.T)/self.num_permutations
        
    @property
    def num_permutations(self):
        return len(self.T_list)
    
    def __call__(self, num_swaps_per_step, num_steps):
        
        for i in range(num_steps):
            self.perform_step(num_swaps_per_step)
            print("Step " + str(i+1) + ": T = " + str(self.T_last))

            