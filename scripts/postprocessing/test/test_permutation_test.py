import unittest
import matplotlib.pyplot as plt
import numpy as np

from scripts.postprocessing.distribution_test import PermutationTest, MeanNeighborDistanceDeviation

class TestPermutation(unittest.TestCase):
    
    def setUp(self):
        NUMPOINTS = 5000
        NUMDIM = 128
        
        self.x = np.random.rand(NUMPOINTS, NUMDIM)
        
        self.a = np.random.rand(NUMPOINTS, NUMDIM)
        self.b = self.a + 0.1 
        self.c = self.a + 0.1*np.random.rand(NUMPOINTS, NUMDIM)
        self.c[:,0] += 0.5
        
        self.d = np.random.normal(size = (NUMPOINTS, NUMDIM))
        

    def test_permutation_mndd(self):
        mndd = MeanNeighborDistanceDeviation
        
        
        num_swaps_per_step=500
        num_steps = 50
        
        test = PermutationTest(self.x, self.a, mndd)
        test(num_swaps_per_step, num_steps)
        plt.plot(test.T_list)
        
        test = PermutationTest(self.x, self.b, mndd)
        test(num_swaps_per_step, num_steps)
        plt.plot(test.T_list)
        
        test = PermutationTest(self.x, self.c, mndd)
        test(num_swaps_per_step, num_steps)
        plt.plot(test.T_list)
        
        test = PermutationTest(self.x, self.d, mndd)
        test(num_swaps_per_step, num_steps)
        plt.plot(test.T_list)
        
        plt.savefig('./temp/TestPermutationTest.png')
        
if __name__ == '__main__':
    unittest.main()