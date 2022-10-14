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
        

    def test_permutation(self):
        mndd = MeanNeighborDistanceDeviation(self.x)
        
if __name__ == '__main__':
    unittest.main()