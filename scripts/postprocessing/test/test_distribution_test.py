import unittest
import matplotlib.pyplot as plt
import numpy as np

from scripts.postprocessing.distribution_test import DistributionTest, MeanNeighborDistanceDeviation

class TestMeanNeighborDistanceDeviation(unittest.TestCase):
    
    def setUp(self):
        NUMPOINTS = 5000
        NUMDIM = 128
        
        self.x = np.random.rand(NUMPOINTS, NUMDIM)
        
        self.x_wrong_num_points = np.random.rand(NUMPOINTS*2, NUMDIM)
        self.x_wrong_dim = np.random.rand(NUMPOINTS, NUMDIM+1)
        
        self.a = np.random.rand(NUMPOINTS, NUMDIM)
        self.b = self.a.copy() + 0.1 
        self.c = self.a.copy() 
        self.c[:,0] += 2.0
        self.d = np.concatenate((np.random.normal(size = (NUMPOINTS//2, NUMDIM)), np.random.rand(NUMPOINTS//2, NUMDIM)), axis=0)
        
    #def test_reshape_input(self):
    #Old implementation TODO: Fix Test
    #    x, y = DistributionTest.reshape_input(self.x, self.x_wrong_num_points)
    #    self.assertTrue(x.shape == y.shape)
    
    def test_mndd(self):
        mndd = MeanNeighborDistanceDeviation(self.x)
        self.assertTrue(isinstance(mndd(self.a), float))
        self.assertTrue(mndd(self.b) > mndd(self.a))
    
    def test_mndd_assertion(self):
        mndd = MeanNeighborDistanceDeviation(self.x)
        try:
            mndd(self.x_wrong_num_points)
            self.fail("Didn't raise AssertionError")
        except AssertionError:
            pass
        except:
            self.fail("Didn't raise AssertionError")
            
        try:
            mndd(self.x_wrong_dim)
            self.fail("Didn't raise AssertionError")
        except AssertionError:
            pass
        except:
            self.fail("Didn't raise AssertionError")
        
    def test_mndd_get_deviations(self):
        mndd = MeanNeighborDistanceDeviation(self.x)
        deviations_a = mndd.get_deviations(self.a)
        deviations_b = mndd.get_deviations(self.b)
        deviations_c = mndd.get_deviations(self.c)
        deviations_d = mndd.get_deviations(self.d)
        self.assertTrue(deviations_a.ndim == 1)
        self.assertTrue(deviations_a.shape[0] == self.x.shape[0])
        
        plt.hist(deviations_a, bins=30)
        plt.hist(deviations_b, bins=30, alpha = 0.5)
        plt.hist(deviations_c, bins=30, alpha = 0.5)
        plt.hist(deviations_d, bins=30, alpha = 0.5)
        plt.legend(labels=['self','b','c','d'])
        plt.savefig('./temp/TestDistributionTest_test_mndd.png')
        
if __name__ == '__main__':
    unittest.main()