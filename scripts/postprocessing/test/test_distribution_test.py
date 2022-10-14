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
        self.b = self.a + 0.1 
        self.c = self.a + 0.1*np.random.rand(NUMPOINTS, NUMDIM)
        self.c[:,0] += 0.5
        
    def test_reshape_input(self):
        x, y = DistributionTest.reshape_input(self.x, self.x_wrong_num_points)
        self.assertTrue(x.shape == y.shape)
    
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
        self.assertTrue(deviations_a.ndim == 1)
        self.assertTrue(deviations_a.shape[0] == self.x.shape[0])
        
        plt.hist(deviations_a, bins=30)
        plt.hist(deviations_b, bins=30, alpha = 0.5)
        plt.hist(deviations_c, bins=30, alpha = 0.5)
        plt.savefig('./temp/TestDistributionTest_test_mndd.png')
        
        first_axis = self.x[:,0]
        plt.hist2d(deviations_c, first_axis, bins=30)
        plt.savefig('./temp/TestDistributionTest_test_mndd_2.png')
        
if __name__ == '__main__':
    unittest.main()