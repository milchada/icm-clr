import unittest

import string
import numpy as np
import pandas as pd

from scripts.preprocessing.prepare import DatasetPreparator, LABELS


class TestPrepare(unittest.TestCase):
    '''Consistency test for the cropper classes'''
    
    def setUp(self):
        
        self.num_2_fields = lambda x: list(string.ascii_lowercase)[:x]
        
        NUM_DATA = 100
        NUM_FIELDS = len(LABELS)+2
        
        data = np.random.rand(NUM_DATA, NUM_FIELDS)
        self.df = pd.DataFrame(data, columns=self.num_2_fields(NUM_FIELDS))
        
    def test_scale(self):
        df_scaled, scaler = DatasetPreparator.scale(self.df, self.num_2_fields(len(LABELS)), None)
        self.assertEqual(len(df_scaled), len(self.df), 'incorrect target df length')
        self.assertTrue(list(df_scaled.head(0)) == LABELS, 'incorrect target labels')
        
    def test_scale_None(self):
        df_scaled, scaler = DatasetPreparator.scale(self.df, None, None)
        self.assertEqual(len(df_scaled), len(self.df), 'incorrect target df length')
        self.assertTrue(list(df_scaled.head(0)) == ["None"], 'incorrect target labels')
        
    def test_scale_NoneField(self):
        df_scaled, scaler = DatasetPreparator.scale(self.df, self.num_2_fields(len(LABELS)-1) + [None], None)
        self.assertEqual(len(df_scaled), len(self.df), 'incorrect target df length')
        self.assertTrue(list(df_scaled.head(0)) == LABELS, 'incorrect target labels')
        #self.assertTrue(np.all(df_scaled[LABELS[-1]].to_numpy() == np.nan), 'data for None field should be np.nan')
        
if __name__ == '__main__':
    unittest.main()