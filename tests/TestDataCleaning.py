import unittest
import pandas as pd
import numpy as np
from app.mlprogram.preprocessing.DataCleaning import DataCleaning2

class DataCleaningTest(unittest.TestCase):
    def setUp(self):
        dataset = np.array([[np.NaN, 2, 3, 4], [0, 3, np.NaN, 2],
                                 [1,np.NaN,3,6], [2,4,3,np.NaN]])
        self.df = pd.DataFrame(dataset)

        dataset2 = np.array([[np.NaN, 2, 3, 2], [1, 3, np.NaN, 2],
                                 [1,np.NaN,3,6], [2,3,3,np.NaN]])
        self.df2 = pd.DataFrame(dataset2)

        dataset3 = np.array([[1.0, 2.0, 3.0, 2.0], [1.0, 3.0, 3.0, 2.0],
                                 [1.0,3.0,3.0,6.0], [2.0,3.0,3.0,2.0]])
        self.df3 = pd.DataFrame(dataset3)
        
        self.target_data = np.array([[1], [2], [3], [4]])
        
        self.mode_values  =  { 0 : 1,
                               1 : 3,
                               2 : 3,
                               3 : 2 }

        self.mean_values  =  { 0 : 1,
                               1 : 3,
                               2 : 3,
                               3 : 4 }

        self.median_values = { 0 : 1,
                               1 : 3,
                               2 : 3,
                               3 : 4 }
    def tearDown(self):
        pass

    def test_fit_case1(self):
        #method mode
        self.df2['target'] = self.target_data
        dc = DataCleaning2(method='mode')
        result = dc.fit(self.df2, label='target')
        self.assertDictEqual(result, self.mode_values)

    def test_fit_case2(self):
        #method median
        self.df['target'] = self.target_data
        dc = DataCleaning2(method='median')
        result = dc.fit(self.df, label ='target')
        self.assertDictEqual(result, self.median_values)

    def test_fit_case3(self):
        #method mean
        self.df['target'] = self.target_data
        dc = DataCleaning2(method='mean')
        result=dc.fit(self.df, label='target')
        self.assertDictEqual(result, self.mean_values)

    def test_mean(self):
        dc = DataCleaning2() 
        result = dc.mean(self.df)
        self.assertDictEqual(result, self.mean_values)

    def test_median(self):
        dc = DataCleaning2() 
        result = dc.median(self.df)
        self.assertDictEqual(result, self.median_values)

    def test_mode(self):
        dc = DataCleaning2() 
        result = dc.mode(self.df2)
        self.assertDictEqual(result, self.mode_values)

    def test_transform_case1(self):
        #values is not none
        dc = DataCleaning2()
        newdf = dc.transform(self.df2, values=self.mode_values)
        result = pd.DataFrame.equals(newdf, self.df3)
        self.assertTrue(result)

    def test_transform_case2(self):
        #values is None
        self.df2['target'] = self.target_data
        self.df3['target'] = self.target_data
        dc = DataCleaning2()
        newdf = dc.transform(self.df2, label='target')
        result = pd.DataFrame.equals(newdf, self.df3)
        self.assertTrue(result)

if __name__=='__main__':
    unittest.main()
