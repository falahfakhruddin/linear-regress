import unittest
import pandas as pd
import numpy as np
from app.mlprogram.preprocessing.Normalization import Normalization

class TestNormalization(unittest.TestCase):
    def setUp(self):
        dataset = np.array([[1, 2, 3, 1.0], [2, 4, 2, 2.0], [3, 10, 1, 3.0]])
        self.header = ['a', 'b', 'c']
        self.label = ['d']
        self.header = self.header + self.label
        self.df = pd.DataFrame(dataset, columns=self.header)
        self.parameter = {'a' : {'max_value' : 3,
                                 'min_value' : 1},
                          'b' : {'max_value' : 10,
                                 'min_value' : 2},
                          'c' : {'max_value' : 3,
                                 'min_value' : 1}}

    def tearDown(self):
        pass

    def test_fit(self):
        norm =Normalization()
        result = norm.fit(self.df, self.label[0])
        self.assertDictEqual(result, self.parameter)

    def test_transform_case1(self):
        #parameter is None
        dataset = np.array([[0, 0, 1, 1], [0.5, 0.25, 0.5, 2], [1, 1, 0, 3]])
        test_df = pd.DataFrame(dataset, columns=self.header)
        norm = Normalization()
        newdf = norm.transform(self.df, label = self.label[0])
        result = pd.DataFrame.equals(newdf, test_df)
        self.assertTrue(result)
        

    def test_transform_case2(self):
        #parameter is not None
        dataset = np.array([[0, 0, 1, 1], [0.5, 0.25, 0.5, 2], [1, 1, 0, 3]])
        test_df = pd.DataFrame(dataset, columns=self.header)
        norm = Normalization()
        newdf = norm.transform(self.df, label=self.label[0], values=self.parameter)
        result = pd.DataFrame.equals(newdf, test_df)
        self.assertTrue(result)


