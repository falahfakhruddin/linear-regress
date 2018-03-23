import unittest
import pandas as pd
import numpy as np
from app.mlprogram.preprocessing.FeatureSelection import FeatureSelection

class TestFeatureSelection(unittest.TestCase):
    def setUp(self):
        self.dataset = np.array([[0, 0, 0, 2], [1, 0, 3, 1]])
        
    def tearDown(self):
        pass

    def test_transform(self):
        X = np.array([[np.NaN, np.NaN, np.NaN, 4], [0, np.NaN, np.NaN, 2],[1,np.NaN,3,1], [np.NaN,np.NaN,np.NaN,np.NaN]])
        df = pd.DataFrame(X)
        fs = FeatureSelection()
        newdf = fs.transform(df)
        newarr = newdf.values
        np.place(newarr, np.isnan(newarr), 0)
        result = np.array_equal(newarr, self.dataset)
        self.assertTrue(result)

 
