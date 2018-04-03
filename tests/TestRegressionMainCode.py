import unittest
import pandas as pd
import numpy as np
from app.mlprogram.algorithm.RegressionMainCode import MultiVariateRegression
from unittest.mock import patch

class TestRegressionMainCode(unittest.TestCase):
    def setUp(self):
        dataset = [{'size' : 10,
                    'price' : 20},
                   {'size': 20,
                    'price' : 40},
                   {'size' : 30,
                    'price' : 60},
                   {'size' : 40,
                    'price' : 80}]
        self.df = pd.DataFrame(dataset)
        self.header = ['size']
        self.label = 'price'
        self.test_list = [20, 40, 60, 80]
        self.model = [np.array([0, 2]), ['size']]

    def tearDown(self):
        pass

    def test_training(self):
        df = self.df
        regres = MultiVariateRegression(numSteps = 100000, learningRate=1e-6)
        result = regres.training(df=self.df, label=self.label, type='regression', dummies=False)
        temp =abs(result[0] - self.model[0])
        error= np.mean(temp)
        self.assertEqual(round(error, 1), 0.0)

    def test_predict(self):
        regres= MultiVariateRegression()
        result = regres.predict(df=self.df, model=self.model, dummies = False)
        print(result)
        self.assertEqual(result, self.test_list)
    
    def test_testing(self):
        features = np.array([[10],[20], [30], [40]])
        target = np.array([20, 40, 60, 80])
        regress = MultiVariateRegression()
        result = regress.testing(features, target, model=self.model)
        self.assertEqual(result, 0)

if __name__ == "__main__":
    unittest.main()
