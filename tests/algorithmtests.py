import unittest
import numpy as np
from ..algorithm.LogisticRegression import LogisticRegression
from ..algorithm.MLPClassifier import MLPClassifier
from ..algorithm.NaiveBayess import NaiveBayess
from ..algorithm.RegressionMainCode import MultiVariateRegression

class AlgorithmTests(unittest.TestCase):
    def test_multiplication(self):
        temp = 4
        result = 2*2
        self.assertEqual(result, temp)
    
    def test_divided(self):
        temp = 2
        result = 5/2
        self.assertEqual(result, temp)
    
    def test_sum(self):
        temp = 3
        result = 2+1
        self.assertEqual(result, temp)

    def sigmoid_logistic_regression(self):
        
        temp = 1/(1+np.exp(-scores))
        logreg = LogisticRegression()
        result = logreg.sigmoid
if __name__ == "__main__":
    unittest.main()
