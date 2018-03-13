import unittest
import numpy as np
from app.mlprogram.algorithm.LogisticRegression import LogisticRegression
from app.mlprogram.algorithm.MLPClassifier import MLPClassifier
from app.mlprogram.algorithm.NaiveBayess import NaiveBayess
from app.mlprogram.algorithm.RegressionMainCode import MultiVariateRegression

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.vector = np.arange(10)
        self.value = 1/(1+np.exp(-self.vector))

    def tearDown(self):
        pass

    def test_sigmoid(self):
        log = LogisticRegression()
        result = log.sigmoid(self.vector)
        self.assertEqual(result.tolist(), self.value.tolist())
    
    def test_sum(self):
        temp = 3
        result = 2+1
        self.assertEqual(result, temp)

if __name__ == "__main__":
    unittest.main()
