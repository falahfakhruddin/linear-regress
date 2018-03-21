import unittest
from app.mlprogram import translator as trans
from app.mlprogram.algorithm.RegressionMainCode import MultiVariateRegression
from app.mlprogram.algorithm.NaiveBayess import NaiveBayess
from app.mlprogram.algorithm.MLPClassifier import SklearnNeuralNet
from app.mlprogram.algorithm.LogisticRegression import LogisticRegression
from app.mlprogram.preprocessing.FeatureSelection import FeatureSelection
from app.mlprogram.preprocessing.DataCleaning import DataCleaning2

class TranslatorTest(unittest.TestCase):
    def setUp(self):
        self.string1 = 'regression'
        self.string2 = 'logistic regression'
        self.string3 = 'naive bayess'
        self.string4 = 'neural network'

    def tearDown(self):
        pass

    def test_case1(self):
        algorithm_test = MultiVariateRegression()
        algorithm  = trans.algorithm_trans(self.string1)
        result = algorithm.__class__.__name__ == algorithm_test.__class__.__name__
        self.assertTrue(result)

    def test_case2(self):
        algorithm_test = LogisticRegression()
        algorithm  = trans.algorithm_trans(self.string2)
        result = algorithm.__class__.__name__ == algorithm_test.__class__.__name__
        self.assertTrue(result)
 
    def test_case3(self):
        algorithm_test = NaiveBayess()
        algorithm  = trans.algorithm_trans(self.string3)
        result = algorithm.__class__.__name__ == algorithm_test.__class__.__name__
        self.assertTrue(result)
    
    def test_case4(self):
        algorithm_test = SklearnNeuralNet()
        algorithm  = trans.algorithm_trans(self.string4)
        result = algorithm.__class__.__name__ == algorithm_test.__class__.__name__
        self.assertTrue(result)
    
    def test_case5(self):
        preprocessing_test = [FeatureSelection(), DataCleaning2()]
        list = ['feature selection', 'data cleaning']
        preprocessing = trans.preprocessing_trans(list)
        result = None
        for i in range(len(preprocessing)):    
            result = preprocessing_test[i].__class__.__name__ == preprocessing[i].__class__.__name__
            if result == False:
                break
        self.assertTrue(result)

if __name__=='__main__':
    unittest.main()
