import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from app.mlprogram.algorithm.NaiveBayess import NaiveBayess

class TestNaiveBayees(unittest.TestCase):
    def setUp(self):
        dataset =[{ 'temp' : 'hot',
                    'windy' : 'strong',
                    'humidity' : 'normal',
                    'outlook' : 'sunny',
                    'play' : 'yes'},
                  { 'temp' : 'cool',
                    'windy' : 'strong',
                    'humidity' : 'high',
                    'outlook' : 'rainy',
                    'play' : 'no'},
                  { 'temp' : 'hot', 
                    'windy' : 'weak',
                    'humidity' : 'normal',
                    'outlook' : 'overcast',
                    'play' : 'yes'},
                  { 'temp' : 'cool',
                    'windy' : 'strong',
                    'humidity' : 'high',
                    'outlook' : 'rainy',
                    'play' : 'no'}]
        self.df = pd.DataFrame(dataset)
        self.header = ['windy', 'temp', 'humidity', 'outlook', 'play']
        self.label = 'play'
        self.test_list = ['yes', 'no', 'yes', 'no']
        self.model = [{"no" : {"windy" : {"strong" : 0.571428571428571, "weak" : 0.428571428571429}, "outlook" : {"rainy" : 0.375,"overcast" : 0.125,"sunny" : 0.5},"temp" : {"mild" : 0.375,"cool" : 0.25,"hot" : 0.375},"humidity" : {"high" : 0.857142857142857,"normal" : 0.142857142857143}},"yes" : {"windy" : {"strong" : 0.4,"weak" : 0.6},"outlook": {"rainy" : 0.363636363636364,"overcast" : 0.272727272727273,"sunny" : 0.363636363636364},"temp" : {"mild" : 0.454545454545455,"cool" : 0.363636363636364,"hot" : 0.181818181818182},"humidity" : {"high" : 0.5,"normal" : 0.5}}},{"no" : 5,"yes" : 8},["humidity","outlook","temp","windy"]]   
    
    def tearDown(self):
        pass

    def test_training(self):
        df =self.df
        nb = NaiveBayess()
        trained = nb.training(df=df, label=self.label, type='classification', dummies=False)
        self.weightDict = trained[0]
        self.labelCounts = trained[1]
        self.header = trained[2]

        target = list(set(list(df)).difference(self.header))
        if len(target) != 0:
            del df[target[0]]
        features = df.values

        prediction = list()
        for vector in features:
            list_vector = vector.tolist()
            probabilityPerLabel = {}
            for label in self.labelCounts:
                tempProb = 1
                for featureValue in vector:
                    tempProb *= self.weightDict[label][self.header[list_vector.index(featureValue)]][featureValue]
                tempProb *= self.labelCounts[label]
                probabilityPerLabel[label] = tempProb
            print(probabilityPerLabel)
            prediction.append(max(probabilityPerLabel, key=lambda classLabel: probabilityPerLabel[classLabel]))
        print(prediction)
        self.assertEqual(prediction, self.test_list)

    def test_predict(self):
        nb = NaiveBayess()
        result = nb.predict(df=self.df, model=self.model, dummies=False)
        self.assertEqual(result, self.test_list)
    
    @patch('app.mlprogram.algorithm.NaiveBayess.NaiveBayess.predict')
    def test_testing(self, value):
        features = np.array([['hot', 'strong', 'normal', 'sunny'], ['cool', 'strong', 'high', 'rainy'], 
                             ['hot', 'weak', 'normal', 'outcast'], ['cool', 'strong', 'high', 'rainy']])
        target = np.array([['yes', 'no', 'yes', 'no']])
        nb = NaiveBayess()
        result = nb.testing(features, target, model=self.model)
        self.assertEqual(result, 0)

if __name__ == "__main__":
    unittest.main()
