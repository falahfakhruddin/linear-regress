import unittest
import numpy as np
import pandas as pd
from app.mlprogram.algorithm.LogisticRegression import LogisticRegression

class LogisticRegressionTest(unittest.TestCase):
    def setUp(self):
        self.vector = np.arange(10)
        self.value = 1/(1+np.exp(-self.vector))
        dataset =[{ 'temp' : 'hot',
                    'windy' : 'strong',
                    'humidity' : 'normal',
                    'outlook' : 'sunny',
                    'play' : 'yes'},
                  { 'temp' : 'cool',
                    'windy' : 'strong',
                    'humidity' : 'high',
                    'outlook' : 'rain',
                    'play' : 'no'},
                  { 'temp' : 'hot', 
                    'windy' : 'weak',
                    'humidity' : 'low',
                    'outlook' : 'outcast',
                    'play' : 'yes'},
                  { 'temp' : 'cool',
                    'windy' : 'strong',
                    'humidity' : 'high',
                    'outlook' : 'rain',
                    'play' : 'no'}]
        self.df = pd.DataFrame(dataset)
        self.header = ['windy', 'outlook', 'humidity', 'outlook', 'play']
        self.label = 'play'
        self.test_list = ['yes', 'no', 'yes', 'no']

    def tearDown(self):
        pass

    def test_sigmoid(self):
        log = LogisticRegression()
        result = log.sigmoid(self.vector)
        self.assertEqual(result.tolist(), self.value.tolist())
    
    def test_train(self):
        log = LogisticRegression()
        weights = log.training(df=self.df, label=self.label, type='classification', dummies=True)
        self.listWeights = weights[0]
        self.uniqueTarget = weights[1]
        self.header = weights[-1]

        df = pd.get_dummies(self.df)

        for key in self.header:
            if key not in list(df):
                df[key] = pd.Series(np.zeros((len(df)),dtype=int))

        features = list()
        for key in self.header:
            for key2 in list(df):
                if key == key2:
                    features.append(df[key])
        features = np.array(features).T

        final_scores_list = []
        prediction = []

        #calculate the scores in test set
        for kind in range(len(self.uniqueTarget)):
            final_scores = np.dot(np.hstack((np.ones((features.shape[0], 1)),
                                               features)), self.listWeights[kind])
            final_scores_list.append(final_scores)

        #predict the label from scores
        for set in range (features.shape[0]):
            predict_dictionary = {}
            for kind in range(len(self.uniqueTarget)):
                predict_dictionary[self.uniqueTarget[kind]] = final_scores_list[kind][set]
            prediction.append(max(predict_dictionary, key = lambda classLabel: predict_dictionary[classLabel]))
        self.assertEqual(prediction,self.test_list)

    def test_predict(self):

        model = [[np.array([-0.18724959,  0.84022002, -0.45191971, -0.5755499 , -0.45191971,
            0.84022002, -0.5755499 ,  0.84022002, -1.02746961,  0.26467013,
            -0.45191971]), np.array([ 0.18724959, -0.84022002,  0.45191971,  0.5755499 ,  0.45191971,
            -0.84022002,  0.5755499 , -0.84022002,  1.02746961, -0.26467013,0.45191971])], np.array(['no', 'yes'], dtype='<U3'), ['humidity_high', 'humidity_low', 'humidity_normal', 'outlook_outcast', 'outlook_rain', 'outlook_sunny', 'temp_cool', 'temp_hot', 'windy_strong', 'windy_weak']]
        log=LogisticRegression()
        result = log.predict(df=self.df, model=model, dummies=True)
        self.assertEqual(result, self.test_list)
        
if __name__ == "__main__":
    unittest.main()
