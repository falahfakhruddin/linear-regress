import unittest
import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from app.mlprogram.algorithm.MLPClassifier import SklearnNeuralNet

class TestMLPClassifier(unittest.TestCase):
    def setUp(self):
        dataset = [{'sepal' : 2.7,
                    'petal' : 1.5,
                    'species': 'iris-setosa'},
                    {'sepal' : 3.8,
                     'petal' : 4.5,
                     'species' : 'iris-versicolor'},
                    {'sepal' : 3.3,
                     'petal' : 4.7,
                     'species' : 'iris-versicolor'},
                    {'sepal' : 2.4,
                     'petal' : 1.9,
                     'species' : 'iris-setosa'}]
        self.df = pd.DataFrame(dataset)
        self.header = ['sepal', 'petal']
        self.label = 'species'
        self.test_list = ['iris-versicolor', 'iris-setosa', 'iris-setosa', 'iris-versicolor']

    def tearDown(self):
        pass

    def test_training(self):
        neuralnet = SklearnNeuralNet()
        value = neuralnet.training(df=self.df, label=self.label, type='classification', dummies=False)
        for key in self.header:
            if key not in list(self.df):
                self.df[key] = pd.Series(np.zeros((len(self.df)), dtype=int))

        features = list()
        for key in self.header:
            for key2 in list(self.df):
                if key == key2:
                    features.append(self.df[key])
        features = np.array(features).T

        binary_mlp = bytearray(value[0])
        self.mlp = pickle.loads(binary_mlp)
        prediction = (self.mlp.predict(features))
        prediction = prediction.tolist()
        self.assertEqual(prediction, self.test_list)

    def test_predict(self):
        features = np.array([[2.7, 1.5], [3.8, 4.5], [3.3, 4.7], [2.4, 1.9]])
        target = np.array(['iris-setosa', 'iris-versicolor', 'iris-versicolor', 'iris-setosa'])
        mlp = MLPClassifier()
        mlp.fit(features, target)
        binary_mlp = pickle.dumps(mlp)
        model = list()
        model.append(binary_mlp)
        model.append(self.header)

        neuralnet = SklearnNeuralNet()
        result = neuralnet.predict(features=features, model=model, dummies=False)
        self.assertEqual(result, target.tolist())
       
    def test_testing(self):
        features = np.array([[2.7, 1.5], [3.8, 4.5], [3.3, 4.7], [2.4, 1.9]])
        target = np.array(['iris-setosa', 'iris-versicolor', 'iris-versicolor', 'iris-setosa'])
        mlp = MLPClassifier()
        mlp.fit(features, target)
        binary_mlp = pickle.dumps(mlp)
        model = list()
        model.append(binary_mlp)
        model.append(self.header)
        
        neuralnet = SklearnNeuralNet()
        result = neuralnet.testing(features, target, model=model)
        self.assertEqual(result, 1)

    def test_predict2(self):
        features = np.array([[2.7, 1.5], [3.8, 4.5], [3.3, 4.7], [2.4, 1.9]])
        target = np.array(['iris-setosa', 'iris-versicolor', 'iris-versicolor', 'iris-setosa'])
        mlp = MLPClassifier()
        mlp.fit(features, target)
        binary_mlp = pickle.dumps(mlp)
        model = list()
        model.append(binary_mlp)
        model.append(self.header)

        neuralnet = SklearnNeuralNet()
        result = neuralnet.predict(df = self.df, model=model, dummies=False)
        self.assertEqual(result, target.tolist())

