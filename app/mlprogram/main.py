from .MLrunner import *
from .preprocessing.FeatureSelection import FeatureSelection
from .preprocessing.DataCleaning import DataCleaning2
from .algorithm.NaiveBayess import NaiveBayess
from .algorithm.RegressionMainCode import MultiVariateRegression
from .algorithm.MLPClassifier import SklearnNeuralNet
from .algorithm.LogisticRegression import LogisticRegression
from . import translator as trans

def training():
    # training step
    dataset = "irisdataset"
    target = "species"
    method = "classification"
    dummies = False
    database = 'MLdb'
    algorithm = LogisticRegression()
    preprocessing = [FeatureSelection(), DataCleaning2()]
    ml = MLtrain(dataset, target, method, algorithm, preprocessing, dummies, database)
    listWeights = ml.training_step()
    return listWeights

def prediction(dataset, str_prepro, str_algo):
    # testing step
    preprocessing = trans.preprocessing_trans(str_prepro)
    algorithm = trans.algorithm_trans(str_algo)
    ml = MLtest(dataset, preprocessing, algorithm)
    prediction = ml.prediction_step()
    return prediction
   
if __name__ == "__main__":
    dataset = "irisdataset"
    algorithm = SklearnNeuralNet()
    preprocessing = [FeatureSelection(), DataCleaning2()]
    ml=MLtest(dataset, preprocessing, algorithm)
    prediction = ml.prediction_step()
    print (prediction)
