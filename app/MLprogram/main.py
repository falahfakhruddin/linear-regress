from MLrunner import *
from preprocessing.FeatureSelection import FeatureSelection
from preprocessing.DataCleaning import DataCleaning2
from algorithm.NaiveBayess import NaiveBayess
from algorithm.RegressionMainCode import MultiVariateRegression
from algorithm.MLPClassifier import SklearnNeuralNet
from algorithm.LogisticRegression import LogisticRegression

if __name__ == "__main__":

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

    # testing step
    dataset = "irisdataset"
    preprocessing = [FeatureSelection(), DataCleaning2()]
    algorithm = SklearnNeuralNet()
    ml = MLtest(dataset, preprocessing, algorithm)
    prediction = ml.prediction_step()
    print (prediction)
    
