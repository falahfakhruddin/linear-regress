from MLrunner import *
from FeatureSelection import FeatureSelection
from DataCleaning import DataCleaning2
from NaiveBayess import NaiveBayess
from RegressionMainCode import MultiVariateRegression
from MLPClassifier import SklearnNeuralNet
from LogisticRegression import LogisticRegression

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
