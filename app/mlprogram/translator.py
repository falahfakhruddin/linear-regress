from app.mlprogram.preprocessing.FeatureSelection import FeatureSelection
from app.mlprogram.preprocessing.DataCleaning import DataCleaning2
from app.mlprogram.algorithm.NaiveBayess import NaiveBayess
from app.mlprogram.algorithm.RegressionMainCode import MultiVariateRegression
from app.mlprogram.algorithm.MLPClassifier import SklearnNeuralNet
from app.mlprogram.algorithm.LogisticRegression import LogisticRegression
from app.mlprogram.preprocessing.Normalization import Normalization
from app.mlprogram.preprocessing.FeatureSelection import FeatureSelection
from app.mlprogram.preprocessing.DataCleaning import DataCleaning2

def algorithm_trans(algorithm):
    if algorithm == 'naive bayess':
        return NaiveBayess()
    elif algorithm == 'regression':
        return MultiVariateRegression()
    elif algorithm == 'logistic regression':
        return LogisticRegression()
    elif algorithm == 'neural network':
        return  SklearnNeuralNet()

def preprocessing_trans(prepo_str):
    prepo_list = []
    for prepo in prepo_str:
        if prepo == "data cleaning":
            prepo_list.append(DataCleaning2())
        elif prepo == "feature selection":
            prepo_list.append(FeatureSelection())
        elif prepo == "normalization":
            prepo_list.append(Normalization())
    return prepo_list
