from .preprocessing.FeatureSelection import FeatureSelection
from .preprocessing.DataCleaning import DataCleaning2
from .algorithm.NaiveBayess import NaiveBayess
from .algorithm.RegressionMainCode import MultiVariateRegression
from .algorithm.MLPClassifier import SklearnNeuralNet
from .algorithm.LogisticRegression import LogisticRegression

def algorithm_trans(algo_str):
    if algo_str == 'naive bayess':
        return NaiveBayess()
    elif algo_str == 'regression':
        return MultiVariateRegression()
    elif algo_str == 'logistic regression':
        return LogisticRegression()
    elif algo_str == 'neural network':
        return SklearnNeuralNet()

def preprocessing_trans(prepo_str):
    prepo_list = []
    for prepo in prepo_str:
        if prepo == "data cleaning":
            prepo_list.append(DataCleaning2())
        elif prepo == "feature selection":
            prepo_list.append(FeatureSelection())
    return prepo_list
