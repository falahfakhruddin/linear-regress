import json
from app.mlprogram.validation import CrossValidation as cv
from app.mlprogram.MLrunner import *
from app.mlprogram.preprocessing.FeatureSelection import FeatureSelection
from app.mlprogram.preprocessing.DataCleaning import DataCleaning2
from app.mlprogram.algorithm.NaiveBayess import NaiveBayess
from app.mlprogram.algorithm.RegressionMainCode import MultiVariateRegression
from app.mlprogram.algorithm.MLPClassifier import SklearnNeuralNet
from app.mlprogram.algorithm.LogisticRegression import LogisticRegression
from app.mlprogram import translator as trans

def training():
    # training step
    dataset = "irisdataset"
    target = "species"
    method = "classification"
    dummies = False
    database = 'MLdb'
    algorithm = LogisticRegression()
    preprocessing = [FeatureSelection(), DataCleaning2()]
    ml = run.MLtrain(dataset, target, method, algorithm, preprocessing, dummies, database)
    listWeights = ml.training_step()
    return listWeights

def prediction(dataset, str_prepro, str_algo, instance):
    # testing step
    preprocessing = trans.preprocessing_trans(str_prepro)
    algorithm = trans.algorithm_trans(str_algo)
    ml = MLtest(dataset, preprocessing, algorithm, instance)
    prediction = ml.prediction_step()
    return prediction

def evaluate(dataset, str_algo, label, method, dummies, fold):
    algorithm = trans.algorithm_trans(str_algo)    
    db = DatabaseConnector()
    df=db.get_collection(dataset)
    list_df = tl.dataframe_extraction(df, label, method, dummies)
    features = list_df[0]
    target = list_df[1]
    header = list_df[2]
    errors = cv.kfoldcv(algorithm, features, target, header, fold)
    mean = sum(errors)/fold
    return mean

