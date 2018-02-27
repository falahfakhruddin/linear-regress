import json
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

def prediction(dataset, str_prepro, str_algo):
    # testing step
    preprocessing = trans.preprocessing_trans(str_prepro)
    algorithm = trans.algorithm_trans(str_algo)
    ml = MLtest(dataset, preprocessing, algorithm)
    prediction = ml.prediction_step()
    return prediction

#if __name__ == "__main__":
def run():
    post = json.dumps({ 
        "preprocessing" : ["feature selection", "data cleaning"],
        "dataset" : "irisdataset",
        "algorithm" : "neural network"
        })

    requestjson = json.loads(post)
    dataset = requestjson['dataset']
    preprocessing = trans.preprocessing_trans(requestjson['preprocessing'])
    algorithm = trans.algorithm_trans(requestjson['algorithm'])
    print (preprocessing)
    ml=run.MLtest(dataset, preprocessing, algorithm)
    prediction = ml.prediction_step()
    #print(prediction)
