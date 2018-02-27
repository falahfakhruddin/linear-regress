import json
from mongoengine import *
from app.mlprogram import MLrunner as run
from app.mlprogram.preprocessing.FeatureSelection import FeatureSelection
from app.mlprogram.preprocessing.DataCleaning import DataCleaning2
from app.mlprogram.algorithm.NaiveBayess import NaiveBayess
from app.mlprogram.algorithm.RegressionMainCode import MultiVariateRegression
from app.mlprogram.algorithm.MLPClassifier import SklearnNeuralNet
from app.mlprogram.algorithm.LogisticRegression import LogisticRegression
from app.mlprogram import translator as trans

if __name__ == "__main__":
    post = json.dumps({	"preprocessing" : ["feature selection", "data cleaning"],
                           "dataset" : "irisdataset",
                           "algorithm" : "neural network"
                           })

    requestjson = json.loads(post)
    dataset = requestjson['dataset']
    preprocessing = trans.preprocessing_trans(requestjson['preprocessing'])
    algorithm = trans.algorithm_trans(requestjson['algorithm'])
    print (preprocessing)
    print (dataset)
    print (algorithm)
    ml=run.MLtest(dataset, preprocessing, algorithm)
    prediction = ml.prediction_step()
    print(prediction)
