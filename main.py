import pandas as pd
from DatabaseConnector import *
from mongoengine import *
import numpy as np
import tools as tl
from LogisticRegression import LogisticRegression
from RegressionMainCode import MultiVariateRegression
from MLPClassifier import MLPClassifier
from NaiveBayess import NaiveBayess
from DataCleaning import DataCleaning
from Normalization import Normalization
from FeatureSelection import FeatureSelection

def preprocessing_step(dataset, method, preprocessing):
    #get db
    db = DatabaseConnector()
    list_db = db.get_collection(dataset, type=method, database='rawdb')
    df = list_db[0]
    features = np.array(df)
    header = list_db[2]

    #preprocessing step
    list_preprocessing = preprocessing
    for item in list_preprocessing:
        item.fit(features)
        features = item.transform(features)

    #save db
    features_frame = pd.DataFrame(features, columns=header)
    json_features = tl.transform_dataframe_json(features_frame)
    new_collection = "clean_"+str(dataset)
    db = DatabaseConnector()
    db.export_collection(json_features, new_collection, database='cleandb')

def training_step(dataset, target, method, algorithm, preprocessing):
    # get db
    db = DatabaseConnector()
    list_db = db.get_collection(dataset, target, type=method, database='cleandb')
    df = list_db[0]
    features = np.array(df)
    header = list_db[2]

    #training
    model = algorithm
    listWeights = model.training(features, target)

    #savedb
    connect(db="model")
    SaveModel(dataset=dataset, algorithm=algorithm, preprocessing=preprocessing, model=listWeights)

    return listWeights


if __name__ == "__main__":
    #preprocessing
    dataset = "irisdataset"
    preprocessing = [FeatureSelection(), DataCleaning()]
    preprocessing_step(dataset, preprocessing)

    # training
    dataset = "clean_irisdataset"
    target = "species"
    method = "classification"
    algorithm = LogisticRegression()
    training_step(dataset, target, method, algorithm, preprocessing)


"""
    prediction = model.predict(features, listWeights=weights)
    testing = model.testing(features, target, weights=weights)
"""

    """
    db = DatabaseConnector()
    homeprice = pd.read_csv("homeprice.txt")
    json_homperice = tl.transform_dataframe_json(homeprice)
    db.import_collection(json_homperice, "homeprice")
    """





