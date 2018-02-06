import pandas as pd
from DatabaseConnector import *
from mongoengine import *
import numpy as np
import tools as tl
from LogisticRegression import LogisticRegression
from RegressionMainCode import MultiVariateRegression
from MLPClassifier import MLPClassifier
#from NaiveBayess import NaiveBayess
from DataCleaning import DataCleaning
from Normalization import Normalization
from FeatureSelection import FeatureSelection

def preprocessing_step(dataset, method, label, preprocessing):
    #get db
    db = DatabaseConnector()
    list_db = db.get_collection(dataset, label=label, type=method, database='rawdb')
    df = list_db[0]
    target = list_db[1]
    features = np.array(df)
    header = list_db[2]
    header = header + [label]

    #join features with target
    target = target.reshape(len(target), 1)
    new_dataset = np.append(features, target, axis=1)

    #preprocessing step
    list_preprocessing = preprocessing
    for item in list_preprocessing:
        item.fit(new_dataset)
        new_dataset = item.transform(new_dataset)

    #save db
    dataset_frame = pd.DataFrame(new_dataset, columns=header)
    json_features = tl.transform_dataframe_json(dataset_frame)
    new_collection = "clean_"+str(dataset)
    db = DatabaseConnector()
    result = db.export_collection(json_features, new_collection, database='cleandb')

def training_step(dataset, target, method, algorithm, preprocessing):
    # get db
    db = DatabaseConnector()
    list_db = db.get_collection(dataset, target, type=method, database='cleandb')
    df = list_db[0]
    features = np.array(df)
    target = list_db[1]
    header = list_db[2]

    #training
    model = algorithm
    listWeights = model.training(features, target)

    #savedb
    connect(db="modeldb")
    savedb = SaveModel(dataset=dataset, algorithm=str(algorithm), preprocessing=preprocessing, model=listWeights)
    savedb.save()
    return listWeights


if __name__ == "__main__":
    #preprocessing
    dataset = "irisdataset"
    method = "classification"
    label = "species"
    preprocessing = [FeatureSelection(missingValues='nan'), DataCleaning(missingValues='nan')]
    preprocessing_step(dataset, method, label, preprocessing)

    # training
    dataset = "clean_homeprice"
    target = "Price"
    method = "regression"
    algorithm = MultiVariateRegression()
    training_step(dataset, target, method, algorithm, preprocessing)



    prediction = model.predict(features, listWeights=weights)
    testing = model.testing(features, target, weights=weights)


    
    db = DatabaseConnector()
    homeprice = pd.read_csv("rawplaytennis.csv")
    json_homperice = tl.transform_dataframe_json(homeprice)
    db.export_collection(homeprice, "homeprice", database="datapool")

    db = DatabaseConnector()
    list_db = db.get_collection('playtennis', label='play', database='rawdb')
    df = list_db[0]
    print (df[0][0])
    type(df[0][0])



