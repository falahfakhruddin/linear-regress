import pandas as pd
from DatabaseConnector import *
from mongoengine import *
import numpy as np
import tools as tl
from LogisticRegression import LogisticRegression
from RegressionMainCode import MultiVariateRegression
from MLPClassifier import SklearnNeuralNet
from NaiveBayess import NaiveBayess
from DataCleaning import DataCleaning2
from Normalization import Normalization
from FeatureSelection import FeatureSelection

class MLrun():
    def preprocessing_step(self, dataset, preprocessing):
        #get db
        db = DatabaseConnector()
        df = db.get_collection(dataset, database='rawdb')

        #preprocessing step
        list_preprocessing = preprocessing
        for item in list_preprocessing:
            df = item.transform(df)

        #save db
        json_features = tl.transform_dataframe_json(df)
        new_collection = "clean_"+str(dataset)
        db = DatabaseConnector()
        result = db.export_collection(json_features, new_collection, database='MLdb')
        return df

    def training_step(self, dataset, label, type, algorithm, preprocessing, dummies, database):

        if database == "rawdb":
            df = self.preprocessing_step(dataset, preprocessing)

        elif database == "MLdb":
            # get db
            db = DatabaseConnector()
            df = db.get_collection(dataset, database='MLdb')

        preprocessing2 = list()
        for item in preprocessing:
            preprocessing2.append(item.__class__.__name__)

        #training
        ml = algorithm
        model = ml.training(df=df,label=label,type=type,dummies=dummies)

        #savedb
        connect(db="MLdb")
        savedb = SaveModel(dataset=dataset, algorithm= algorithm.__class__.__name__, preprocessing=preprocessing2, model=model)
        savedb.save()
        return model

