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
from datetime import date


class MLrun():
    def __init__(self, dataset, label, type, algorithm, preprocessing, dummies, database):
        self.dataset = dataset
        self.label = label
        self.type = type
        self.algorithm = algorithm
        self.preprocessing = preprocessing
        self.dummies = dummies
        self.database = database

    def preprocessing_step(self):
        #get db
        db = DatabaseConnector()
        df = db.get_collection(self.dataset, database='rawdb')

        #preprocessing step
        self.prepo_parameter = dict()
        list_preprocessing = self.preprocessing
        for item in list_preprocessing:
            value = item.fit(df)
            self.prepo_parameter[item.__class__.__name__]=value
            df = item.transform(df)
            self.dataset = self.dataset+"_"+item.__class__.__name__

        #save collection
        strdate = date.today().isoformat()
        parameter = self.prepo_parameter
        json_features = tl.transform_dataframe_json(df)
        db = DatabaseConnector()
        result = db.export_collection(json_features, self.dataset, database='MLdb')

        #save prepro parameter
        connect(db="MLdb")
        savedb = SavePrepocessing(dataset=self.dataset, preprocessing=parameter)
        savedb.save()

        return df

    def training_step(self):

        if self.database == "rawdb":
            df = self.preprocessing_step()

        elif self.database == "MLdb":
            # get db
            db = DatabaseConnector()
            df = db.get_collection(self.dataset, database='MLdb')

            connect('MLdb')
            temp = SaveModel.objects(dataset=self.dataset)
            for data in temp:
                self.prepo_parameter = data.preprocessing


        #training
        ml = self.algorithm
        model = ml.training(df=df,label=self.label,type=self.type,dummies=self.dummies)

        #savedb
        connect(db="MLdb")
        savedb = SaveModel(dataset=self.dataset, algorithm= self.algorithm.__class__.__name__, preprocessing=self.prepo_parameter, model=model)
        savedb.save()
        return model

if __name__ == "__main__":
    dataset = "homeprice"
    preprocessing = [FeatureSelection(), DataCleaning2()]
    ml = MLrun()
    result = ml.preprocessing_step(dataset, preprocessing)