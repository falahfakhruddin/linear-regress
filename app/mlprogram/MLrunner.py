import pandas as pd
from mongoengine import *
import numpy as np
from datetime import date
from app.mlprogram.DatabaseConnector import *
from app.mlprogram import tools as tl
from app.mlprogram.algorithm.LogisticRegression import LogisticRegression
from app.mlprogram.algorithm.RegressionMainCode import MultiVariateRegression
from app.mlprogram.algorithm.MLPClassifier import SklearnNeuralNet
from app.mlprogram.algorithm.NaiveBayess import NaiveBayess
from app.mlprogram.preprocessing.DataCleaning import DataCleaning2
from app.mlprogram.preprocessing.Normalization import Normalization
from app.mlprogram.preprocessing.FeatureSelection import FeatureSelection


class MLtrain():
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

            for item in self.preprocessing:
                self.dataset = self.dataset + "_" + item.__class__.__name__

            # get db
            db = DatabaseConnector()
            if self.dataset in db.check_collection(database='MLdb'):

                connect('MLdb')
                temp = SavePrepocessing.objects(dataset=self.dataset)
                for data in temp:
                    self.prepo_parameter = data.preprocessing
                df = db.get_collection(self.dataset, database='MLdb')

            else:
                self.dataset, waste = self.dataset.split("_", 1)
                df = self.preprocessing_step()

        print ("aselole")
        print(self.prepo_parameter)

        #training
        ml = self.algorithm
        model = ml.training(df=df,label=self.label,type=self.type,dummies=self.dummies)

        #savedb
        connect(db="MLdb")
        savedb = SaveModel(dataset=self.dataset, algorithm= self.algorithm.__class__.__name__,
                           preprocessing=self.prepo_parameter, model=model, dummies=self.dummies)
        savedb.save()
        return model

class MLtest():
    def __init__(self, dataset, preprocessing, algorithm):
        self.dataset = dataset
        self.preprocessing = preprocessing
        self.algorithm = algorithm

    def implement_preprocessing(self):
        # get db
        db = DatabaseConnector()
        df = db.get_collection(self.dataset , database='MLdb')

        for item in self.preprocessing:
            prepro_name = item.__class__.__name__
            connect('MLdb')
            temp = SaveModel.objects(dataset=self.dataset)
            for data in temp:
                prepo_dict = data.preprocessing
            values = (prepo_dict[prepro_name])
            df = item.transform(df, values=values)

        return df

    def prediction_step(self):
        for item in self.preprocessing:
            self.dataset = self.dataset + "_" + item.__class__.__name__

        df = self.implement_preprocessing()
        print(df)

        connect('MLdb')
        temp = SaveModel.objects(algorithm=self.algorithm.__class__.__name__)
        for data in temp:
            model = data.model
            dummies = data.dummies

        ml = self.algorithm
        prediction = ml.predict(df=df, model=model, dummies=dummies)

        return prediction



if __name__ == "__main__":
    dataset = "homeprice"
    preprocessing = [FeatureSelection(), DataCleaning2()]
    ml = MLtrain()
    result = ml.preprocessing_step(dataset, preprocessing)

    from mongoengine import *
    from DatabaseConnector import *

    dataset = "irisdataset"
    preprocessing = [FeatureSelection(), DataCleaning2()]

    for item in preprocessing:
        dataset = dataset + "_" + item.__class__.__name__

    print(dataset)
    for item in preprocessing:
        prepro_name = item.__class__.__name__
        connect('MLdb')
        temp = SaveModel.objects(dataset=dataset)
        for data in temp:
            prepo_dict=data.preprocessing
        values = (prepo_dict[prepro_name])
        df = item.transform(values=values)
