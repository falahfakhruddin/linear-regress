# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:14:53 2017

@author: falah.fakhruddin
"""

from pymongo import MongoClient
import pandas as pd
import datetime

class DatabaseConnector():
    def get_collection(self, datafile, label, type='classification', database='newdb', dummies='no'): #get dataframe
        client = MongoClient()
        db = client[database]
        collection = db[datafile].find()
        df = pd.DataFrame(list(collection))
        del df['_id']

        if type == 'classification':
            target = df[label].values.astype(str)
        elif type == 'regression':
            target = df[label].values.astype(float)
        del df[label]

        if dummies =='yes':
            features = pd.get_dummies(df)
            header = list(features)
            features = features.values

        else:
            header = list(df)
            features = df.iloc[:, :].values

        return features, target, header,

    def import_collection(self, jsonfile, collection, database='newdb'): #upload json file into database
        client = MongoClient()
        db = client[database]
        upload = db[collection]
        if type(jsonfile) == list:
            result = upload.insert_many(jsonfile).inserted_ids
        elif type(jsonfile) == dict:
            result = upload.insert_one(jsonfile).inserted_id

        return result

"""
logout = {
    "topic" : "pyeongyang",
    "mapper" : {
        "-$type" : "&$logout",
        "-$api_key" : "#SIFTSCIENCE.apikey",
        "-$user_id" : "$.result.result.username",
    },
    "is_added" : False,
    "time_stamp" : datetime.datetime.utcnow(),
    "is_active" : False
}
"""
