# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:14:53 2017

@author: falah.fakhruddin
"""

from pymongo import MongoClient
import pandas as pd


def playtennis():
    client = MongoClient()
    db = client.newdb
    collection = db.playtennis.find()
    df = pd.DataFrame(list(collection))
    del df['_id']
    print(df)
    target = df['play'].values.astype(str)
    del df['play']
    df = pd.get_dummies(df)
    features = df.iloc[:, :].values
    print(df)
    return [features, target]


def iris():
    client = MongoClient()
    db = client.newdb
    collection = db.irisdataset.find()
    df = pd.DataFrame(list(collection))
    del df['_id']
    print(df)
    features = df.iloc[:, :-1].values.astype(float)
    target = df.iloc[:, -1].values.astype(str)
    return [features, target]

