# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:14:53 2017

@author: falah.fakhruddin
"""

from pymongo import MongoClient
import pandas as pd

client = MongoClient()
db=client.newdb
collection=db.things.find()
df =pd.DataFrame(list(collection))
del df['_id']
features = df.iloc[:,:-1].values
label = df.iloc[:,-1].values.astype(str)

