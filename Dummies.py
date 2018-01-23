# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:00:25 2018

@author: falah.fakhruddin
"""
import pandas as pd 

def dummies(features):
      features = pd.DataFrame(features)
      features = pd.get_dummies(features)
      features = df.iloc[:,:].values.astype(int)
      return features 