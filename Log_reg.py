# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:28:20 2017

@author: falah.fakhruddin
"""

import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
import pandas as pd

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))
    
def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll

def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))    
  
    weights = np.zeros(features.shape[1])
 
    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with log likelihood gradient
        output_error_signal = target - predictions
        
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        if step % 10000 == 0:
            print (log_likelihood(features, target, weights))
        
    return weights


def playtennis() :
      client = MongoClient()
      db=client.newdb
      collection=db.playtennis.find()
      df =pd.DataFrame(list(collection))
      del df['_id']
      df2=pd.DataFrame(df['play'])
      del df['play']
      df = pd.get_dummies(df)
      df2.set_value(df2['play']=='no',['play'],0)
      df2.set_value(df2['play']=='yes',['play'],1)
      features = df.iloc[:,:].values
      target = df2.iloc[:,-1].values.astype(int)
      return [features, target]

def iris() :
      client = MongoClient()
      db=client.newdb
      collection=db.irisdataset.find()
      df =pd.DataFrame(list(collection))
      del df['_id']
      df.set_value(df['species']=='Iris-setosa',['species'],0)
      df.set_value(df['species']=='Iris-versicolor',['species'],1)
      features = df.iloc[:89,:-1].values.astype(float)
      target = df.iloc[:89,-1].values.astype(int)
      return [features, target]

#Load data and Preperation Data
#[features, target] = playtennis()
[features, target] = iris()

#Training Step
weights = logistic_regression(features, target,
                     num_steps = 50000, learning_rate = 5e-5, add_intercept=True)
print (weights)

#Testing Step
final_scores = np.dot(np.hstack((np.ones((features.shape[0], 1)),
                                 features)), weights)
preds = np.round(sigmoid(final_scores))

#Plot Figure
plotx=np.linspace(0,6, num=features.shape[0])
plotx.resize(plotx.shape[0],1)
final_scores2 = np.dot(np.hstack((np.ones((plotx.shape[0], 1)),
                                 plotx)), weights[:2])
plt.figure(figsize = (4, 3))
plt.scatter(features[:,0], preds,
            c = preds == target - 1, alpha = .8, s = 10)
plt.plot(plotx,sigmoid(final_scores2))
