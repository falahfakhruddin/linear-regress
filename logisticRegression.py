# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:28:20 2017

@author: falah.fakhruddin
"""

import numpy as np
from pymongo import MongoClient
import pandas as pd

class logisticRegression ():
      def sigmoid(self, scores):
          return 1 / (1 + np.exp(-scores))
      
      def trainingMethod(self, features, target, num_steps, learning_rate, add_intercept = False):
          if add_intercept:
              intercept = np.ones((features.shape[0], 1))
              features = np.hstack((intercept, features))    
        
          self.weights = np.zeros(features.shape[1])
       
          for step in range(num_steps):
              scores = np.dot(features, self.weights)
              predictions = self.sigmoid(scores)
      
              # Update weights with log likelihood gradient
              output_error_signal = target - predictions
              
              gradient = np.dot(features.T, output_error_signal)
              self.weights += learning_rate * gradient
              
          return weights  
         
      def predict(self, features):
          final_scores = np.dot(np.hstack((np.ones((features.shape[0], 1)),
                                 features)), self.weights)
          preds = np.round(self.sigmoid(final_scores))
          
          return preds

def playtennis() :
      client = MongoClient()
      db=client.newdb
      collection=db.playtennis.find()
      df =pd.DataFrame(list(collection))
      del df['_id']  
      df.set_value(df['play']=='no',['play'],0)
      df.set_value(df['play']=='yes',['play'],1)
      target = df['play'].values.astype(int)
      del df['play']
      df = pd.get_dummies(df)
      features = df.iloc[:,:].values
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
[features, target] = playtennis()
#[features, target] = iris()

#Training Step
log_reg = logisticRegression()
weights = log_reg.trainingMethod(features, target,
                     num_steps = 50000, learning_rate = 5e-5, add_intercept=True)
print (weights)

#Testing Step
print (log_reg.predict(features))