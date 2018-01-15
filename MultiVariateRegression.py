# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 08:56:49 2017

@author: falah.fakhruddin
"""
import numpy as np
from pymongo import MongoClient
import pandas as pd
from Normalization import Normalization

class MultiVariateRegression():
      def trainingMethod(self, features, target, num_steps, learning_rate, add_intercept = False):
          if add_intercept:
              intercept = np.ones((features.shape[0], 1))
              features = np.hstack((intercept, features))    
        
          self.weights = np.zeros(features.shape[1])
       
          for step in range(num_steps):
              predictions = np.dot(features, self.weights)
      
              # Update weights gradient descent
              output_error_signal = target - predictions
              
              gradient = np.dot(features.T, output_error_signal) / len(features)
              self.weights += learning_rate * gradient 
              return self.weights
              
      def predict(self, features):
          final_scores = np.dot(np.hstack((np.ones((features.shape[0], 1)),
                                 features)), self.weights)
          preds = np.round(final_scores)
          
          return preds
          
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

[features, target] = iris()

normalize = Normalization()
normalize.fit(features)
features = normalize.transform(features)

#Training Step
var_reg = MultiVariateRegression()
weights = var_reg.trainingMethod(features, target,
                     num_steps = 500, learning_rate = 5e-5, add_intercept=True)
print (weights)

#Testing Step
print (var_reg.predict(features))