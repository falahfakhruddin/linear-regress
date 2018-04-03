# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:41:39 2018

@author: falah.fakhruddin
"""
from app.mlprogram.Abstraction import AbstractML
import sys
from ..DatabaseConnector import DatabaseConnector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app.mlprogram import tools as tl


class MultiVariateRegression(AbstractML):
      def __init__(self, numSteps = 50000, learningRate = 1e-7, addIntercept = True):
            self.numSteps = numSteps
            self.learningRate = learningRate
            self.intercept = addIntercept

      def training(self, features=None, target=None, header=None, df=None, label=None, type=None, dummies=None):
          self.header=header
          if df is not None:
              list_df = tl.dataframe_extraction(df=df, label=label, type=type, dummies=dummies)
              features = list_df[0]
              target = list_df[1]
              self.header = list_df[2]

          model = list()

          if self.intercept:
              intercept = np.ones((features.shape[0], 1))
              features = np.hstack((intercept, features))

          self.weights = np.zeros(features.shape[1])
          error = []
          iteration = []
          for step in range(self.numSteps):
              predictions = np.dot(features, self.weights)

              # Update weights gradient descent
              deltaError = target - predictions
              mae = abs(sum(deltaError)/len(deltaError))
              gradient = -(1/len(features)) * (np.dot(features.T, deltaError))
              self.weights = self.weights - (self.learningRate * gradient)

              error.append(mae)
              iteration.append(step)

          # plot MAE
          plt.plot(iteration, error)
          plt.ylabel("Error")
          plt.xlabel("Iteration")
          plt.show()

          model.append(self.weights)
          if df is not None:
              model.append(self.header)

          return model

      def predict(self, features=None, df=None, model=None, dummies=None):
          if model is not None:
              self.weights = model[0]
              self.header = model[1]

          if df is not None:
              if dummies:
                  df = pd.get_dummies(df)

              for key in self.header:
                  if key not in list(df):
                      df[key] = pd.Series(np.zeros((len(df)),dtype=int))

              features = list()
              for key in self.header:
                  for key2 in list(df):
                      if key == key2:
                          features.append(df[key])
              features = np.array(features).T

          if self.intercept:
              intercept = np.ones((features.shape[0] , 1))
              features = np.hstack((intercept , features))

          prediction = np.dot(features, self.weights)
          print("\nPredictions :")
          print(prediction)
          prediction = prediction.tolist()
          return prediction

      def testing(self, features, target, model=None):
          #getprediction
          prediction = self.predict(features, model=model)
          
          #calculate error
          error = np.sum(abs(np.array(prediction) - target))
          error = np.sum(error)/len(target)
          return error

if __name__ == "__main__":
    
    """
    datafile = "homeprice"
    label = "Price"
    type = "regression"
    dummies = False

    # Load data and Preperation Data
    db = DatabaseConnector()
    df = db.get_collection(datafile)

    #training phase
    step = 10000
    multipleReg = MultiVariateRegression(addIntercept=True)
    model = multipleReg.training(df=df, label=label, type=type, dummies=dummies)

    #predict
    prediction= multipleReg.predict(df=df, model=model, dummies=dummies)

    #testing method
    error = multipleReg.testing(features, target)
    print ("\nError : %f" %error )
    """

