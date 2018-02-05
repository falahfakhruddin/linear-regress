# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:41:39 2018

@author: falah.fakhruddin
"""
from Abstraction import AbstractML
import sys
from DatabaseConnector import DatabaseConnector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MultiVariateRegression(AbstractML):
      def __init__(self, numSteps = 50000, learningRate = 1e-7, addIntercept = True):
            self.numSteps = numSteps
            self.learningRate = learningRate
            self.intercept = addIntercept

      def training(self, features, target):
          listWeights = list()

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

          listWeights.append(self.weights)

          return listWeights

      def predict(self, features, listWeights=None):
          if listWeights != None:
              newWeights = listWeights[0]
              self.weights = newWeights

          prediction = np.dot(np.hstack((np.ones((features.shape[0], 1)),
                                 features)), self.weights)
          print("\nPredictions :")
          print(prediction)
          return prediction

      def testing(self, features, target, listWeights=None):
          #getprediction
          prediction = self.predict(features, listWeights)

          #calculate error
          error = np.sum(abs(prediction - target))
          error = np.sum(error)/len(target)
          return error

if __name__ == "__main__":
    temp=sys.argv
    #extract data from txt file into array
    db = DatabaseConnector()
    list_db = db.get_collection("homeprice", "Price", type="regression")
    df = list_db[0]
    target = list_db[1]
    header = list(df)
    bias = ['bias']
    header = bias + header

    # extract feature
    features = np.array(df)

    #training phase
    step = 10000
    multipleReg = MultiVariateRegression()
    weights = multipleReg.training(features, target)

    print ("Weights :")
    print (weights)

    #testing method
    error = multipleReg.testing(features, target)
    print ("\nError : %f" %error )


