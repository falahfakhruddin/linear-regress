# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:41:39 2018

@author: falah.fakhruddin
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from MultiVariateRegression import MultiVariateRegression

class MultiVariateRegression():
      def __init__(self, numSteps = 5000, learningRate = 10e-5, addIntercept = True):
            self.numSteps = numSteps
            self.learningRate = learningRate
            self.intercept = addIntercept
            
      def training(self, features, target):
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

          return self.weights
              
      def predict(self, features, weights=None):
          if weights == None:
              weights = self.weights

          prediction = np.dot(np.hstack((np.ones((features.shape[0], 1)),
                                 features)), self.weights)
          print("\nPredictions :")
          print(prediction)
          return prediction

      def testing(self, features, target, weights=None):
          #getprediction
          prediction = self.predict(features, weights)

          #calculate error
          error = np.sum(abs(prediction - target))
          error = np.sum(error)/len(target)
          return error

if __name__ == "__main__":
    temp=sys.argv
    #extract data from txt file into array
    txtfile = "homeprice.txt"
    data = pd.read_csv(txtfile)
    print (data)
    header = list(data)
    features = data.iloc[:,:-1].values.astype(float)
    target = data.iloc[:,-1].values.astype(float)

    #training phase
    step = 5000
    multipleReg = MultiVariateRegression(numSteps = step, learningRate = 1e-7, addIntercept=True)
    weights = multipleReg.training(features, target)

    print ("Weights :")
    print (weights)

    #testing method
    error = multipleReg.testing(features, target)
    print ("\nError : %f" %error )


b = np.array([[1,2,3],[3,2,1]])
a = np.array([[1,2,3],[3,2,1]])
