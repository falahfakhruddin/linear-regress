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
          return self.weights, error, iteration
              
      def predict(self, features):
          final_scores = np.dot(np.hstack((np.ones((features.shape[0], 1)),
                                 features)), self.weights)
          
          return final_scores

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
    weights, error, iteration = multipleReg.training(features, target)

    print ("Weights :")
    print (weights)

    #plot MAE
    plt.plot(iteration, error)
    plt.ylabel("Error")
    plt.xlabel("Iteration")
    plt.show()

    #testing method
    prediction = multipleReg.predict(features)
    print ("\nPredictions :")
    print (prediction)
