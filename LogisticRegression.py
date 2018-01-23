# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:28:20 2017

@author: falah.fakhruddin
"""
from Abstraction import AbstractML
import sys
import numpy as np
from pymongo import MongoClient
import pandas as pd

class LogisticRegression (AbstractML):
      def __init__(self, num_steps=50000, learning_rate=5e-5, intercept=True):
          self.num_steps = num_steps
          self.learning_rate = learning_rate
          self.intercept = intercept

      def sigmoid(self, scores):
          return 1 / (1 + np.exp(-scores))

      def training(self, features, target):
          #adding extra features for coefficient parameter
          if self.intercept:
              intercept = np.ones((features.shape[0], 1))
              features = np.hstack((intercept, features))

          self.uniqueTarget = np.unique(target) #get unique value of target
          self.listWeights=[]

          for kind in range (len(self.uniqueTarget)): #looping for each unique target
                #set binary value for unique target
                dfTarget = pd.DataFrame(target)
                dfTarget.set_value(dfTarget[0]!= self.uniqueTarget[kind],[0],0)
                dfTarget.set_value(dfTarget[0]== self.uniqueTarget[kind],[0],1)
                numericalTarget = dfTarget.iloc[:,-1].values.astype(int)

                #initiate weights value
                weights = np.zeros(features.shape[1])

                #gradient descent step
                for step in range(self.num_steps):
                    scores = np.dot(features, weights) #calculate prediction value
                    predictions = self.sigmoid(scores) #transform with sigmoid function

                    # delta error of real value and predict value
                    output_error_signal = numericalTarget - predictions

                    gradient = np.dot(features.T, output_error_signal) #value of derivative function for each parameter
                    weights += self.learning_rate * gradient #update weights
                self.listWeights.append(weights)

          return self.listWeights

      def predict(self, features, weights=None):
          if weights == None:
              weights = self.listWeights

          final_scores_list = []
          prediction = []

          #calculate the scores in test set
          for kind in range(len(self.uniqueTarget)):
                final_scores = np.dot(np.hstack((np.ones((features.shape[0], 1)),
                                       features)), weights[kind])
                final_scores_list.append(final_scores)

          #predict the label from scores
          for set in range (features.shape[0]):
                predict_dictionary = {}
                for kind in range(len(self.uniqueTarget)):
                      predict_dictionary[self.uniqueTarget[kind]] = final_scores_list[kind][set]
                prediction.append(max(predict_dictionary, key = lambda classLabel: predict_dictionary[classLabel]))

          prediction = np.array(prediction)
          print("\nPrediction :")
          print (prediction)
          return prediction

      def testing(self, features, target, weights=None):
          if weights == None:
              weights = self.listWeights

          #get prediction
          prediction = self.predict(features, weights)

          #calculate error
          error = 0
          for i in range(len(prediction)):
              if target[i] != prediction[i]:
                  error = error + 1

          error = float(error*100/len(target))

          return float(error)

if __name__ == "__main__":
      def playtennis() :
            client = MongoClient()
            db=client.newdb
            collection=db.playtennis.find()
            df =pd.DataFrame(list(collection))
            del df['_id']
            print (df)
            target = df['play'].values.astype(str)
            del df['play']
            df = pd.get_dummies(df)
            features = df.iloc[:,:].values
            print (df)
            return [features, target]

      def iris() :
            client = MongoClient()
            db=client.newdb
            collection=db.irisdataset.find()
            df =pd.DataFrame(list(collection))
            del df['_id']
            print (df)
            features = df.iloc[:,:-1].values.astype(float)
            target = df.iloc[:,-1].values.astype(str)
            return [features, target]

      temp=sys.argv
      #Load data and Preperation Data

      #Training Step

      dataset = temp[-1]

      if dataset == "playtennis":
            [features, target] = playtennis()
      else:
            [features, target] = iris()

      #Training Step
      log_reg = LogisticRegression(num_steps=50000, learning_rate=5e-5)
      weights = log_reg.training(features, target)

      print ("Weights:")
      print (weights)

      #Testing Step
      error = log_reg.testing(features, target)
      print ("\n Error : %f" %error +"%")