# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:28:20 2017

@author: falah.fakhruddin
"""
import sys
from app.mlprogram.Abstraction import AbstractML
import numpy as np
import pandas as pd
from app.mlprogram import tools as tl
from ..DatabaseConnector import *

class LogisticRegression (AbstractML):
      def __init__(self, num_steps=50000, learning_rate=5e-5, intercept=True):
          self.num_steps = num_steps
          self.learning_rate = learning_rate
          self.intercept = intercept

      def sigmoid(self, scores):
          return 1 / (1 + np.exp(-scores))

      def training(self, features=None, target=None, header=None, df=None, label=None, type=None, dummies=None):
          self.header=header
          #extracting value of dataframe
          if df is not None:
              list_df = tl.dataframe_extraction(df=df, label=label, type=type, dummies=dummies)
              features = list_df[0]
              target = list_df[1]
              self.header = list_df[2]

          model = list()
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

          model.append(self.listWeights)
          model.append(self.uniqueTarget)
          model.append(self.header)

          return model

      def predict(self, features=None, df=None, model=None, dummies=None):
          if model is not None:
              self.listWeights = model[0]
              self.uniqueTarget = model[1]
              self.header = model[-1]

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

          final_scores_list = []
          prediction = []

          #calculate the scores in test set
          for kind in range(len(self.uniqueTarget)):
              final_scores = np.dot(np.hstack((np.ones((features.shape[0], 1)),
                                               features)), self.listWeights[kind])
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

      def testing(self, features, target, model=None):
          #get prediction
          prediction = self.predict(features, model=model)

          #calculate error
          error = 0
          for i in range(len(prediction)):
              if target[i] != prediction[i]:
                  error = error + 1

          error = float(error*100/len(target))

          return float(error)

if __name__ == "__main__":

      datafile = "irisdataset"
      label = "species"
      type = "classification"
      dummies="no"

      #Load data and Preperation Data
      db = DatabaseConnector()
      df = db.get_collection(datafile)
      #Training Step
      log_reg = LogisticRegression(num_steps=50000, learning_rate=5e-5)
      model = log_reg.training(df=df,label=label,type=type,dummies=dummies)

      predicton = log_reg.predict(df=df, model=model, dummies=dummies)

      #Testing Step
      print ("\n Error : %f" %error +"%")

      list_dum = np.array([["high", "false", "outcast", "hot"],["low", "truw", "rain", "cold"],
                           ["high", "truw", "rain", "hot"], ["low", "false", "outcast", "cold"]])
      dum_head = ["humidity", "windy", "outlook", "temp"]
      df2=pd.DataFrame(list_dum, columns=dum_head)
      dummies_df2 = pd.get_dummies(df2, prefix="", prefix_sep='')

      tes_dum = np.array([["high", "truw", "hot"], ["low", "false", "hot"]])
      tes_head = ["humidity", "windy", "temp"]
      df3=pd.DataFrame(tes_dum, columns=tes_head)
      dummies_df3 = pd.get_dummies(df3, prefix="", prefix_sep='')

      dummies_df3.shape
      len(dummies_df3)
      for key in list(dummies_df2):
          if key not in list(dummies_df3):
              dummies_df3[key] = pd.Series(np.zeros((len(dummies_df3)), dtype=int))
              print (key)
