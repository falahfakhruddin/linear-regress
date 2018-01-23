# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 08:53:37 2017

@author: falah.fakhruddin
"""
import AbstractPreprocessing
import numpy as np

class FeatureSelection(Abstraction):
      def __init__(self, missingValues = "NaN", percentage = 0.5):
            self.list = []
            self.percentage = percentage
            self.missingValues = missingValues
            
      def fit(self, X) :      
            lengthRow = len(X[0])
            threshold = int(self.percentage * lengthRow)
            sequence = 0
            
            for row in X :
                  if self.missingValues == "NaN":
                        np.isnan(X[sequence])
                        sumMissValues = len(np.where(np.isnan(X[sequence]))[0])
                  else: 
                        sumMissValues = len(np.where(X[sequence] == self.missingValues)[0])
                  
                  if sumMissValues >= threshold:
                        self.list.append(sequence)
                  sequence +=1
                  
      def transform(self, X):
            return np.delete(X, self.list, 0)
            

if __name__ == "__main__":
      X = np.array([[np.NaN, np.NaN, 3, 4], [0, 3, np.NaN, 2],[1,np.NaN,3,1], [np.NaN,4,3,5],
              [2,3,2,np.NaN],[3,np.NaN,4,2],[2,3,4,2],[1,np.NaN,4,np.NaN]])
      fs = FeatureSelection()
      fs.fit(X)
      X=fs.transform(X)