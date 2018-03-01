# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:14:33 2017

@author: falah.fakhruddin
"""
import numpy as np
import numpy.ma as ma
from scipy import stats
from random import randint
from app.mlprogram.Abstraction import AbstractPreprocessing

class DataCleaning(AbstractPreprocessing):
      def __init__ (self, missingValues = "NaN", strategy = "mode", axis = 0):
            self.missingValues = missingValues
            self.strategy = strategy 
            self.axis = axis
            
      def getMask(self, X, valueMask):
          """Compute the boolean mask X == missingValues."""
          if valueMask == "NaN":
              return np.isnan(X)
          else:
              return X == valueMask
        
      def fit (self, X):

            #check parameter
            allowedStrategy = ["mean", "median", "mode"]
            if self.strategy not in allowedStrategy:
                  raise ValueError("Can only use these strategies: {0} "
                             " got strategy={1}".format(allowedStrategy,
                                                        self.strategy))
            if self.axis not in [0, 1]:
                  raise ValueError("Can only impute missing values on axis 0 and 1, "
                             " got axis={0}".format(self.axis))

            
            mask = self.getMask(X, self.missingValues)
            maskedX = ma.masked_array(X, mask=mask)
                  
            if self.strategy == "mean":
                  meanMasked = np.ma.mean(maskedX, axis=self.axis)
                  self.mean = np.ma.getdata(meanMasked)
                  #self.mean = np.append(self.mean, [1])
                  return self.mean
                        
            elif self.strategy == "median":
                   medianMasked = np.ma.median(maskedX, axis=self.axis)
                   self.median = np.ma.getdata(medianMasked)
                   #self.median = np.append(self.median, [1])
                   return self.median
                      
            elif self.strategy == "mode":

                   mode = stats.mode(X)
                   self.mostFrequent = mode[0][0]
                   #self.mostFrequent = np.append(self.mostFrequent, [1])
                   return self.mostFrequent  
                  
      def transform (self, X):
            mask = self.getMask(X, self.missingValues)
            maskedX = ma.masked_array(X, mask=mask)
            if self.strategy == "mean":
                filledX = maskedX.filled(self.mean)
             
            elif self.strategy == "median":
                filledX = maskedX.filled(self.median)
                
            elif self.strategy == "mode":
                  filledX = maskedX.filled(self.mostFrequent)
                
            return filledX
                
if __name__ == "__main__":
      a = np.array([[np.NaN, 2, 3, 4], [0, 3, np.NaN, 2],[1,np.NaN,3,1], [np.NaN,4,3,5],
                    [2,3,2,np.NaN],[3,np.NaN,4,2],[2,3,4,2],[1,2,4,np.NaN]])
      b = np.array([[1],[2],[3],[4],[5], [6], [7], [8]])
      ab = np.append(a, b, axis=1)
      dc=DataCleaning(strategy="median")
      dc.fit(ab)
      ab = dc.transform(ab)

class DataCleaning2(AbstractPreprocessing):
    def __init__(self, style='mode'):
        self.style = style

    def median(self , newdf):
        median_values = dict()
        for head in list(newdf):
            median = newdf[head].median()
            median_values[head] = median
        return median_values

    def mean(self , newdf):
        mean_values = dict()
        for head in list(newdf):
            mean = newdf[head].mean()
            mean_values[head] = mean
        return mean_values

    def mode(self, newdf):
        mode_values = dict()
        for head in list(newdf):
            mode = newdf[head].mode()
            mode2 = mode[randint(0,len(mode)-1)]
            mode_values[head] = mode2
        return mode_values

    def fit(self, newdf):
        if self.style == 'mode':
            values = self.mode(newdf)
        elif self.style == 'mean':
            values = self.mean(newdf)
        elif self.style == 'median':
            values = self.median(newdf)

        return values

    def transform(self, newdf, values=None):
        if values == None:
            values = self.fit(newdf)
        newdf = newdf.fillna(value=values)
        return newdf
