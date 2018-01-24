# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:14:33 2017

@author: falah.fakhruddin
"""
import numpy as np
import numpy.ma as ma
from Abstraction import AbstractPreprocessing
from scipy import stats            
            

class DataCleaning(AbstractPreprocessing):
      def __init__ (self, missingValues = "NaN", strategy = "mean", axis = 0):
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
                  return self.mean
                        
            elif self.strategy == "median":
                   medianMasked = np.ma.median(maskedX, axis=self.axis)
                   self.median = np.ma.getdata(medianMasked)
                   return self.median
                      
            elif self.strategy == "mode":
                   mode = stats.mode(X)
                   self.mostFrequent = mode[0][0]
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
      dc=DataCleaning(strategy="mode")
      dc.fit(a)
      a = dc.transform(a)