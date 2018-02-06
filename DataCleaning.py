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

"""
#fill missing value
from random import randint
#fit
mode_values = dict()
for head in list(newdf):
    mode = newdf[head].mode()
    mode2 = mode[randint(0,len(mode)-1)]
    mode_values[head] = mode2 

#trasform
newnewdf = newdf.fillna(value=mode_values)
"""