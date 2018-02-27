# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 08:53:37 2017

@author: falah.fakhruddin
"""
from Abstraction import AbstractPreprocessing
import numpy as np

class FeatureSelection(AbstractPreprocessing):
      def __init__(self, percentage = 0.5):
            self.percentage = percentage

      def fit (self, df):
            return None

      def transform(self, df, values=None):
            threshold = int((self.percentage * df.shape[1]) + 0.5)  # fit
            return df.dropna(thresh=threshold)

if __name__ == "__main__":
      X = np.array([[np.NaN, np.NaN, 3, 4], [0, 3, np.NaN, 2],[1,np.NaN,3,1], [np.NaN,4,3,5],
              [2,3,2,np.NaN],[3,np.NaN,4,2],[2,3,4,2],[1,np.NaN,4,np.NaN]])
      Y = np.array([["satu"],["dua"],["tiga"],["empat"],["lima"], ["enam"], ["tujuh"], ["delapan"]])
      XY = np.append(X, Y, axis=1)
      fs = FeatureSelection()
      fs.fit(XY)
      XY=fs.transform(XY)


      Z = np. array([1,2,3,4,5,6,7,8])
      ZT = Z.reshape(len(Z),1)
"""
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

      threshold = int((self.percentage * df.shape[1]) + 0.5)  # fit
      return df.dropna(thresh=threshold)  # transform


#dropping missing value
threshold=int((0.5 * df.shape[1]) + 0.5) # fit
newdf=df.dropna(thresh=threshold) # transform
"""