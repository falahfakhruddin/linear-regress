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

class DataCleaning2(AbstractPreprocessing):
    def __init__(self, method='mode'):
        self.method = method

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

    def fit(self, newdf, label):
        temp_df = newdf[label]
        del newdf[label]
        
        if self.method == 'mode':
            values = self.mode(newdf)
        elif self.method == 'mean':
            values = self.mean(newdf)
        elif self.method == 'median':
            values = self.median(newdf)
        newdf[label] = temp_df

        return values

    def transform(self, newdf, label=None, values=None):
        if values == None:
            values = self.fit(newdf, label)
        newdf = newdf.fillna(value=values)
        return newdf
                
if __name__ == "__main__":
      a = np.array([[np.NaN, 2, 3, 4], [0, 3, np.NaN, 2],[1,np.NaN,3,1], [np.NaN,4,3,5],
                    [2,3,2,np.NaN],[3,np.NaN,4,2],[2,3,4,2],[1,2,4,np.NaN]])
      b = np.array([[1],[2],[3],[4],[5], [6], [7], [8]])
      ab = np.append(a, b, axis=1)
      dc=DataCleaning(strategy="median")
      dc.fit(ab)
      ab = dc.transform(ab)


