# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:39:41 2017

@author: falah.fakhruddin
"""
import numpy as np

class Normalization():

      def fit(self, data):
            self.data_min=np.min(data, axis=0)
            self.data_max=np.max(data, axis=0)
            self.data_range = self.data_max - self.data_min
            
      def transform(self, data):
            for i  in range (0,data.shape[1]):
                  for j in range (0,data.shape[0]):
                        data[j][i] = ((data[j][i] - self.data_min[i]) / 
                                     (self.data_max[i]-self.data_min[i]))
            return data
