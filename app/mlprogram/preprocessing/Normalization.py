# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:39:41 2017

@author: falah.fakhruddin
"""
from app.mlprogram.Abstraction import AbstractPreprocessing
import numpy as np

class Normalization(AbstractPreprocessing):
    def fit(self, df, label):
        temp_df = df[label]
        del df[label]
        parameter = {}
        header = list(df)
        for item in header:
            parameter[item] = {}
            max_value = df[item].max()
            min_value = df[item].min()
            parameter[item]['max_value'] = max_value
            parameter[item]['min_value'] = min_value
        df[label] = temp_df
        return parameter

    def transform(self, df, label=None, values=None):
        newdf = df.copy()
        if values == None:
            values = self.fit(df, label)
        
        header = list(df)

        if label != None:
            del_index = header.index(label)
            del header[del_index]

        for item in header:
            min_value = values[item]['min_value']
            max_value = values[item]['max_value']
            newdf[item] = (df[item]-min_value)/(max_value-min_value)
        return newdf

