# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:39:41 2017

@author: falah.fakhruddin
"""
from app.mlprogram.Abstraction import AbstractPreprocessing
import numpy as np

class Normalization(AbstractPreprocessing):
    def fit(self, df):
        parameter = {}
        header = list(df)
        for item in header:
            parameter[item] = {}
            max_value = df[item].max()
            min_value = df[item].min()
            parameter[item]['max_value'] = max_value
            parameter[item]['min_value'] = min_value
        return parameter

    def transform(self, df, parameter=None):
        header = list(df)
        newdf = df.copy()
        if parameter == None:
            parameter = self.fit(df)

        for item in header:
            min_value = parameter[item]['min_value']
            max_value = parameter[item]['max_value']
            newdf[item] = (df[item]-min_value)/(max_value-min_value)
        return newdf

