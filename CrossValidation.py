# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:06:58 2017

@author: falah.fakhruddin
"""
import numpy as np
from numpy.random import shuffle

# Split a dataset into a train and test set
class CrossValidation():
      def train_test_split(self, dataset, test_size = 0.60):
            train_size = int(test_size * len(dataset))
            shuffle(dataset)
            train = dataset[:train_size]
            dataset = np.delete(dataset, np.s_[0:train_size], axis = 0)
            train = np.array_split(train, [-1], axis = 1)
            test = np.array_split(dataset, [-1], axis = 1)
            X_train = np.array(train[0])
            y_train = train[1]
            X_test = test[0]
            y_test = test[1]
            return X_train, X_test, y_train, y_test

