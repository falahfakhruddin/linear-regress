# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Importing Librariees
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
dataset= pd.read_csv('Data.csv')
X= dataset.iloc[:, -2].values
Y= dataset.iloc[:, 1].values

#Splitting training set and test set
from sklearn.cross_validation import train_test_split
X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size = 2/5, random state=0)