# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:06:58 2017

@author: falah.fakhruddin
"""
import numpy as np
from numpy.random import shuffle
import numpy as np
import random
import math
import pandas as pd
from app.mlprogram.algorithm.LogisticRegression import LogisticRegression
from app.mlprogram.algorithm.RegressionMainCode import MultiVariateRegression
from app.mlprogram.algorithm.NaiveBayess import NaiveBayess
from app.mlprogram.algorithm.MLPClassifier import SklearnNeuralNet
from app.mlprogram.DatabaseConnector import *
from app.mlprogram import tools as tl

def partition(features, target, k):
    size = math.ceil(len(features) / float(k))
    partition_feature = [np.zeros([1, len(features[0])]) for i in range(k)]
    partition_target = [np.zeros([1, len(target[0])]) for i in range(k)]

    for entry in range(len(features)):
        x = assign(partition_feature, size, k)
        temp_feature = features[entry].reshape(1, len(features[entry]))
        temp_target = target[entry].reshape(1, len(target[entry]))
        partition_feature[x] = np.append(partition_feature[x], temp_feature, axis=0)
        partition_target[x] = np.append(partition_target[x], temp_target, axis=0)

    for x in range(k):
        partition_feature[x] = np.delete(partition_feature[x], 0, 0)
        partition_target[x] = np.delete(partition_target[x], 0, 0)

    return partition_feature, partition_target


def assign(partition_feature, size, k):
    x = random.randint(0, k - 1)
    while len(partition_feature[x]) > size:
        x = random.randint(0, k - 1)

    return x


def kfoldcv(classifier, features, target, header, k):
    target = target.reshape(len(target), 1)
    partition_feature, partition_target = partition(features, target, k)
    errors = list()

    # Run the algorithm k times, record error each time
    for i in range(k):
        training_set_feature = np.empty([1, len(features[0])])
        training_set_target = np.empty([1, len(target[0])])
        for j in range(k):
            if j != i:
                training_set_feature = np.append(training_set_feature, partition_feature[j], axis=0)
                training_set_target = np.append(training_set_target, partition_target[j], axis=0)

        # flatten training set
        training_set_feature = np.delete(training_set_feature, 0, 0)
        training_set_target = np.delete(training_set_target, 0, 0)
        training_set_target, =np.array(training_set_target.T)
        training_set = training_set_feature, training_set_target

        test_set_feature = partition_feature[i]
        test_set_target = partition_target[i]
        test_set_target, = np.array(test_set_target.T)
        test_set = test_set_feature, test_set_target


        # Train and classify model
        algorithm = classifier
        trained_classifier = train(algorithm, training_set, header)
        errors.append(testing(algorithm, test_set))

    return errors

def train(classifier, training_set, header):
    feature = training_set[0]
    target = training_set[1]
    return classifier.training(feature, target, header)

def testing(classifier, test_set):
    feature = test_set[0]
    target = test_set[1]
    return classifier.testing(feature, target)


if __name__ == "__main__":

    #extract data
    db = DatabaseConnector()
    df = db.get_collection("irisdataset")
    list_df = tl.dataframe_extraction(df, label='species', type='classification', dummies=False)
    features = list_df[0]
    target = list_df[1]
    header = list_df[2]

    #kfold
    k = 4
    errors = kfoldcv(SklearnNeuralNet(), features, target, k=k)

    # Compute statistics
    mean = sum(errors) / k
    variance = sum([(error - mean) ** 2 for error in errors]) / (k)
    standardDeviation = variance ** .5
    confidenceInterval = (mean - 1.96 * standardDeviation, mean + 1.96 * standardDeviation)
    print (mean)

