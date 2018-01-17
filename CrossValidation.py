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


# Split a dataset into a train and test set
class CrossValidation():
    def train_test_split(self, dataset, test_size=0.60):
        train_size = int(test_size * len(dataset))
        shuffle(dataset)
        train = dataset[:train_size]
        dataset = np.delete(dataset, np.s_[0:train_size], axis=0)
        train = np.array_split(train, [-1], axis=1)
        test = np.array_split(dataset, [-1], axis=1)
        X_train = np.array(train[0])
        y_train = train[1]
        X_test = test[0]
        y_test = test[1]
        return X_train, X_test, y_train, y_test


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


def kfoldcv(classifier, features, target, k):
    partition_feature, partition_target = partition(features, target, k)
    errors = list()

    # Run the algorithm k times, record error each time
    for i in range(k):
        i = 0
        training_set_feature = np.empty([1, len(features[0])])
        training_set_target = np.empty([1, len(target[0])])
        for j in range(k):
            if j != i:
                training_set_feature = np.append(training_set_feature, partition_feature[j], axis=0)
                training_set_target = np.append(training_set_target, partition_target[j], axis=0)

        # flatten training set
        training_set_feature = np.delete(training_set_feature, 0, 0)
        training_set_target = np.delete(training_set_target, 0, 0)
        training_set = training_set_feature, training_set_target

        test_set_feature = partition_feature[i]
        test_set_target = partition_target[i]
        test_set = test_set_feature, test_set_target

        # Train and classify model
        model = classifier
        trained_classifier = train(model, training_set)
        errors.append(predict(model, test_set))

    # Compute statistics
    mean = sum(errors) / k
    variance = sum([(error - mean) ** 2 for error in errors]) / (k)
    standardDeviation = variance ** .5
    confidenceInterval = (mean - 1.96 * standardDeviation, mean + 1.96 * standardDeviation)

    _output(
        "\t\tMean = {0:.2f} \n\t\tVariance = {1:.4f} \n\t\tStandard Devation = {2:.3f} \n\t\t95% Confidence interval: [{3:.2f}, {4:.2f}]" \
            .format(mean, variance, standardDeviation, confidenceInterval[0], confidenceInterval[1]))

    return (errors, mean, variance, confidenceInterval, k)


def train(classifier, training_set):
    feature = training_set[0]
    target = training_set[1]

    return classifier.training(feature, target)

def predict(classifier, test_set):
    feature =
"""
features = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 5, 2, 3], [4, 5, 2, 3],
                     [5, 5, 2, 4], [6, 6, 4, 3], [7, 5, 4, 1], [8, 3, 5, 1]])
target = np.array(['1', '2', '3', '4', '5', '6', '7', '8'])

training_set = training_set_feature, training_set_target
"""