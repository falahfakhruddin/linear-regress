import pandas as pd
from Abstraction import AbstractML
import collections
from DatabaseConnector import DatabaseConnector

class NaiveBayess(AbstractML):
    def __init__(self, header):
        self.header = header
        self.weightDict = {}
        self.labelCounts = collections.defaultdict(lambda: 0)

    def training(self, features, target):
        for counter in range(0, len(target)):  # label count
            self.labelCounts[target[counter]] += 1  # udpate count of the label
            self.weightDict[target[counter]] = {}  # fv[-1] == target name

        for i in range(0, len(self.header)):
            y = features[:, i]
            used = set()
            unique = [x for x in y if x not in used and (used.add(x) or True)]
            for label in self.labelCounts:
                self.weightDict[label][self.header[i]] = {}
                for line in unique:
                    temp = 1 / (self.labelCounts[label] + len(unique))
                    self.weightDict[label][self.header[i]].update({line: temp})

        target_count = 0
        for fv in features:
            for counter in range(0, len(fv)):
                temp = 1 / (self.labelCounts[target[target_count]] + len(self.weightDict[target[target_count]][self.header[counter]]))
                self.weightDict[target[target_count]][self.header[counter]][fv[counter]] += temp
            target_count += 1

        return self.weightDict


    def predict(self, features, weightDict=None):
        if weightDict == None:
            weightDict = self.weightDict

        prediction = list()
        for vector in features:
            list_vector = vector.tolist()
            probabilityPerLabel = {}
            for label in self.labelCounts:
                tempProb = 1
                for featureValue in vector:
                    tempProb *= weightDict[label][self.header[list_vector.index(featureValue)]][featureValue]
                tempProb *= self.labelCounts[label]
                probabilityPerLabel[label] = tempProb
            print(probabilityPerLabel)
            prediction.append(max(probabilityPerLabel, key=lambda classLabel: probabilityPerLabel[classLabel]))

        print("\nPrediction :")
        print(prediction)
        return prediction


    def testing(self, features, target, weights=None):
        if weights == None:
            weights = self.weightDict

        # get prediction
        prediction = self.predict(features, weights)

        # calculate error
        error = 0
        for i in range(len(prediction)):
            if target[i] != prediction[i]:
                error += 1

        error = float(error * 100 / len(target))
        return float(error)

if __name__ == "__main__":
    #get collection
    db = DatabaseConnector()
    df = db.get_collection("playtennis", "play", dummies='no')
    features = df[0]
    target = df[1]
    header = df[2]

    #training
    nb = NaiveBayess(header)
    weight = nb.training(features, target)

    #testing
    result = nb.testing(features, target)
