import pandas as pd
from Abstraction import AbstractML
import collections
from DatabaseConnector import DatabaseConnector
import tools as tl

class NaiveBayess(AbstractML):
    def __init__(self, header=None):
        self.header=header
        self.weightDict = {}
        self.labelCounts = collections.defaultdict(lambda: 0)

    def training(self, features=None, target=None, df=None, label=None, type=None, dummies='no'):
        #extract value of dataframe
        if df is not None:
            list_df = tl.dataframe_extraction(df=df , label=label , type=type )
            features = list_df[0]
            target = list_df[1]
            self.header = list_df[2]

        listWeights = list()
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


        listWeights.append(self.weightDict)
        listWeights.append(self.labelCounts)
        listWeights.append(self.header)
        return listWeights


    def predict(self, features=None, df=None, model=None, dummies='no'):
        if model is not None:
            self.weightDict = model[0]
            self.labelCounts = model[1]
            self.header = model[2]

        if df is not None:
            target = list(set(list(df)).difference(self.header))
            del df[target[0]]
            features = df.values

        prediction = list()
        for vector in features:
            list_vector = vector.tolist()
            probabilityPerLabel = {}
            for label in self.labelCounts:
                tempProb = 1
                for featureValue in vector:
                    tempProb *= self.weightDict[label][self.header[list_vector.index(featureValue)]][featureValue]
                tempProb *= self.labelCounts[label]
                probabilityPerLabel[label] = tempProb
            print(probabilityPerLabel)
            prediction.append(max(probabilityPerLabel, key=lambda classLabel: probabilityPerLabel[classLabel]))

        print("\nPrediction :")
        print(prediction)
        return prediction


    def testing(self, features, target, weights=None):
        # get prediction
        prediction = self.predict(features, model=weights)

        # calculate error
        error = 0
        for i in range(len(prediction)):
            if target[i] != prediction[i]:
                error += 1

        error = float(error * 100 / len(target))
        return float(error)

if __name__ == "__main__":
    #get collection
    datafile = "playtennis_FeatureSelection_DataCleaning2"
    label = "play"
    type = "classification"

    # Load data and Preperation Data
    db = DatabaseConnector()
    df = db.get_collection(datafile, database='MLdb')

    # Training Step
    nb = NaiveBayess()
    model2 = nb.training(df=df , label=label , type=type )

    from mongoengine import *
    from DatabaseConnector import *
    connect('MLdb')
    temp = SaveModel.objects(algorithm=nb.__class__.__name__)
    for data in temp:
        model = data.model

    #predict
    predicton = nb.predict(df=df , model=model)

    model[1]['yes']

    if model is not None:
        weightDict = model[0]
        labelCounts = model[1]
        header = model[2]

    if df is not None:
        features = df.values

    prediction = list()
    for vector in features:
        list_vector = vector.tolist()
        probabilityPerLabel = {}
        for label in labelCounts:
            x = label
            tempProb = 1
            for featureValue in vector:
                y = featureValue
                tempProb *= weightDict[x][header[list_vector.index(featureValue)]][featureValue]
            tempProb *= labelCounts[x]
            probabilityPerLabel[x] = tempProb
        print(probabilityPerLabel)
        prediction.append(max(probabilityPerLabel , key=lambda classLabel: probabilityPerLabel[classLabel]))

    print("\nPrediction :")
    print(prediction)

    target = list(set(list(df)).difference(header))

    del df[target[0]]
    target[0]