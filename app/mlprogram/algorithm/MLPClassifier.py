from sklearn.neural_network import MLPClassifier
from app.mlprogram.Abstraction import AbstractML
from ..DatabaseConnector import DatabaseConnector
import pandas as pd
import numpy as np
import pickle
from app.mlprogram import tools as tl


class SklearnNeuralNet(AbstractML):
    def __init__(self,size=10, solver='sgd', learning_rate=0.01, max_iter=5000):
        self.mlp = MLPClassifier(hidden_layer_sizes=size, solver=solver, learning_rate_init=learning_rate, max_iter=max_iter)

    def training(self, features=None, target=None, df=None, label=None, type=None, dummies=None):
        if df is not None:
            list_df = tl.dataframe_extraction(df=df , label=label , type=type, dummies=dummies)
            features = list_df[0]
            target = list_df[1]
            self.header = list_df[2]

        model = list()
        self.mlp.fit(features, target)
        binary_mlp = pickle.dumps(self.mlp)
        model.append(binary_mlp)

        if df is not None:
            model.append(self.header)

        return model

    def predict(self, features=None, df=None, model=None, dummies=None):
        if model != None:
            binary_mlp = bytearray(model[0])
            self.mlp = pickle.loads(binary_mlp)
            self.header = model[1]

        if df is not None:
            if dummies:
                df = pd.get_dummies(df)

            for key in self.header:
                if key not in list(df):
                    df[key] = pd.Series(np.zeros((len(df)) , dtype=int))

            features = list()
            for key in self.header:
                for key2 in list(df):
                    if key == key2:
                        features.append(df[key])
            features = np.array(features).T

        prediction = (self.mlp.predict(features))
        prediction = prediction.tolist()
        return prediction

    def testing(self, features, target, model=None):
        if model != None:
            binary_mlp = model[0]
            weights = pickle.loads(binary_mlp)
            self.mlp = weights

        error = self.mlp.score(features, target)
        return error

if __name__ == "__main__":
    datafile = "irisdataset"
    label = "species"
    type = "classification"
    dummies = "no"

    # Load data and Preperation Data
    db = DatabaseConnector()
    df = db.get_collection(datafile)

    # Training Step
    mlp = SklearnNeuralNet()
    model = mlp.training(df=df , label=label , type=type , dummies=dummies)

    predicton = mlp.predict(df=df , model=model , dummies=dummies)

