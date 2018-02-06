import pandas as pd
import json


def dict_from_str(dict_str):
    while True:
        try:
            file = eval(dict_str)
        except NameError as e:
            key = e.message.split("'")[1]
            dict_str = dict_str.replace(key, "'{}'".format(key))
        else:
            return file


def transform_dataframe_json(dataframe):
    json_file = json.loads(dataframe.T.to_json()).values()
    return json_file


def dummies(features):
    features = pd.DataFrame(features)
    features = pd.get_dummies(features)
    features = features.iloc[:, :].values.astype(int)
    return features

