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

def dataframe_extraction(df, label, type='classification', dummies=False ):
    if type == 'classification':
        target = df[label].values.astype(str)
    elif type == 'regression':
        target = df[label].values.astype(float)
    del df[label]

    if dummies:
        features = pd.get_dummies(df)*1
        header = list(features)
        features = features.values

    else:
        header = list(df)
        features = df.iloc[: , :].values

    return features, target, header

