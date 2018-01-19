import pandas as pd


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
    json = pd.DataFrame.to_json(dataframe, orient='records')
    json_file = dict_from_str(json)

    return json_file
