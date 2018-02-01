from DatabaseConnector import DatabaseConnector
from LogisticRegression import LogisticRegression
from RegressionMainCode import MultiVariateRegression
import numpy as np
import pandas as pd

db = DatabaseConnector()
method = 'classification'
list_db = db.get_collection("irisdataset", "species", type=method)
df = list_db[0]
target = list_db[1]
header = list_db[2]
bias = ["bias"]
header = bias +header

# extract feature
features = np.array(df)

model = LogisticRegression()
weights = model.training(features, target)

weight_frame = pd.DataFrame(weights.reshape(1, len(weights)), columns=header)
weights_json = tl.transform_dataframe_json(weight_frame)


"""
db = DatabaseConnector()
homeprice = pd.read_csv("homeprice.txt")
json_homperice = tl.transform_dataframe_json(homeprice)
db.import_collection(json_homperice, "homeprice")
"""





