from DatabaseConnector import DatabaseConnector
from LogisticRegression import LogisticRegression
from RegressionMainCode import MultiVariateRegression
import numpy as np
import pandas as pd
import tools as tl

db = DatabaseConnector()
homeprice = pd.read_csv("homeprice.txt")
json_homperice = tl.transform_dataframe_json(homeprice)
db.import_collection(json_homperice, "homeprice")


db = DatabaseConnector()
list_db = db.get_collection("playtennis", "play")
df =list_db[0]
target = list_db[1]
header = list(df)
bias = ['bias']
header = bias + header

#extract feature
features = np.array(df)

model = LogisticRegression()
weights = model.training(features, target)



weights.shape
weight_frame = pd.DataFrame(weights.reshape(1,len(weights)), columns=header)
weights_json = tl.transform_dataframe_json(weight_frame)