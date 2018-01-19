from DatabaseConnector import DatabaseConnector
from LogisticRegression import LogisticRegression
from RegressionMainCode import MultiVariateRegression

db = DatabaseConnector()
list_db = db.get_collection("playtennis", "play")
df =list_db[0]
target = list_db[1]
header = list(df)
bias = ['bias']
header = bias + header
