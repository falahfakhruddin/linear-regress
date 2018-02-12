from MLrunner import MLrun
from FeatureSelection import FeatureSelection
from DataCleaning import DataCleaning2
from NaiveBayess import NaiveBayess

if __name__ == "__main__":
    #preprocessing
    dataset = "playtennis"
    target = "play"
    method = "classification"
    dummies = 'no'
    database = 'rawdb'
    algorithm = NaiveBayess()
    preprocessing = [FeatureSelection(), DataCleaning2()]
    ml = MLrun()
    listWeights = ml.training_step(dataset, target, method, algorithm, preprocessing, dummies, database)

    #load model for testing


    # str(preprocessing)

    # prediction = model.predict(features, listWeights=weights)
    # testing = model.testing(features, target, weights=weights)
    #
    #
    #
    # db = DatabaseConnector()
    # homeprice = pd.read_csv("rawplaytennis.csv")
    # json_homperice = tl.transform_dataframe_json(homeprice)
    # db.export_collection(homeprice, "homeprice", database="datapool")
    #
    # db = DatabaseConnector()
    # list_db = db.get_collection('playtennis', label='play', database='rawdb')
    # df = list_db[0]
    # print (df[0][0])
    # type(df[0][0])
    #

#
#     yx = bytearray(y[0])
#     import pickle
#     zz = pickle.loads(yx)
#     new = zz.predict(features)
#
#     j=0
#     for i in range(0, len(target)):
#         if new[i] != target[i]:
#             j+=1
#     print (j)
# len(features)