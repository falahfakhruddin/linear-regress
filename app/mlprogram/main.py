import json
import seaborn as sns
import matplotlib.pyplot as plt
from app.mlprogram.validation import CrossValidation as cv
from app.mlprogram.MLrunner import *
from app.mlprogram.preprocessing.FeatureSelection import FeatureSelection
from app.mlprogram.preprocessing.DataCleaning import DataCleaning2
from app.mlprogram.algorithm.NaiveBayess import NaiveBayess
from app.mlprogram.algorithm.RegressionMainCode import MultiVariateRegression
from app.mlprogram.algorithm.MLPClassifier import SklearnNeuralNet
from app.mlprogram.algorithm.LogisticRegression import LogisticRegression
from app.mlprogram import translator as trans

def training(dataset, target, str_algo, str_prepro, method, dummies, database):
    # training step
    algorithm = trans.algorithm_trans(str_algo)
    preprocessing = trans.preprocessing_trans(str_prepro)
    ml = MLtrain(dataset, target, method, algorithm, preprocessing, dummies, database)
    massage = ml.training_step()
    return massage

def prediction(dataset, str_prepro, str_algo, instance):
    # testing step
    preprocessing = trans.preprocessing_trans(str_prepro)
    algorithm = trans.algorithm_trans(str_algo)
    ml = MLtest(dataset, preprocessing, algorithm, instance)
    prediction = ml.prediction_step()
    return prediction

def evaluate(dataset, str_algo, label, method, dummies, fold):
    algorithm = trans.algorithm_trans(str_algo)    
    db = DatabaseConnector()
    df=db.get_collection(dataset)
    list_df = tl.dataframe_extraction(df, label, method, dummies)
    features = list_df[0]
    target = list_df[1]
    header = list_df[2]
    errors = cv.kfoldcv(algorithm, features, target, header, fold)
    
    performance = None
    if method == 'regression':
        mean = sum(errors)/fold
        variance = sum([(error - mean)**2 for error in errors])/fold
        standardDeviation = variance**.5
        performance = {'Error' : mean,
                       'Standard Deviation' : standardDeviation}
    
    elif method == 'classification':
        sum_cm = sum(errors)
        print(sum_cm)
        cm_df = pd.DataFrame(sum_cm)
        plt.figure()
        sns.heatmap(cm_df, annot=True)
        plt.ylabel('True Label')
        plt.xlabel('Prediction Label')
        plt.show()

        #calculate accuracy
        accuracy = sum([sum_cm[i][i] for i in range(len(sum_cm))])/np.sum(sum_cm)
        performance = { 'Accuracy' : accuracy}

    return performance

def model_query(field, feature):
    def del_id(data):
        del data['_id']
        del data['create']
        return data
    data = [del_id(temp.to_mongo()) for temp in SaveModel.objects(**{'{}'.format(field): feature})]
    return data
