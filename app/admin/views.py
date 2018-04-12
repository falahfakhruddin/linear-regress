from . import admin
from app.mlprogram import main
from flask import request, render_template, redirect, url_for, flash
from app.mlprogram.DatabaseConnector import *
import json

@admin.route('/')
def root():
    return 'hello world'

@admin.route('/api/playtennis', methods=['POST'])
def playtennis_insert():
    data = request.get_json()
    admin.logger.info(data)
    play=PlayTennis(windy = data['windy'], humidity=data['humidity'], outlook=data['outlook'], 
                    temp=data['temp'], play=data['play'], number=data['number'])
    play.save()
    return json.dumps({'status' : 'success'})

@admin.route('/testing', methods=['GET', 'POST'])
def testing():
    if request.method == 'POST':
        data= request.form['gender']
        return json.dumps({'data' : str(data)})
    return render_template('radio.html')

@admin.route('/predict', methods = ['POST'])
def predicting():
    data = request.get_json()
    preprocessing = data['preprocessing']
    algorithm = data['algorithm']
    dataset = data['dataset']
    instance = data['instance']
    predict = main.prediction(dataset, preprocessing, algorithm, instance)
    return json.dumps({'prediction' : predict})

@admin.route('/evaluate', methods = ['POST'])
def evaluation():
    data = request.get_json()
    dataset = data['dataset']
    algorithm = data['algorithm']
    target = data['target']
    method = data['method']
    dummies = data['dummies']
    fold = data['fold']
    performance = main.evaluate(dataset, algorithm, target, method, dummies, fold)
    return json.dumps({'performance' : performance})
        
@admin.route('/training', methods=['POST'])
def training():
    data = request.get_json()
    dataset = data['dataset']
    target = data['target']
    algorithm = data['algorithm']
    preprocessing = data['preprocessing']
    method = data['method']
    dummies = data['dummies']
    database = data['database']
    train_result = main.training(dataset, target, algorithm, preprocessing, method, dummies, database)
    return json.dumps({"massage" : train_result})

@admin.route('/model/<string:field>/<string:feature>')
def model(field, feature):
    data = main.model_query(field, feature)
    return json.dumps(list(data))
