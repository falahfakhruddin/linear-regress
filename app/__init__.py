
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_mongoengine import MongoEngine
#from .flaskmongo import PlayTennis
import json
from app.mlprogram import main

db=MongoEngine()
app=Flask(__name__)
app.config['MONGODB_SETTINGS']={
        'db':'newdb'
        }

db.init_app(app)

class PlayTennis(db.Document):
    windy = db.StringField()
    humidity = db.StringField()
    outlook = db.StringField()
    temp = db.StringField()
    play = db.StringField()
    number = db.IntField()

@app.route('/')
def root():
    return 'hello world'

@app.route('/playtennis', methods=['GET', 'POST'])
def input_field():
    if request.method == 'POST':
        field = request.form['Field']
        value = request.form['Value']
        return redirect(url_for('show_list', field=field, feature=value)) 
    return render_template('playtennis.html')

@app.route('/playtennis/<string:field>/<string:feature>')
def show_list(field, feature):
    def del_id(data) : 
        del data['_id']
        return data
    data = [del_id(temp.to_mongo()) for temp in Playtennis.objects(**{'{}'.format(field) : feature})]
    app.logger.info(data)
    return json.dumps(list(data))

@app.route('/playtennis/add', methods=['GET'])
def playtennis_add_frm():
    return render_template('add_form.html')

@app.route('/api/playtennis', methods=['POST'])
def playtennis_insert():
    data = request.get_json()
    app.logger.info(data)
    play=PlayTennis(windy = data['windy'], humidity=data['humidity'], outlook=data['outlook'], 
                    temp=data['temp'], play=data['play'], number=data['number'])
    play.save()
    return json.dumps({'status' : 'success'})

@app.route('/testing', methods=['GET', 'POST'])
def testing():
    if request.method == 'POST':
        data= request.form['gender']
        return json.dumps({'data' : str(data)})
    return render_template('radio.html')

@app.route('/api/predict', methods = ['POST'])
def predicting():
    data = request.get_json()
    preprocessing = data['preprocessing']
    algorithm = data['algorithm']
    dataset = data['dataset']
    predict = main.prediction(dataset, preprocessing, algorithm)
    return json.dumps({'prediction' : predict})
