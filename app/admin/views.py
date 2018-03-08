from . import admin
from app.mlprogram import main
from flask import request, render_template, redirect, url_for, flash
import json

#adding new line
@admin.route('/')
def root():
    return 'hello world'

@admin.route('/playtennis', methods=['GET', 'POST'])
def input_field():
    if request.method == 'POST':
        field = request.form['Field']
        value = request.form['Value']
        return redirect(url_for('show_list', field=field, feature=value)) 
    return render_template('playtennis.html')

@admin.route('/playtennis/<string:field>/<string:feature>')
def show_list(field, feature):
    def del_id(data) : 
        del data['_id']
        return data
    data = [del_id(temp.to_mongo()) for temp in Playtennis.objects(**{'{}'.format(field) : feature})]
    admin.logger.info(data)
    return json.dumps(list(data))

@admin.route('/playtennis/add', methods=['GET'])
def playtennis_add_frm():
    return render_template('add_form.html')

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

@admin.route('/api/predict', methods = ['POST'])
def predicting():
    data = request.get_json()
    preprocessing = data['preprocessing']
    algorithm = data['algorithm']
    dataset = data['dataset']
    instance = data['instance']
    predict = main.prediction(dataset, preprocessing, algorithm, instance)
    return json.dumps({'prediction' : predict})

@admin.route('/evaluate', methods = ['POST'])
def validation():
    data = request.get_json()
    dataset = data['dataset']
    algorithm = data['algorithm']
    target = data['target']
    method = data['method']
    dummies = data['dummies']
    fold = data['fold']
    mean_error = main.evaluate(dataset, algorithm, target, method, dummies, fold)
    return json.dumps({'errors' : mean_error})
