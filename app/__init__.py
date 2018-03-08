
from flask import Flask  
from flask_mongoengine import MongoEngine

db=MongoEngine()
app=Flask(__name__)
app.config['MONGODB_SETTINGS']={
        'db':'MLdb'
        }

db.init_app(app)

from .admin import admin as admin_api
app.register_blueprint(admin_api, url_prefix='/admin')
