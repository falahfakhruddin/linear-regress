import unittest
from pymongo import MongoClient 
from app.mlprogram.DatabaseConnector import *

class DatabaseConnectorTest(unittest.TestCase):
    def setUp(self):
        self.database = 'testDB'
        self.dataset = [{'outlook' : 'outcast',
                         'temp' : 'hot',
                         'humidity' : 'high',
                         'windy' : True,
                         'play' : 'yes'},
                        {'outlook' : 'sunny',
                         'temp' : 'hot',
                         'humidity' : 'high',
                         'windy' : False,
                         'play' : 'yes'},
                        {'outlook' : 'rainy',
                         'temp' : 'cold',
                         'humidity' : 'low',
                         'windy' : True,
                         'play' : 'no'}]

        self.collection = ['playtennis']

    def tearDown(self):
        pass

    def test_export_collection(self):
        dc = DatabaseConnector()
        input_collection = dc.export_collection(jsonfile=self.dataset, collection=self.collection[0], database = self.database)
        client = MongoClient()
        db = client[self.database]
        collection = db.collection_names(include_system_collections=False)
        for item in collection:
            if item == self.collection[0]:
                result = True
                break
            else:
                result = False
        self.assertTrue(result)        

    def test_get_collection(self):
        db = DatabaseConnector()
        temp = db.get_collection(datafile=self.collection[0], database=self.database)
        result = isinstance(temp, object)
        self.assertTrue(result)
    
    def test_check_collection(self):
        db = DatabaseConnector()
        result = db.check_collection(database=self.database)
        self.assertEqual(result, self.collection)

if __name__ == "__main__":
    unittest.main()
