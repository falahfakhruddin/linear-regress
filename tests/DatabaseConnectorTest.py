import unittest
from pymongo import MongoClient 
from app.mlprogram.DatabaseConnector import *
from unittest.mock import patch, Mock, MagicMock
from mockupdb import *

class DatabaseConnectorTest(unittest.TestCase):
    def setUp(self):
        self.database = 'testDB'
        self.dataset = [{'_id' : "abc123",
                         'outlook' : 'rainy',
                         'temp' : 'cold',
                         'humidity' : 'low',
                         'windy' : True,
                         'play' : 'no'}]

        self.collection = ['playtennis']
        
        def tearDown(self):
            pass

    def test_export_collection(self):
        with patch('pymongo.MongoClient') as mock_mongo:
            dc = DatabaseConnector()
            result = dc.export_collection(jsonfile=self.dataset, collection=self.collection[0], 
                                                    database=self.database)
            self.assertTrue(result)

    def test_check_collection(self):
        with patch('pymongo.MongoClient') as mock_mongo:
            dc = DatabaseConnector()
            dc.check_collection = MagicMock(return_value=self.collection)
            result = dc.check_collection(self.database)
            self.assertEqual(result, self.collection)
"""
    def test_get_collection(self):
        with patch('pymongo.MongoClient') as mock_mongo:
            db = DatabaseConnector()
            temp = db.get_collection(datafile=self.collection[0], database=self.database)
            result = isinstance(temp, object)
            self.assertTrue(result)
"""
if __name__ == "__main__":
    unittest.main()
