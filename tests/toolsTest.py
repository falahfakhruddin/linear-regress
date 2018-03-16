import unittest
import pandas as pd
from app.mlprogram import tools as tl

class ToolsTest(unittest.TestCase):
    def setUp(self):
        dataset = [{'outlook' : 'outcast',
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
        self.dataframe = pd.DataFrame(dataset)
    
    def tearDown(self):
        pass

    def test_transform_dataframe_json(self):
        temp = tl.transform_dataframe_json(self.dataframe)
        result = isinstance(temp, object)
        self.assertTrue(result)

if __name__=='__main__':
    unittest.main()
