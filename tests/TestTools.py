import unittest
import pandas as pd
from app.mlprogram import tools as tl

class ToolsTest(unittest.TestCase):
    def setUp(self):
        dataset = [{'outlook' : 'outcast',
                    'temp' : 'hot',
                    'humidity' : 'high',
                    'windy' : True,
                    'play' : 1},
                   {'outlook' : 'sunny',
                    'temp' : 'hot',
                    'humidity' : 'high',
                    'windy' : False,
                    'play' : 1},
                   {'outlook' : 'rainy',
                    'temp' : 'cold',
                    'humidity' : 'low',
                    'windy' : True,
                    'play' : 0}]

        self.dataframe = pd.DataFrame(dataset)
        self.label = 'play'

    def tearDown(self):
        pass

    def test_transform_dataframe_json(self):
        temp = tl.transform_dataframe_json(self.dataframe)
        result = isinstance(temp, object)
        self.assertTrue(result)

    def test_dataframe_extraction_case1(self):
        values = len(list(self.dataframe))-1#label doesnt count
        
        #dummies True
        temp = tl.dataframe_extraction(self.dataframe, self.label, dummies=True)
        features = temp[0]
        result = len(features[0])
        self.assertGreater(result, values)

    def test_dataframe_extraction_case2(self):
        values = len(list(self.dataframe))-1#label doesnt count
        
        #dummies False
        temp = tl.dataframe_extraction(self.dataframe, self.label, dummies=False)
        features = temp[0]
        result = len(features[0])
        self.assertEqual(result, values)

    def test_dataframe_extraction_case3(self):
        #type classification
        temp = tl.dataframe_extraction(self.dataframe, self.label, type='classification')
        target = temp[1]
        result = isinstance(target[0], str)
        self.assertTrue(result)

    def test_dataframe_extraction_case4(self):
        #type regression
        temp = tl.dataframe_extraction(self.dataframe, self.label, type='regression')
        target = temp[1]
        result = isinstance(target[0], float)
        self.assertTrue(result)


if __name__=='__main__':
    unittest.main()
