import unittest
import numpy as np
from app.mlprogram.validation import CrossValidation as cv

class CrossValidationTest(unittest.TestCase):
    def setUp(self):
        self.features = np.array([['high', 'sunny', 'hot', 'false'],
                             ['high', 'sunny', 'hot', 'true'],
                             ['high', 'rainy', 'mild', 'false'],
                             ['high', 'overcast', 'hot', 'false'],
                             ['normal', 'rainy', 'cool', 'false'],
                             ['high', 'rainy', 'cool', 'true'],
                             ['normal', 'overcast', 'cool', 'true'],
                             ['high', 'sunny', 'mild', 'false'],
                             ['normal', 'sunny', 'cool', 'false'],
                             ['normal', 'rainy', 'mild', 'false'],
                             ['normal', 'sunny', 'mild', 'true'],
                             ['high', 'overcast', 'mild', 'true'],
                             ['normal', 'overcast', 'hot', 'false'],
                             ['high', 'rainy', 'mild', 'true']])

        self.target = np.array(['no', 'no', 'yes', 'yes', 
                                'yes', 'no', 'yes', 'no', 
                                'yes', 'yes','yes', 'yes', 
                                'yes', 'no'])

    def tearDown(self):
        pass
    
    def test_assign(self):
        size = 5
        k = 4
        partition_feature = [np.zeros([1,5]) for i in range(k)]
        result = cv.assign(partition_feature, size, k)
        self.assertLessEqual(result, k)

    def test_partition(self):
        k = 4
        target_reshape = self.target.reshape(len(self.target), 1)
        result = cv.partition(self.features, target_reshape, k)
        self.assertEqual(len(result[0]), k)

if __name__ == '__main__':
    unittest.main()
