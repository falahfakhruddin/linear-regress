import unittest

class AlgorithmTests(unittest.TestCase):
    def test_multiplication(self):
        temp = 4
        result = 2*2
        self.assertEqual(result, temp)
    
    def test_divided(self):
        temp = 2
        result = 5/2
        self.assertEqual(result, temp)

if __name__ == "__main__":
    unittest.main()
