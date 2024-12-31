import os
import unittest
import pandas as pd

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.homeprices_file = os.path.join(self.data_dir, 'homeprices.csv')
        self.areas_file = os.path.join(self.data_dir, 'areas.csv')

    def test_homeprices_data(self):
        # Test that the homeprices.csv file exists and has the expected columns
        self.assertTrue(os.path.exists(self.homeprices_file))
        homeprices_df = pd.read_csv(self.homeprices_file)
        self.assertListEqual(list(homeprices_df.columns), ['area', 'price'])

    def test_areas_data(self):
        # Test that the areas.csv file exists and has the expected column
        self.assertTrue(os.path.exists(self.areas_file))
        areas_df = pd.read_csv(self.areas_file)
        self.assertListEqual(list(areas_df.columns), ['area'])

if __name__ == '__main__':
    unittest.main()