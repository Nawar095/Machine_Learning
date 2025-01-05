import os
import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_file = os.path.join(self.data_dir, '..', 'data', 'canada_per_capita_income.csv')
        self.test_data_file = os.path.join(self.data_dir, '..','data', 'test_data.csv')
        self.data = pd.read_csv(self.data_file)
        self.test_data = pd.read_csv(self.test_data_file)

    def test_data_loading(self):
        self.assertIsInstance(self.data, pd.DataFrame)
        self.assertListEqual(list(self.data.columns), ['year', 'per_capita_income'])

    def test_model_training(self):
        X = self.data[['year']].values
        y = self.data['per_capita_income'].values
        model = LinearRegression()
        model.fit(X, y)
        self.assertIsNotNone(model.coef_[0])
        self.assertIsNotNone(model.intercept_)

    def test_model_predictions(self):
        X = self.data[['year']].values
        y = self.data['per_capita_income'].values
        model = LinearRegression()
        model.fit(X, y)
        test_years = self.test_data['year'].values.reshape(-1, 1)
        predictions = model.predict(test_years)
        expected_income = self.test_data['expected_income'].values
        np.testing.assert_allclose(predictions, expected_income, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
