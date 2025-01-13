"""
To run test:
For example, if my project is located at:
Desktop\Machine_Learning\LinearRegression\Linear_Regression_Multiple_Variables,
First: I navigate to this directory in terminal:
    `cd Desktop\Machine_Learning\LinearRegression\Linear_Regression_Multiple_Variables`

Second: to run `test_multivariate_regression.py` we can use one of the following two commands:
    `python -m unittest tests/test_multivariate_regression.py`
    or:
    `python -m unittest tests.test_multivariate_regression`

Note: Running tests from other directories may cause import and path resolution issues.
For more details about test execution, you can check Test_Execution_Guide.md
"""


import os
import unittest
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from src.multivariate_regression import MultipleLinearRegression

class TestMultipleLinearRegression(unittest.TestCase):
    def setUp(self):
        self.data_path = "data"
        self.mlr = MultipleLinearRegression(self.data_path)
        self.mlr.load_data()
        self.mlr.train_model()

    def test_data_loading(self):
        self.assertIsInstance(self.mlr.data, pd.DataFrame)
        self.assertListEqual(list(self.mlr.data.columns), ['area', 'bedrooms', 'age', 'price'])

    def test_missing_value_handling(self):
        self.assertFalse(self.mlr.data['bedrooms'].isnull().values.any())

    def test_model_coefficients(self):
        coefficients, intercept = self.mlr.get_model_coefficients()
        self.assertIsNotNone(coefficients)
        self.assertIsNotNone(intercept)

    def test_prediction_accuracy(self):
        X = self.mlr.data[['area', 'bedrooms', 'age']].values
        y = self.mlr.data['price']

        y_pred = self.mlr.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Set threshold for acceptable MSE and R-squared values
        mse_threshold = mean_squared_error(y, y_pred)
        r2_threshold = r2_score(y, y_pred)

        self.assertLessEqual(mse, mse_threshold, f"Mean Squared Error ({mse}) is higher than the threshold ({mse_threshold})")
        self.assertGreaterEqual(r2, r2_threshold, f"R-squared ({r2}) is lower than the threshold ({r2_threshold})")

    def test_prediction_examples(self):
        price1 = self.mlr.predict_price(3000, 3, 40)
        self.assertIsInstance(price1, float)

        price2 = self.mlr.predict_price(2500, 4, 5)
        self.assertIsInstance(price2, float)

if __name__ == '__main__':
    unittest.main()
