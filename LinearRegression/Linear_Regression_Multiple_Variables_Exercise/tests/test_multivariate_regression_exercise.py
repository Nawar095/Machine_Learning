"""
To run test:
For example, if my project is located at:
Desktop\Machine_Learning\LinearRegression\Linear_Regression_Multiple_Variables_Exercise,
First: I navigate to this directory in terminal:
    `cd Desktop\Machine_Learning\LinearRegression\Linear_Regression_Multiple_Variables_Exercise`

Second: to run `test_multivariate_regression_exercise.py` we can use one of the following two commands:
    `python -m unittest tests/test_multivariate_regression_exercise.py`
    or:
    `python -m unittest tests.test_multivariate_regression_exercise`

Note: Running tests from other directories may cause import and path resolution issues.
For more details about test execution, you can check Test_Execution_Guide.md.

"""

import unittest
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from word2number import w2n
from src.multivariate_regression_exercise import MultipleLinearRegression

class TestMultipleLinearRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test environment before any tests are run."""
        cls.data_path = "data"
        cls.mlr = MultipleLinearRegression(cls.data_path)
        cls.mlr.load_data()
        cls.mlr.train_model()

    def test_load_data(self):
        """Test if the data is loaded correctly."""
        self.assertIsInstance(self.mlr.data, pd.DataFrame)
        self.assertEqual(len(self.mlr.data), 8)

    def test_nan_to_zero(self):
        """Test if NaN values in the 'experience' column are replaced with 'zero'."""
        self.assertFalse(self.mlr.data['experience'].isnull().any())
        self.assertTrue((self.mlr.data['experience'] == 0).any())

    def test_str_to_nums(self):
        """Test if string values in the 'experience' column are converted to numbers."""
        self.assertTrue(all(isinstance(x, int) for x in self.mlr.data['experience']))

    def test_handle_missing_values(self):
        """Test if missing values in the 'test_score(out of 10)' column are filled with the mean."""
        self.assertFalse(self.mlr.data['test_score(out of 10)'].isnull().any())

    def test_train_model(self):
        """Test if the model is trained correctly."""
        self.assertIsInstance(self.mlr.model, LinearRegression)
        self.assertIsNotNone(self.mlr.model.coef_)
        self.assertIsNotNone(self.mlr.model.intercept_)

    def test_get_model_coefficients(self):
        """Test if the model coefficients and intercept are returned correctly."""
        coefficients, intercept = self.mlr.get_model_coefficients()
        self.assertIsInstance(coefficients, np.ndarray)
        self.assertIsInstance(intercept, float)

    def test_predict_salary(self):
        """Test if the salary prediction works correctly."""
        predicted_salary = self.mlr.predict_salary(2, 9, 6)
        self.assertIsInstance(predicted_salary, float)
        self.assertGreater(predicted_salary, 0)

        predicted_salary = self.mlr.predict_salary(12, 10, 10)
        self.assertIsInstance(predicted_salary, float)
        self.assertGreater(predicted_salary, 0)

if __name__ == "__main__":
    unittest.main()