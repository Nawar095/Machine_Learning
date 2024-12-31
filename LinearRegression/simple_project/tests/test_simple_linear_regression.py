import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class TestSimpleLinearRegression(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment.
        Load datasets and create a sample model for testing.
        """
        # Sample training data
        self.df = pd.DataFrame({
            'area': [2600, 3000, 3200, 3600, 4000],
            'price': [550000, 565000, 610000, 680000, 725000]
        })

        # Initialize and train the model
        self.model = LinearRegression()
        self.model.fit(self.df[['area']], self.df['price'])

    def test_model_coefficient(self):
        """
        Test if the model's coefficient is calculated correctly.
        """
        expected_coefficient = 135.78767123  # Replace with your expected value
        np.testing.assert_almost_equal(self.model.coef_[0], expected_coefficient, decimal=5)

    def test_model_intercept(self):
        """
        Test if the model's intercept is calculated correctly.
        """
        expected_intercept = 180616.43835616432  # Replace with your expected value
        np.testing.assert_almost_equal(self.model.intercept_, expected_intercept, decimal=5)

    def test_prediction(self):
        """
        Test if the model predicts values correctly.
        """
        test_area = [[3300]]
        expected_prediction = 628715.75342466  # Replace with your expected value
        prediction = self.model.predict(test_area)
        np.testing.assert_almost_equal(prediction[0], expected_prediction, decimal=5)

    def test_dataset_integrity(self):
        """
        Test if the dataset is loaded correctly and has no missing values.
        """
        self.assertFalse(self.df.isnull().values.any(), "Dataset contains null values")

    def test_prediction_output(self):
        """
        Test if the prediction output matches the expected format.
        """
        areas = pd.DataFrame({'area': [1000, 1500, 2300]})
        predictions = self.model.predict(areas)
        self.assertEqual(len(predictions), len(areas), "Number of predictions does not match input areas")

if __name__ == "__main__":
    unittest.main()
