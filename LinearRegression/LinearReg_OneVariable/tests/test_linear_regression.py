import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.linear_regression import train_model, predict_prices

class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        # Setup initial data
        self.df = pd.DataFrame({
            'area': [2600, 3000, 3200, 3600, 4000],
            'price': [550000, 565000, 610000, 680000, 725000]
        })
        self.areas = pd.DataFrame({
            'area': [1000, 1500, 2300, 3540, 4120]
        })

        # Train the model
        self.model = train_model(self.df)

    def test_train_model(self):
        # Check if the model is an instance of LinearRegression
        self.assertIsInstance(self.model, LinearRegression)

        # Check if the model has been trained correctly
        self.assertAlmostEqual(self.model.coef_[0], 135.78767123, places=5)
        self.assertAlmostEqual(self.model.intercept_, 180616.43835616, places=5)

    def test_predict_prices(self):
        # Predict prices for the given areas
        predicted_prices = predict_prices(self.model, self.areas)

        # Expected prices calculated manually
        expected_prices = [316404.10958904, 384297.94520547, 492928.0821917808, 661304.794520548, 740061.6438356165]

        # Check if the predicted prices match the expected prices
        for predicted, expected in zip(predicted_prices, expected_prices):
            self.assertAlmostEqual(predicted, expected, places=2)

    def test_prediction_output(self):
        # Predict prices for the given areas
        self.areas['prices'] = predict_prices(self.model, self.areas)

        # Check if the DataFrame has a new column 'prices'
        self.assertIn('prices', self.areas.columns)

        # Check if the number of rows in the output is correct
        self.assertEqual(len(self.areas), 5)

if __name__ == '__main__':
    unittest.main()