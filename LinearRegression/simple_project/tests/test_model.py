import os
import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression

class TestLinearRegressionModel(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.homeprices_file = os.path.join(self.data_dir, 'homeprices.csv')
        self.homeprices_df = pd.read_csv(self.homeprices_file)

    def test_model_training(self):
        # Test that the linear regression model can be trained on the homeprices data
        X = self.homeprices_df['area'].values.reshape(-1, 1)
        y = self.homeprices_df['price'].values
        model = LinearRegression()
        model.fit(X, y)
        self.assertIsNotNone(model.coef_)
        self.assertIsNotNone(model.intercept_)

    def test_model_prediction(self):
        # Test that the linear regression model can make predictions on new data
        X = self.homeprices_df['area'].values.reshape(-1, 1)
        y = self.homeprices_df['price'].values
        model = LinearRegression()
        model.fit(X, y)
        new_area = [[3300]]
        predicted_price = model.predict(new_area)
        self.assertAlmostEqual(predicted_price[0], 628715.75342466, places=2)

if __name__ == '__main__':
    unittest.main()