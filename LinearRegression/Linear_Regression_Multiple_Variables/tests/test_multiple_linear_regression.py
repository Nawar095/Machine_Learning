"""
To run test:
For example, if my project is located at:
Desktop\Machine_Learning\LinearRegression\Linear_Regression_Multiple_Variables,
First: I navigate to this directory in terminal:
    `cd Desktop\Machine_Learning\LinearRegression\Linear_Regression_Multiple_Variables`

Second: to run `test_multiple_linear_regression.py` we can use one of the following two commands:
    `python -m unittest tests/test_multiple_linear_regression.py`
    or:
    `python -m unittest tests.test_multiple_linear_regression`

Note: Running tests from other directories may cause import and path resolution issues.
For more details about test execution, you can check Test_Execution_Guide.md
"""
import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.multiple_linear_regression import load_and_preprocess_data, train_model, predict_price

class TestMultipleLinearRegression(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset for testing
        self.sample_data = pd.DataFrame({
            'area': [1500, 2000, 2500, 3000, 3500],
            'bedrooms': [3, 4, 3, 3, 5],
            'age': [10, 20, 15, 30, 5],
            'price': [300000, 400000, 350000, 500000, 600000]
        })

        # Save the sample data to a CSV file
        self.sample_data.to_csv("data/sample_homeprices.csv", index=False)

        # Load and preprocess the sample dataset
        self.df = load_and_preprocess_data("data/sample_homeprices.csv")

        # Train the model on the sample dataset
        self.model = train_model(self.df)

    def test_load_and_preprocess_data(self):
        # Test if the data is loaded correctly
        self.assertEqual(len(self.df), 5)
        self.assertListEqual(list(self.df.columns), ['area', 'bedrooms', 'age', 'price'])

        # Test if missing values are filled correctly
        self.df_with_nan = self.sample_data.copy()
        self.df_with_nan.loc[2, 'bedrooms'] = np.nan
        self.df_with_nan.to_csv("data/sample_homeprices_with_nan.csv", index=False)
        df_processed = load_and_preprocess_data("data/sample_homeprices_with_nan.csv")
        self.assertEqual(df_processed.bedrooms[2], 3.5)  # Median value

    def test_train_model(self):
        # Test if the model is an instance of LinearRegression
        self.assertIsInstance(self.model, LinearRegression)

        # Test if the model coefficients and intercept are not None
        self.assertIsNotNone(self.model.coef_)
        self.assertIsNotNone(self.model.intercept_)

    def test_predict_price(self):
        # Test the prediction functionality
        predicted_price = predict_price(self.model, 3000, 3, 40)
        self.assertIsInstance(predicted_price, float)

    def test_model_performance(self):
        # Test the performance metrics of the model
        X = self.df[['area', 'bedrooms', 'age']].values
        y = self.df['price']
        y_pred = self.model.predict(X)

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        self.assertLess(mse, 1e10)  # Ensure MSE is below a certain threshold
        self.assertGreater(r2, 0.8)  # Ensure R^2 score is above a certain threshold

    def test_manual_prediction(self):
        # Test manual prediction calculation
        area, bedrooms, age = 2500, 4, 5
        manual_prediction = area * self.model.coef_[0] + bedrooms * self.model.coef_[1] + age * self.model.coef_[2] + self.model.intercept_
        model_prediction = predict_price(self.model, area, bedrooms, age)
        self.assertAlmostEqual(manual_prediction, model_prediction, places=2)

if __name__ == '__main__':
    unittest.main()