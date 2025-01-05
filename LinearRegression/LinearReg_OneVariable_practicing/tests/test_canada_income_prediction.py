# """
# Unit tests for the Canadian Per Capita Income Prediction Model
# This module contains test cases to validate the model's functionality,
# data processing, and prediction accuracy.
# run the test in terminal: python -m unittest -v test_canada_income_prediction.py
# """

import unittest
import pandas as pd
import numpy as np
from sklearn import linear_model
import os
from src.canada_income_prediction import (
    load_and_display_data,
    train_model,
    calculate_income,
    create_test_dataset,
    get_model_parameters
)

class TestCanadaIncomePrediction(unittest.TestCase):
    """Test cases for the Canada Income Prediction model."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        # Create a sample dataset for testing
        cls.test_df = pd.DataFrame({
            'year': range(1970, 1975),
            'per_capita_income': [3399.29, 3768.29, 4251.17, 4804.46, 5576.51]
        })
        
        # Save test dataset temporarily
        cls.test_df.to_csv('test_canada_per_capita_income.csv', index=False)
        
        # Train model with test data
        cls.model = train_model(cls.test_df)

    def test_data_loading(self):
        """Test if the data loading function works correctly."""
        df = load_and_display_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ['year', 'per_capita_income'])
        self.assertTrue(len(df) > 0)

    def test_model_training(self):
        """Test if the model training produces valid results."""
        reg = train_model(self.test_df)
        self.assertIsInstance(reg, linear_model.LinearRegression)
        self.assertTrue(hasattr(reg, 'coef_'))
        self.assertTrue(hasattr(reg, 'intercept_'))

    def test_model_parameters(self):
        """Test if model parameters are extracted correctly."""
        coefficient, intercept = get_model_parameters(self.model)
        self.assertIsInstance(coefficient, float)
        self.assertIsInstance(intercept, float)
        self.assertTrue(coefficient != 0)  # Ensure we have a non-zero slope

    def test_income_calculation(self):
        """Test if income calculation works correctly."""
        coefficient, intercept = get_model_parameters(self.model)
        test_year = 2020
        income = calculate_income(coefficient, intercept, test_year)
        self.assertIsInstance(income, float)
        self.assertTrue(income > 0)  # Income should be positive

    def test_test_dataset_creation(self):
        """Test if test dataset is created correctly."""
        coefficient, intercept = get_model_parameters(self.model)
        future_years = [2020, 2021, 2022]
        test_data, exp_income = create_test_dataset(coefficient, intercept, future_years)
        
        # Test DataFrame creation
        self.assertIsInstance(test_data, pd.DataFrame)
        self.assertEqual(len(test_data), len(future_years))
        self.assertEqual(list(test_data.columns), ['years', 'expected_per_capita_income'])
        
        # Test if CSV is created
        self.assertTrue(os.path.exists('test_data.csv'))
        
        # Test expected income calculations
        self.assertEqual(len(exp_income), len(future_years))
        self.assertTrue(all(isinstance(x, float) for x in exp_income))

    def test_model_predictions(self):
        """Test if model predictions are consistent."""
        coefficient, intercept = get_model_parameters(self.model)
        test_year = 2020
        
        # Calculate income using our function
        expected_income = calculate_income(coefficient, intercept, test_year)
        
        # Calculate income using direct model prediction
        predicted_income = self.model.predict([[test_year]])[0]
        
        # Test if both calculations match (within floating-point precision)
        self.assertTrue(np.isclose(expected_income, predicted_income))

    def test_data_validation(self):
        """Test data validation and error handling."""
        df = load_and_display_data()
        
        # Test for required columns
        self.assertIn('year', df.columns)
        self.assertIn('per_capita_income', df.columns)
        
        # Test for data types
        self.assertEqual(df['year'].dtype, np.int64)
        self.assertTrue(np.issubdtype(df['per_capita_income'].dtype, np.number))
        
        # Test for missing values
        self.assertTrue(df['year'].notna().all())
        self.assertTrue(df['per_capita_income'].notna().all())

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after running tests."""
        # Remove temporary test files
        if os.path.exists('test_canada_per_capita_income.csv'):
            os.remove('test_canada_per_capita_income.csv')
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')

if __name__ == '__main__':
    unittest.main(verbosity=2)