#!/usr/bin/env python3

"""
Linear Regression Model for Canadian Per Capita Income Prediction
This script implements a linear regression model to predict Canadian per capita income
based on historical data and makes future predictions.
"""
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def load_and_display_data():
    """Load the dataset and display first few rows"""
    df = pd.read_csv('canada_per_capita_income.csv')
    print("First few rows of the dataset:")
    print(df.head())
    return df

def plot_income_distribution(df):
    """Visualize the relationship between year and per capita income"""
    plt.figure(figsize=(10, 6))
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Per Capita Income (US $)", fontsize=14)
    plt.scatter(df.year, df.per_capita_income, color="blue", label="Data Points")
    plt.plot(df.year, df.per_capita_income, color="green", label="Line of Best Fit")
    plt.legend()
    plt.show()

def train_model(df):
    """Train the linear regression model"""
    reg = linear_model.LinearRegression()
    reg.fit(df[['year']], df.per_capita_income)
    return reg

def get_model_parameters(reg):
    """Extract and display model parameters"""
    coefficient = reg.coef_[0]
    intercept = reg.intercept_
    print(f"Model coefficient: {coefficient:.2f}")
    print(f"Model intercept: {intercept:.2f}")
    return coefficient, intercept

def calculate_income(coefficient, intercept, year):
    """Calculate expected income for a given year using the model parameters"""
    return coefficient * year + intercept

def create_test_dataset(coefficient, intercept, years):
    """Create a test dataset for future years"""
    exp_income = [calculate_income(coefficient, intercept, year) for year in years]
    
    # Create DataFrame with predictions
    test_data = pd.DataFrame({
        'years': years,
        'expected_per_capita_income': exp_income
    })
    
    # Save predictions to CSV
    test_data.to_csv("test_data.csv", index=False)
    return test_data, exp_income

def validate_predictions(reg, years, exp_income):
    """Validate model predictions against calculated expectations"""
    years_array = np.array(years).reshape(-1, 1)
    predicted_income = reg.predict(years_array)
    
    print("\nValidating predictions:")
    for i, (pred, exp) in enumerate(zip(predicted_income, exp_income)):
        result = "Pass" if np.isclose(pred, exp) else "Fail"
        print(f"Test {i + 1}: Year {years[i]} - {result}")
        print(f"  Predicted: {pred:.2f}")
        print(f"  Expected:  {exp:.2f}")

def main():
    """Main function to orchestrate the analysis"""
    # Load and visualize data
    df = load_and_display_data()
    plot_income_distribution(df)
    
    # Train model and get parameters
    reg = train_model(df)
    coefficient, intercept = get_model_parameters(reg)
    
    # Create and validate predictions
    future_years = [2018, 2019, 2020, 2021, 2025, 2030]
    test_data, exp_income = create_test_dataset(coefficient, intercept, future_years)
    
    print("\nTest dataset preview:")
    print(test_data)
    
    validate_predictions(reg, future_years, exp_income)

if __name__ == "__main__":
    main() 