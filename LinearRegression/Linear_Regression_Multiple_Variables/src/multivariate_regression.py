"""
To run multivariate_regression.py:
For example, if my project is located at:
Desktop\Machine_Learning\LinearRegression\Linear_Regression_Multiple_Variables,
First: I navigate to this directory in terminal:
    `cd Desktop\Machine_Learning\LinearRegression\Linear_Regression_Multiple_Variables`

Second: to run `multivariate_regression.py` we can use the following command:
    `python src/multivariate_regression.py`

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class MultipleLinearRegression:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.model = None

    def load_data(self):
        """Load the dataset from the specified path."""
        data_file = os.path.join(self.data_path, "homeprices.csv")
        self.data = pd.read_csv(data_file)
        self.handle_missing_values()

    def handle_missing_values(self):
        """Handle missing values in the dataset by filling with the median."""
        self.data['bedrooms'] = self.data['bedrooms'].fillna(self.data['bedrooms'].median())

    def visualize_data(self):
        """Visualize the relationship between independent variables and the target variable."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for i, feature in enumerate(['area', 'bedrooms', 'age']):
            axes[i].scatter(self.data[feature], self.data['price'], label="Data Points")
            axes[i].plot(self.data[feature], self.data['price'], color='green', label="Line of Best Fit")
            axes[i].set_title(f'{feature.capitalize()} vs Price')
            axes[i].set_xlabel(feature.capitalize())
            axes[i].set_ylabel('Price ($)')

        plt.tight_layout()
        plt.show()

    def train_model(self):
        """Train the linear regression model on the dataset."""
        X = self.data[['area', 'bedrooms', 'age']].values
        y = self.data['price']
        self.model = LinearRegression()
        self.model.fit(X, y)

    def get_model_coefficients(self):
        """Get the coefficients and intercept of the trained model."""
        coefficients = self.model.coef_
        intercept = self.model.intercept_
        return coefficients, intercept

    def predict_price(self, area, bedrooms, age):
        """Predict the price of a home based on the given area, bedrooms, and age."""
        input_data = np.array([area, bedrooms, age]).reshape(1, -1)
        predicted_price = self.model.predict(input_data)
        return predicted_price[0]

if __name__ == "__main__":
    data_path = "data"
    mlr = MultipleLinearRegression(data_path)
    mlr.load_data()
    mlr.visualize_data()
    mlr.train_model()
    coefficients, intercept = mlr.get_model_coefficients()
    print("Coefficients:", coefficients)
    print("Intercept:", intercept)

    # Predict prices for new homes
    price1 = mlr.predict_price(3000, 3, 40)
    print("Price of home with 3000 sqr ft area, 3 bedrooms, 40 years old:", price1)

    price2 = mlr.predict_price(2500, 4, 5)
    print("Price of home with 2500 sqr ft area, 4 bedrooms, 5 years old:", price2)