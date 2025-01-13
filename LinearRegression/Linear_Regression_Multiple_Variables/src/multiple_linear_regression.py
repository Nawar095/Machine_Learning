"""
To run multiple_linear_regression.py:
For example, if my project is located at:
Desktop\Machine_Learning\LinearRegression\Linear_Regression_Multiple_Variables,
First: I navigate to this directory in terminal:
    `cd Desktop\Machine_Learning\LinearRegression\Linear_Regression_Multiple_Variables`

Second: to run `multiple_linear_regression.py` we can use the following command:
    `python src/multiple_linear_regression.py`

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define the function to load and preprocess the dataset
def load_and_preprocess_data(filepath):
    """
    Load and preprocess the dataset.

    Parameters:
    filepath (str): The path to the CSV file containing the dataset.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    df = pd.read_csv(filepath)
    df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())
    return df

# Define the function to visualize the dataset
def visualize_data(df):
    """
    Visualize the relationship between independent variables and the target variable.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the dataset.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(df['area'], df['price'], color='blue', label="Data Points")
    axes[0].plot(df['area'], df['price'], color='green', label="Line of Best Fit")
    axes[0].set_title('Area vs Price')
    axes[0].set_xlabel('Area (sq ft)')
    axes[0].set_ylabel('Price ($)')

    axes[1].scatter(df['bedrooms'], df['price'], color='green', label="Data Points")
    axes[1].plot(df['bedrooms'], df['price'], color='blue', label="Line of Best Fit")
    axes[1].set_title('Bedrooms vs Price')
    axes[1].set_xlabel('Bedrooms')
    axes[1].set_ylabel('Price ($)')

    axes[2].scatter(df['age'], df['price'], color='red', label="Data Points")
    axes[2].plot(df['age'], df['price'], color='blue', label="Line of Best Fit")
    axes[2].set_title('Age vs Price')
    axes[2].set_xlabel('Age (years)')
    axes[2].set_ylabel('Price ($)')

    plt.tight_layout()
    plt.show()

# Define the function to train the model
def train_model(df):
    """
    Train the linear regression model.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
    LinearRegression: The trained linear regression model.
    """
    X = df[['area', 'bedrooms', 'age']].values
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)
    return model

# Define the function to make predictions
def predict_price(model, area, bedrooms, age):
    """
    Predict the price of a home given its area, number of bedrooms, and age.

    Parameters:
    model (LinearRegression): The trained linear regression model.
    area (float): The area of the home in square feet.
    bedrooms (int): The number of bedrooms in the home.
    age (int): The age of the home in years.

    Returns:
    float: The predicted price of the home.
    """
    return model.predict([[area, bedrooms, age]])[0]

# Main execution
if __name__ == "__main__":
    # Load and preprocess the dataset
    df = load_and_preprocess_data("data/homeprices.csv")

    # Visualize the dataset
    visualize_data(df)

    # Train the model
    model = train_model(df)

    # Display the model coefficients and intercept
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")

    # Predict prices for new data
    price1 = predict_price(model, 3000, 3, 40)
    print(f"Predicted price for a 3000 sq ft, 3 bedroom, 40 year old home: ${price1}")

    price2 = predict_price(model, 2500, 4, 5)
    print(f"Predicted price for a 2500 sq ft, 4 bedroom, 5 year old home: ${price2}")

    # Example of manual prediction calculation
    manual_prediction = 2500 * model.coef_[0] + 4 * model.coef_[1] + 5 * model.coef_[2] + model.intercept_
    print(f"Manual prediction for a 2500 sq ft, 4 bedroom, 5 year old home: ${manual_prediction}")