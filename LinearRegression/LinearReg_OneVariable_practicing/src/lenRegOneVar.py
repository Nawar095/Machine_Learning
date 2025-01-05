import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("../data/canada_per_capita_income.csv")

# Visualize the data
plt.figure(figsize=(10, 6))
plt.xlabel("Year", fontsize=14)
plt.ylabel("Per Capita Income (US $)", fontsize=14)
plt.scatter(data.year, data.per_capita_income, color="blue", label="Data Points")
plt.plot(data.year, data.per_capita_income, color="green", label="Line of Best Fit")
plt.legend()
plt.show()

# Create and train the linear regression model
X = data[['year']].values
y = data['per_capita_income'].values
model = LinearRegression()
model.fit(X, y)

# Get the model coefficients
coefficient = model.coef_[0]
intercept = model.intercept_

print(f"Coefficient: {coefficient:.2f}")
print(f"Intercept: {intercept:.2f}")


def calculate_income(year, coefficient, intercept):
    """
    Calculate the predicted per capita income for a given year using the linear regression model.

    Args:
        year (int): The year for which to calculate the per capita income.
        coefficient (float): The coefficient of the linear regression model.
        intercept (float): The intercept of the linear regression model.

    Returns:
        float: The predicted per capita income for the given year.
    """
    return coefficient * year + intercept


# Generate test data
test_years = [2018, 2019, 2020, 2021, 2025, 2030]
expected_income = [calculate_income(year, coefficient, intercept) for year in test_years]

# Create a DataFrame for the test data
test_data = pd.DataFrame({"year": test_years, "expected_income": expected_income})

# Save the test data to a CSV file
test_data.to_csv("../data/test_data.csv", index=False)

# Make predictions using the model
model_predictions = model.predict(np.array(test_years).reshape(-1, 1))

# Compare the model predictions with the expected income
print("\nComparison of Model Predictions and Expected Income:")
for i, (predicted, expected) in enumerate(zip(model_predictions, expected_income), start=1):
    if np.isclose(predicted, expected):
        print(f"Test {i}: The predicted value matches the expected value --> Pass")
    else:
        print(f"Test {i}: The predicted value does not match the expected value --> Fail")
