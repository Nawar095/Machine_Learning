import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load the training dataset
df = pd.read_csv("data/homeprices.csv")

# Visualize the data
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (US $)")
plt.scatter(df.area, df.price, color='green', marker='+')
plt.show()

#Create a linear regression object
reg = linear_model.LinearRegression()
#fitting my data (training the linear regression model using the available data points)
reg.fit(df[['area']], df.price)

# Predict the price for a given area
predicted_value = reg.predict([[3300]])
print(f"Predicted price for 3300 sq ft: {predicted_value[0]}")

# Display the coefficient and intercept of the linear regression model
coefficient = reg.coef_[0]
intercept = reg.intercept_
print(f"Coefficient (m): {coefficient}")
print(f"Intercept (b): {intercept}")

# How linear regression model works under the hood:
# the model predict the value by following the formula below:
# formula: y = mx + b
# price = coefficient * area + intercept
# Verify prediction formula: y = mx + b
price_formula = coefficient * 3300 + intercept
print(f"Price using formula: {price_formula}")
print(f"Price matches prediction: {np.isclose(price_formula, predicted_value)}")

# CSV file contain areas to predict the prices:
areas = pd.read_csv("areas.csv")
prices = reg.predict(areas)

# Add predicted prices to the dataframe and save to a new CSV file
areas['prices'] = prices
areas.to_csv("data\prediction.csv", index=False)
print("Predicted prices saved to 'data\prediction.csv'.")

# Load and visualize the predictions
predicted_df = pd.read_csv("data\prediction.csv")
plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(predicted_df.area, predicted_df.prices, color='red')
plt.show()