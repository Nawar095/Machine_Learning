import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def train_model(df):
    reg = linear_model.LinearRegression()
    reg.fit(df[['area']], df.price)
    return reg

def predict_prices(model, areas):
    return model.predict(areas[['area']])

if __name__ == "__main__":
    # Load datasets
    df = pd.read_csv("data\homeprices.csv")
    areas = pd.read_csv("data\areas.csv")

    # Data Visualization
    plt.xlabel("area (sqr ft)")
    plt.ylabel("price (US $)")
    plt.scatter(df.area, df.price, color='green', marker='+')
    plt.show()

    # Train the model
    model = train_model(df)

    # Predict prices for new areas
    prices = predict_prices(model, areas)
    areas['prices'] = prices

    # Save the predictions to a new CSV file
    areas.to_csv("data\prediction.csv", index=False)

    # Load and display the predictions
    pr = pd.read_csv("data\prediction.csv")
    print(pr)

    # Visualization of predicted prices
    plt.xlabel('area')
    plt.ylabel('price')
    plt.scatter(pr.area, pr.prices, color='red')
    plt.show()