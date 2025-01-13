<p align="center"><h1 align="center">Multiple Linear Regression - Home Price Prediction</h1></p>
<p align="center">
	<em>A machine learning project for predicting home prices using multiple features</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="python-version">
	<img src="https://img.shields.io/badge/scikit--learn-1.6.0-orange.svg" alt="sklearn-version">
  	<img src="https://img.shields.io/badge/matplotlib-3.8.0-yellow.svg" alt="pandas-version">
	<img src="https://img.shields.io/badge/numpy-1.26.4-green.svg" alt="numpy-version">
	<img src="https://img.shields.io/badge/pandas-2.2.2-red.svg" alt="pandas-version">

</p>
<br>

## 🔗 Table of Contents

- [📍 Overview](#-overview)
- [👾 Features](#-features)
- [🧮 Mathematical Background](#-mathematical-background)
- [📊 Dataset](#-dataset)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [📈 Results and Analysis](#-results-and-analysis)
- [🔰 Contributing](#-contributing)
- [🛠️ Troubleshooting](#troubleshooting)
- [📝 License](#-license)

## 📍 Overview

This project implements a multiple linear regression model to predict home prices in Monroe Township, New Jersey. The model utilizes three key features:
- Area (square feet)
- Number of bedrooms
- Age of the home (years)

The implementation includes both object-oriented and procedural approaches, along with detailed visualization and analysis in Jupyter notebooks.

## 👾 Features

- **Data Preprocessing**: Handles missing values and prepares the dataset for analysis.
- **Data Visualization**: Visualizes relationships between features and the target variable.
- **Model Training**: Implements a linear regression model using scikit-learn.
- **Price Prediction**: Predicts house prices based on input features.
- **Interactive Notebook**: Provides a Jupyter notebook for detailed analysis and visualization.

## 🧮 Mathematical Background

The price prediction is based on the multiple linear regression equation:
```
Price = m₁ * area + m₂ * bedrooms + m₃ * age + b

where:
m₁ = 112.06 (area coefficient)
m₂ = 23388.88 (bedrooms coefficient)
m₃ = -3231.72 (age coefficient)
b = 221323.00 (intercept)
```

## 📊 Dataset

The project uses `homeprices.csv` containing real estate data with the following structure:

| Feature  | Description | Type |
|----------|-------------|------|
| area     | House area in square feet | float |
| bedrooms | Number of bedrooms | int |
| age      | Age of house in years | int |
| price    | House price (target variable) | float |

## 📁 Project Structure

```sh
└── Linear_Regression_Multiple_Variables/
    ├── data/
    │   └── homeprices.csv          # Dataset file
    ├── img/
    │   ├── equation.jpg            # Mathematical equation visualization
    |   ├── general_equation.jpg    # General mathematical equation
    |   ├── mean.png                # Mean formula
    |   └── median.png              # Median formula
    ├── notebooks/
    │   ├── multivariate_regression.ipynb  # Interactive analysis notebook
    ├── src/
    │   ├── __init__.py          # mark directory as Python package directory
    │   ├── multivariate_regression.py      # OOP implementation
    │   └── multiple_linear_regression.py   # Procedural implementation
    └── tests/                      # Unit tests
        ├── Test_Execution_Guide.md          # mark
        ├── test_multiple_linear_regression.py  # unit test for OOP implementation
        └── test_multivariate_regression.py     # unit test for Procedural implementation

```

## 🚀 Getting Started

### ☑️ Prerequisites
- Python 3.10 or higher
- pip package manager

### ⚙️ Installation

1. **Clone the Repository**:
```sh
git clone https://github.com/Nawar095/Machine_Learning.git
cd Linear_Regression_Multiple_Variables
```

2. **Install Dependencies**:
```sh
pip install -r requirements.txt
```

### 🤖 Usage

#### Using Python Script
```python
from src.multivariate_regression import MultipleLinearRegression

# Initialize and train model
model = MultipleLinearRegression("data")
model.load_data()
model.train_model()

# Make predictions
price = model.predict_price(area=3000, bedrooms=3, age=40)
```

#### Using Jupyter Notebook
```sh
jupyter notebook notebooks/multivariate_regression.ipynb
```

## 📈 Results and Analysis

The model demonstrates strong predictive capabilities:

### Sample Predictions:
1. House Specifications:
   - Area: 3000 sq ft
   - Bedrooms: 3
   - Age: 40 years
   - **Predicted Price**: $498,408.25

2. House Specifications:
   - Area: 2500 sq ft
   - Bedrooms: 4
   - Age: 5 years
   - **Predicted Price**: $578,876.03

### Model Coefficients:
- Area (m₁): 112.06 \$ per sq ft   
- Bedrooms (m₂): 23,388.88 \$ per bedroom
- Age (m₃): -3,231.72 \$ per year
- Base Price (b): $221,323.00

## 🔰 Contributing

Contributions are welcome! Please follow these steps:

<details closed>
<summary>Contributing Guidelines</summary>

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Submit a pull request
</details>

## 🛠️ Troubleshooting

1. **Dataset Not Found:**
   Ensure `homeprices.csv` is in the `data/` folder or update the path in the scripts.
2. **Missing Libraries:**
   Run `pip install -r requirements.txt` to install dependencies.
3. **Prediction Errors:**
   Ensure input features are within the training dataset's range.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Nawar095/Machine_Learning/blob/main/LICENSE) file for details.

---