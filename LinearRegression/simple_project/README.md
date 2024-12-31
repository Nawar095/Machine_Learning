# Simple Linear Regression - Predicting Home Prices

## Overview
This project implements a simple linear regression algorithm to predict home prices based on the area. It includes training a linear regression model using `homeprices.csv` and predicting house prices for areas provided in `areas.csv`. The results are saved in a `prediction.csv` file.

## Project Structure
- **data/**: Contains the input datasets (`homeprices.csv` and `areas.csv`) and the output prediction dataset (`prediction.csv`).
- **notebooks/**: Contains the Jupyter notebook (`SimpleLinearRegression.ipynb`) used for implementing the linear regression model.
- **src/**: Python scripts for the model (if later modularized).
- **tests/**: Unit tests (to be added for validating the model and dataset processing).
- **requirements.txt**: Lists the dependencies required for the project.
- **README.md**: Detailed description of the project.

## Data Description
### `homeprices.csv`
- **area**: The size of the house (in square feet).
- **price**: The price of the house (in US dollars).

### `areas.csv`
- **area**: Areas for which house prices need to be predicted.

### `prediction.csv`
- **area**: Input areas.
- **prices**: Predicted house prices based on the linear regression model.

## Steps to Run the Project
1. **Install Dependencies**:
   Ensure you have Python installed, then install the required libraries:
   ```bash
    pip install -r requirements.txt
    ```

2. **Run the Jupyter Notebook**:
   Open `SimpleLinearRegression.ipynb` in Jupyter Notebook and run the cells sequentially.

3. **Output**:
   - The notebook trains the linear regression model using homeprices.csv.
   - Predicts prices for areas in areas.csv.
   - Saves the results in prediction.csv.
   - Displays a scatter plot visualizing the predictions.

## Usage

The Jupyter Notebook `SimpleLinearRegression.ipynb` contains the step-by-step implementation of the linear regression model, including:

- Loading and exploring the dataset
- Visualizing the data
- Training the linear regression model
- Making predictions on new data
- Evaluating the model's performance
- Generating a CSV file with the predicted prices

You can modify the notebook cells as needed or use the provided code as a starting point for your own projects.