import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from word2number import w2n

class MultipleLinearRegression:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.model = None

    def load_data(self):
        """Load the dataset from the specified path."""
        data_file = os.path.join(self.data_path, "hiring.csv")
        self.data = pd.read_csv(data_file)
        self.nan_to_zero()
        self.str_to_nums()
        self.handle_missing_values()

    def nan_to_zero(self):
        """Replace NaN values with zero in the experience column."""
        self.data['experience'] = self.data['experience'].fillna("zero")

    def str_to_nums(self):
        """Convert string values to numbers in the experience column."""
        self.data['experience'] = self.data['experience'].apply(w2n.word_to_num)


    def handle_missing_values(self):
        """Handle missing values in the "test_score(out of 10)" column by filling with the arithmetic mean."""
        self.data['test_score(out of 10)'] = self.data['test_score(out of 10)'].fillna(int(self.data['test_score(out of 10)'].mean()))


    def train_model(self):
        """Train the linear regression model on the dataset."""
        X = self.data[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']].values
        y = self.data['salary($)']
        self.model = LinearRegression()
        self.model.fit(X, y)

    def get_model_coefficients(self):
        """Get the coefficients and intercept of the trained model."""
        coefficients = self.model.coef_
        intercept = self.model.intercept_
        return coefficients, intercept

    def predict_salary(self, experience, test_score, interview_score):
        """Predict the candidate salary based on the given experience, test_score, and interview_score."""
        input_data = np.array([experience, test_score, interview_score]).reshape(1, -1)
        predicted_salary = self.model.predict(input_data)
        return predicted_salary[0]

if __name__ == "__main__":
    data_path = "data"
    mlr = MultipleLinearRegression(data_path)
    mlr.load_data()
    mlr.train_model()
    coefficients, intercept = mlr.get_model_coefficients()
    print("Coefficients:", coefficients)
    print("Intercept:", intercept)

    # Predict salary for new candidates
    candidate1_salary = mlr.predict_salary(2, 9, 6)
    print(f"Predict salary for candidate with 2 yr of experience, 9 test score, 6 interview score: {candidate1_salary} $")

    candidate2_salary = mlr.predict_salary(12, 10, 10)
    print(f"Predict salary for candidate with 12 yr of experience, 10 test score, 10 interview score: {candidate2_salary} $")