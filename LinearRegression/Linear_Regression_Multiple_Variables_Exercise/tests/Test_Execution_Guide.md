
# Instructions for Running Tests:

* Ensure you are in the correct working directory before running this test script.
* This test file is part of the Linear Regression Multiple Variables project.
* For reliable test execution, always run from the project root directory.

  

### Directory Structure:

Linear_Regression_Multiple_Variables_Exercise/  
├── data/  
├── notebooks/  
├── src/  
└── tests/

  

### Proper Test Execution:

1. Navigate to project root directory:

	cd path/to/Linear_Regression_Multiple_Variables_Exercise

  

2. Run test using the following command:

	python -m unittest tests/test_multivariate_regression_exercise.py

  

**Note:** *Running tests from other directories may cause import and path resolution issues.*



★ For example, if my project is located at:
Desktop\Machine_Learning\LinearRegression\Linear_Regression_Multiple_Variables_Exercise:  

- First: I navigate to this directory in terminal:

`cd Desktop\Machine_Learning\LinearRegression\Linear_Regression_Multiple_Variables_Exercise`

  
- Second: to run `test_multivariate_regression_exercise.py` we can use one of the following two commands:

`python -m unittest tests\test_multivariate_regression_exercise.py`

or:

`python -m unittest tests.test_multivariate_regression_exercise`
