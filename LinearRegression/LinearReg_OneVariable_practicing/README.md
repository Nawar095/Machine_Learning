# Linear Regression: Canadian Per Capita Income Prediction

## Overview
This repository demonstrates multiple implementations of simple linear regression to predict Canadian per capita income. The project showcases different approaches to implementing the same algorithm, from basic to advanced implementations, as part of a learning journey in machine learning.

## Project Structure
LinearReg_OneVariable_practicing/
├── data/
│ ├── canada_per_capita_income.csv # Historical income data (1970-2016)
│ └── test_data.csv # Generated test predictions
├── notebooks/
│ ├── Enhanced_LRexercise.ipynb # Enhanced implementation with visualization
│ └── LRexercise.ipynb # Basic implementation and testing
├── src/
│ ├── init.py # Python package marker
│ ├── canada_income_prediction.py # OOP implementation
│ └── lenRegOneVar.py # Modular implementation
└── tests/
├── test_canada_income_prediction.py # Unit tests for OOP implementation
└── test_lenRegOneVar.py # Unit tests for modular implementation

## Dataset
The project uses Canadian per capita income data (`canada_per_capita_income.csv`) from 1970 to 2016. The dataset includes:
- Year
- Per Capita Income (US $)

## Implementations

### 1. Basic Implementation (LRexercise.ipynb)
- Simple linear regression using sklearn
- Basic data visualization
- Prediction validation

### 2. Enhanced Implementation (Enhanced_LRexercise.ipynb)
- Improved visualization techniques
- Enhanced model evaluation
- More comprehensive testing approach

### 3. Production-Ready Implementation (src/canada_income_prediction.py)
- Object-oriented design
- Modular functions
- Comprehensive documentation
- Error handling
- Unit testing support

## Key Features
- Multiple implementation approaches
- Data visualization using matplotlib
- Model training and evaluation
- Future predictions (2018-2030)
- Comprehensive unit testing
- Modular code structure

## Getting Started

### Prerequisites
- pandas==1.5.3
- numpy==1.24.3
- matplotlib==3.7.1
- scikit-learn==1.2.2

### Installation
1. Clone the repository:
```bash
git clone https://github.com/Nawar095/Machine_Learning.git
```

2. Navigate to the project directory:
```bash
cd LinearReg_OneVariable_practicing
```

3. Install the required libraries:
```bash
pip install -r requirements.txt
```
### Running the Project

1. Using Jupyter Notebooks:
```bash
jupyter notebook notebooks/Enhanced_LRexercise.ipynb
```

2. Using Python Scripts:
```bash
python src/canada_income_prediction.py
```

3. Running Tests:
```bash
python -m unittest -v tests/test_canada_income_prediction.py
python -m unittest -v tests/test_lenRegOneVar.py
```

## Testing
The project includes two test suites:
1. `test_canada_income_prediction.py`: Tests for the OOP implementation
   - Data loading validation
   - Model training verification
   - Prediction accuracy
   - Parameter validation

2. `test_lenRegOneVar.py`: Tests for the modular implementation
   - Data integrity checks
   - Model functionality
   - Prediction validation

## Development Philosophy
This project represents a learning journey in machine learning, specifically focusing on:
1. Understanding linear regression fundamentals
2. Exploring different implementation approaches
3. Writing clean, maintainable code
4. Implementing proper testing practices
5. Building modular and reusable components

## Future Improvements
- [ ] Add cross-validation techniques
- [ ] Implement feature scaling
- [ ] Add more visualization options
- [ ] Include model performance metrics
- [ ] Add data preprocessing options

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
>You can check out the full license [here](https://github.com/Nawar095/Machine_Learning/blob/main/LICENSE)

This project is licensed under the terms of the **MIT** license.


---
*This project is part of a learning journey in machine learning, focusing on building strong fundamentals through practice and continuous improvement.*

