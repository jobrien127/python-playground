# Machine Learning From Scratch

This repository contains implementations of various machine learning algorithms from scratch using NumPy. The goal is to provide clear, well-documented implementations that help understand the underlying mechanics of these algorithms.

## Project Structure

```
.
├── supervised_learning/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── neural_network.py
│   ├── decision_tree.py
│   ├── svm.py
│   └── README.md
├── unsupervised_learning/
│   ├── kmeans.py
│   ├── pca.py
│   └── README.md
├── requirements.txt
└── README.md
```

## Implemented Models

### Supervised Learning
- Linear Regression (with gradient descent)
- Logistic Regression (binary classification)
- Neural Network (multi-layer with backpropagation)
- Decision Tree (with Gini impurity)
- Support Vector Machine (with gradient descent)

### Unsupervised Learning
- K-Means Clustering
- Principal Component Analysis (PCA)

## Features
- Pure NumPy implementations
- Detailed documentation and type hints
- Visualization capabilities
- Example usage with synthetic datasets
- Training history tracking
- Model evaluation metrics

## Requirements
All implementations use minimal dependencies:
```
numpy>=1.24.0
matplotlib>=3.8.0
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/machine-learning-from-scratch.git
cd machine-learning-from-scratch
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Each model can be used independently. Here's a general pattern:

```python
# Import the model
from supervised_learning.linear_regression import LinearRegression

# Create and train the model
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

See individual model directories for specific examples and documentation.

## Testing
Each model includes a `__main__` block with example usage. To test a model:

```bash
python supervised_learning/linear_regression.py
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- These implementations are meant for educational purposes
- Inspired by various machine learning textbooks and courses
- Thanks to the NumPy and Matplotlib teams for their excellent libraries 