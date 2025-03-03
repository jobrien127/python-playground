# Supervised Learning Models

This directory contains implementations of various supervised learning algorithms from scratch using NumPy and PyTorch. Each implementation includes detailed documentation, example usage, and visualization capabilities.

## Models

### Linear Regression (`linear_regression.py`)
A from-scratch implementation of Linear Regression using gradient descent optimization.
- Features:
  - Mean Squared Error (MSE) loss function
  - Batch gradient descent optimization
  - Learning rate and iteration control
  - Training history tracking
  - Parameter inspection methods

### Logistic Regression (`logistic_regression.py`)
Binary classification model implemented using gradient descent optimization.
- Features:
  - Binary Cross-Entropy loss function
  - Sigmoid activation function
  - Probability predictions
  - Configurable classification threshold
  - Training history tracking

### Neural Network (`neural_network.py`)
A flexible neural network implementation supporting multiple layers.
- Features:
  - Configurable layer sizes
  - ReLU and Sigmoid activation functions
  - Binary Cross-Entropy loss
  - Backpropagation implementation
  - Training history visualization
  - Example XOR problem solution

### Decision Tree (`decision_tree.py`)
Decision Tree classifier with Gini impurity criterion.
- Features:
  - Recursive tree construction
  - Gini impurity splitting criterion
  - Information gain calculation
  - Feature importance computation
  - Maximum depth control
  - Decision boundary visualization

### Support Vector Machine (`svm.py`)
SVM classifier using gradient descent optimization.
- Features:
  - Hinge loss with L2 regularization
  - Support vector identification
  - Decision boundary visualization
  - Training history tracking
  - Configurable learning rate and regularization
  - Support vector highlighting in plots

### Image Classification (`image_classification/`)
Deep learning models for image classification tasks.
- Features:
  - CNN implementations using PyTorch
  - MNIST digit classification
  - Training and evaluation pipelines
  - Model checkpointing
  - Training visualization
  - Device-agnostic training (CPU/GPU)

## Usage

Each model follows a similar API pattern:

```python
# Example usage pattern
model = ModelClass(parameters)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

See individual model files for specific usage examples and parameters.

## Requirements
- NumPy
- Matplotlib (for visualization)
- PyTorch (for image classification models)
- torchvision (for image classification models)

## Testing
Each model includes a `__main__` block with example usage and testing code. Run any model file directly to see it in action:

```bash
python model_name.py
``` 