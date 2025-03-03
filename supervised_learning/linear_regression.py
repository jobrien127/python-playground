import numpy as np
from typing import Tuple


class LinearRegression:
    """
    A from-scratch implementation of Linear Regression using gradient descent.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.history = {'loss': []}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the linear regression model using gradient descent.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            y_pred = self._forward(X)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Store loss
            loss = self._compute_loss(y, y_pred)
            self.history['loss'].append(loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for given features.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        return self._forward(X)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass.
        """
        return np.dot(X, self.weights) + self.bias

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the Mean Squared Error loss.
        """
        return np.mean((y_true - y_pred) ** 2)

    def get_params(self) -> Tuple[np.ndarray, float]:
        """
        Get the learned parameters.
        
        Returns:
            Tuple of (weights, bias)
        """
        return self.weights, self.bias


if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1).flatten()

    # Create and train model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Print results
    weights, bias = model.get_params()
    print(f"Learned weights: {weights}")
    print(f"Learned bias: {bias}") 