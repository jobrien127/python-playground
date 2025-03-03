import numpy as np
from typing import Tuple


class LogisticRegression:
    """
    A from-scratch implementation of Logistic Regression using gradient descent.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.history = {'loss': []}

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model using gradient descent.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Binary target values of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Store loss
            loss = self._compute_loss(y, predictions)
            self.history['loss'].append(loss)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions for given features.
        
        Args:
            X: Features of shape (n_samples, n_features)
            threshold: Classification threshold
            
        Returns:
            Binary predictions of shape (n_samples,)
        """
        linear_pred = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear_pred)
        return (probabilities >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of class 1.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Probabilities of shape (n_samples,)
        """
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the Binary Cross Entropy loss.
        """
        epsilon = 1e-15  # Small constant to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

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
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Create and train model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)

    # Print results
    weights, bias = model.get_params()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Learned weights: {weights}")
    print(f"Learned bias: {bias}") 