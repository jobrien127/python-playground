import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class SVM:
    """
    A from-scratch implementation of Support Vector Machine classifier.
    Uses gradient descent to optimize the dual form of the SVM objective.
    """
    
    def __init__(self, learning_rate: float = 0.001, n_iterations: int = 1000, 
                 lambda_param: float = 0.01):
        """
        Initialize SVM classifier.
        
        Args:
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations for training
            lambda_param: Regularization parameter
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param
        self.w = None
        self.b = None
        self.history = {'loss': []}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the SVM model using gradient descent.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            linear_output = np.dot(X, self.w) + self.b
            
            # Compute gradients
            dw = np.zeros(n_features)
            db = 0
            
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if not condition:
                    dw -= y[idx] * x_i
                    db -= y[idx]
            
            # Add regularization term
            dw += self.lambda_param * self.w
            
            # Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # Store loss
            loss = self._compute_loss(X, y)
            self.history['loss'].append(loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for given features.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the hinge loss with L2 regularization.
        """
        linear_output = np.dot(X, self.w) + self.b
        hinge_loss = np.maximum(0, 1 - y * linear_output)
        l2_loss = 0.5 * self.lambda_param * np.dot(self.w, self.w)
        return np.mean(hinge_loss) + l2_loss

    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray, 
                             title: str = "SVM Decision Boundary") -> None:
        """
        Plot the decision boundary and support vectors (works only for 2D data).
        
        Args:
            X: Data of shape (n_samples, 2)
            y: Labels of shape (n_samples,)
            title: Plot title
        """
        if X.shape[1] != 2:
            raise ValueError("Data must be 2-dimensional for plotting")
            
        # Create a mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        # Make predictions on the mesh grid
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        
        # Plot support vectors
        support_vectors = self._get_support_vectors(X, y)
        if len(support_vectors) > 0:
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                       c='red', marker='x', s=100, linewidths=3, 
                       label='Support Vectors')
        
        plt.title(title)
        plt.xlabel('Feature 0')
        plt.ylabel('Feature 1')
        plt.legend()
        plt.show()

    def _get_support_vectors(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get the support vectors from the training data.
        """
        linear_output = np.dot(X, self.w) + self.b
        margin = np.abs(linear_output - 1)
        support_vector_indices = margin < 1e-5
        return X[support_vector_indices]

    def plot_loss_history(self) -> None:
        """
        Plot the training loss history.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['loss'])
        plt.title('Training Loss History')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    # Create two classes
    X = np.random.randn(n_samples, 2)
    y = np.zeros(n_samples)
    y[X[:, 0] + X[:, 1] > 0] = 1
    y[X[:, 0] + X[:, 1] <= 0] = -1
    
    # Create and train model
    model = SVM(learning_rate=0.01, n_iterations=1000, lambda_param=0.01)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    
    print(f"Accuracy: {accuracy:.4f}")
    
    # Plot results
    model.plot_decision_boundary(X, y)
    model.plot_loss_history() 