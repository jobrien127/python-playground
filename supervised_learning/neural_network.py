import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt


class Layer:
    """
    A single layer in the neural network.
    """
    def __init__(self, n_inputs: int, n_neurons: int, activation: str = 'relu'):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation
        
        # Cache for backpropagation
        self.input = None
        self.output = None
        self.d_weights = None
        self.d_biases = None
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        """
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
        if self.activation == 'relu':
            self.output = np.maximum(0, self.output)
        elif self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-self.output))
            
        return self.output
    
    def backward(self, d_values: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer.
        """
        if self.activation == 'relu':
            d_values = d_values.copy()
            d_values[self.output <= 0] = 0
        elif self.activation == 'sigmoid':
            d_values = d_values * self.output * (1 - self.output)
            
        self.d_weights = np.dot(self.input.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
        
        return np.dot(d_values, self.weights.T)


class NeuralNetwork:
    """
    A simple neural network implementation with configurable layers.
    """
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of integers specifying the size of each layer
            learning_rate: Learning rate for gradient descent
        """
        self.layers = []
        self.learning_rate = learning_rate
        self.history = {'loss': []}
        
        # Create layers
        for i in range(len(layer_sizes) - 1):
            activation = 'sigmoid' if i == len(layer_sizes) - 2 else 'relu'
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], activation))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, d_values: np.ndarray) -> None:
        """
        Backward pass through the network.
        """
        for layer in reversed(self.layers):
            d_values = layer.backward(d_values)
            
            # Update parameters
            layer.weights -= self.learning_rate * layer.d_weights
            layer.biases -= self.learning_rate * layer.d_biases

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000) -> None:
        """
        Train the neural network.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target values of shape (n_samples, n_outputs)
            epochs: Number of training epochs
        """
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, output)
            self.history['loss'].append(loss)
            
            # Backward pass
            d_loss = (output - y) / (output * (1 - output))  # Derivative of binary cross-entropy
            self.backward(d_loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions for given features.
        """
        predictions = self.forward(X)
        return (predictions >= threshold).astype(int)

    def plot_loss_history(self) -> None:
        """
        Plot the training loss history.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['loss'])
        plt.title('Training Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Generate sample data for XOR problem
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create and train model
    model = NeuralNetwork([2, 4, 1], learning_rate=0.1)
    model.fit(X, y, epochs=1000)
    
    # Make predictions
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    
    print("\nPredictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}, Predicted: {predictions[i][0]}")
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Plot loss history
    model.plot_loss_history() 