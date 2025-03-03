import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


class PCA:
    """
    A from-scratch implementation of Principal Component Analysis.
    """
    
    def __init__(self, n_components: int = None):
        """
        Initialize PCA.
        
        Args:
            n_components: Number of components to keep. If None, keep all components.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the PCA model to the data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
        """
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store components
        if self.n_components is None:
            self.n_components = X.shape[1]
        
        self.components = eigenvectors[:, :self.n_components]
        
        # Compute explained variance and ratio
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = (self.explained_variance_ / 
                                        np.sum(eigenvalues))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data by projecting it onto the principal components.
        
        Args:
            X: Data to transform of shape (n_samples, n_features)
            
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to its original space.
        
        Args:
            X_transformed: Transformed data of shape (n_samples, n_components)
            
        Returns:
            Data in original space of shape (n_samples, n_features)
        """
        return np.dot(X_transformed, self.components.T) + self.mean

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and transform the data in one step.
        """
        self.fit(X)
        return self.transform(X)

    def plot_explained_variance_ratio(self) -> None:
        """
        Plot the explained variance ratio for each component.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.explained_variance_ratio_), 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Explained Variance Ratio vs. Number of Components')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 300
    
    # Create correlated data
    t = np.linspace(0, 2*np.pi, n_samples)
    x1 = np.sin(t) + 0.1 * np.random.randn(n_samples)
    x2 = np.cos(t) + 0.1 * np.random.randn(n_samples)
    x3 = 0.2 * np.random.randn(n_samples)  # Noise
    
    X = np.column_stack((x1, x2, x3))
    
    # Create and fit PCA model
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)
    
    # Print results
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance explained:", np.sum(pca.explained_variance_ratio_))
    
    # Plot original vs transformed data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original data
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.5)
    ax1.set_title('Original Data (First 2 Dimensions)')
    ax1.grid(True)
    
    # Transformed data
    ax2.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.5)
    ax2.set_title('PCA Transformed Data')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot explained variance ratio
    pca.plot_explained_variance_ratio() 