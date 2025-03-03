import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


class KMeans:
    """
    A from-scratch implementation of K-Means clustering algorithm.
    """
    
    def __init__(self, n_clusters: int = 3, max_iters: int = 100, random_state: int = None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.history = {'centroids': [], 'inertia': []}

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the K-Means model to the data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # Assign points to nearest centroid
            self.labels = self._assign_clusters(X)
            
            # Update centroids
            for k in range(self.n_clusters):
                if np.sum(self.labels == k) > 0:  # Avoid empty clusters
                    self.centroids[k] = np.mean(X[self.labels == k], axis=0)
            
            # Store history
            self.history['centroids'].append(self.centroids.copy())
            self.inertia_ = self._compute_inertia(X)
            self.history['inertia'].append(self.inertia_)
            
            # Check convergence
            if np.all(old_centroids == self.centroids):
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for samples in X.
        
        Args:
            X: Data of shape (n_samples, n_features)
            
        Returns:
            Cluster labels of shape (n_samples,)
        """
        return self._assign_clusters(X)

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample to the nearest centroid.
        """
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _compute_inertia(self, X: np.ndarray) -> float:
        """
        Compute the sum of squared distances of samples to their closest centroid.
        """
        distances = np.sqrt(((X - self.centroids[self.labels])**2).sum(axis=1))
        return np.sum(distances**2)

    def plot_clusters(self, X: np.ndarray, title: str = "K-Means Clustering") -> None:
        """
        Plot the clusters and centroids (works only for 2D data).
        
        Args:
            X: Data of shape (n_samples, 2)
            title: Plot title
        """
        if X.shape[1] != 2:
            raise ValueError("Data must be 2-dimensional for plotting")
            
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=self.labels, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        plt.title(title)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 300
    
    # Create three clusters
    cluster1 = np.random.normal(loc=[0, 0], scale=0.5, size=(n_samples//3, 2))
    cluster2 = np.random.normal(loc=[2, 2], scale=0.5, size=(n_samples//3, 2))
    cluster3 = np.random.normal(loc=[-1, 2], scale=0.5, size=(n_samples//3, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # Create and train model
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X)
    
    # Plot results
    model.plot_clusters(X)
    print(f"Final inertia: {model.inertia_:.4f}") 