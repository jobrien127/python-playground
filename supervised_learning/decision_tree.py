import numpy as np
from typing import Tuple, Dict, Union, List
from collections import Counter


class Node:
    """
    A node in the decision tree.
    """
    def __init__(self, feature_idx: int = None, threshold: float = None,
                 value: float = None, left: 'Node' = None, right: 'Node' = None):
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.value = value             # For leaf nodes, the predicted value
        self.left = left               # Left child node
        self.right = right             # Right child node


class DecisionTree:
    """
    A from-scratch implementation of a Decision Tree classifier.
    """
    
    def __init__(self, max_depth: int = None, min_samples_split: int = 2):
        """
        Initialize the decision tree.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split a node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes = None
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Build the decision tree.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        """
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively grow the decision tree.
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
            
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:  # No valid split found
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
            
        # Create child nodes
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        # Update feature importances
        self.feature_importances_[best_feature] += (
            self._gini(y) - 
            (len(y[left_idxs]) / len(y)) * self._gini(y[left_idxs]) -
            (len(y[right_idxs]) / len(y)) * self._gini(y[right_idxs])
        )
        
        return Node(best_feature, best_threshold, left=left, right=right)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Union[int, None], Union[float, None]]:
        """
        Find the best split for a node.
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_idxs = X[:, feature_idx] < threshold
                right_idxs = ~left_idxs
                
                if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
                    continue
                
                gain = self._information_gain(y, y[left_idxs], y[right_idxs])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_feature, best_threshold

    def _gini(self, y: np.ndarray) -> float:
        """
        Calculate the Gini impurity.
        """
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)

    def _information_gain(self, parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        """
        Calculate the information gain of a split.
        """
        p = len(left) / len(parent)
        return self._gini(parent) - p * self._gini(left) - (1 - p) * self._gini(right)

    def _most_common_label(self, y: np.ndarray) -> int:
        """
        Return the most common label in a node.
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for samples in X.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        """
        Traverse the tree to make a prediction for a single sample.
        """
        if node.value is not None:
            return node.value
            
        if x[node.feature_idx] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    # Create a simple binary classification problem
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create and train model
    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)
    
    # Make predictions
    y_pred = tree.predict(X)
    accuracy = np.mean(y_pred == y)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nFeature Importances:")
    for i, importance in enumerate(tree.feature_importances_):
        print(f"Feature {i}: {importance:.4f}")
    
    # Visualize decision boundary
    import matplotlib.pyplot as plt
    
    def plot_decision_boundary(X: np.ndarray, y: np.ndarray, model: DecisionTree):
        h = 0.02  # Step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.xlabel('Feature 0')
        plt.ylabel('Feature 1')
        plt.title('Decision Tree Decision Boundary')
        plt.show()
    
    plot_decision_boundary(X, y, tree) 