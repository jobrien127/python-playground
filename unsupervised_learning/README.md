# Unsupervised Learning Models

This directory contains implementations of various unsupervised learning algorithms from scratch using NumPy. Each implementation includes detailed documentation, example usage, and visualization capabilities.

## Models

### K-Means Clustering (`kmeans.py`)
A from-scratch implementation of the K-Means clustering algorithm.
- Features:
  - Random centroid initialization
  - Inertia (within-cluster sum of squares) calculation
  - Convergence detection
  - Cluster assignment
  - Training history tracking
  - Cluster visualization for 2D data
  - Configurable number of clusters and iterations

### Principal Component Analysis (PCA) (`pca.py`)
Implementation of PCA for dimensionality reduction.
- Features:
  - Eigenvalue decomposition
  - Explained variance ratio calculation
  - Data transformation and inverse transformation
  - Component selection
  - Visualization of explained variance
  - Example with synthetic 3D data

## Usage

Each model follows a similar API pattern:

```python
# Example usage pattern
model = ModelClass(parameters)
model.fit(X)  # No y required for unsupervised learning
transformed_data = model.transform(X)  # For PCA
# OR
predictions = model.predict(X)  # For K-Means
```

See individual model files for specific usage examples and parameters.

## Requirements
- NumPy
- Matplotlib (for visualization)

## Testing
Each model includes a `__main__` block with example usage and testing code. Run any model file directly to see it in action:

```bash
python model_name.py
```

## Visualization Examples
Both models include visualization capabilities:
- K-Means: Visualize clusters and centroids in 2D space
- PCA: Plot explained variance ratio and compare original vs transformed data 