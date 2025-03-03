# Image Classification Models

This directory contains implementations of image classification models using PyTorch. Each implementation includes model architecture, training pipeline, and evaluation tools.

## Models

### MNIST Classifier (`mnist/`)
A Convolutional Neural Network (CNN) implementation for MNIST digit classification.
- Features:
  - CNN architecture with 2 convolutional layers
  - Max pooling and dropout for regularization
  - Cross-entropy loss and Adam optimizer
  - Training and evaluation pipeline
  - Training history visualization
  - Model checkpointing
  - Device-agnostic training (CPU/GPU)

#### Architecture
```python
class MNISTClassifier(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)
```

#### Usage
```python
# Train the model
python mnist/mnist_classifier.py

# The script will:
# 1. Download MNIST dataset
# 2. Train the model
# 3. Save training results plot
# 4. Save model checkpoint
```

#### Requirements
- PyTorch
- torchvision
- matplotlib
- NumPy

#### Files
- `mnist_classifier.py`: Main implementation
- `mnist_classifier.pth`: Trained model weights
- `training_results.png`: Training history visualization 