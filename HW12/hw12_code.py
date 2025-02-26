import numpy as np

class SoftmaxActivation:
    """Softmax activation function with forward and backward propagation."""
    
    @staticmethod
    def forward(x):
        """Computes softmax activation for input x."""
        x_exponent = np.exp(x - np.max(x, axis=1, keepdims=True)) # For numerical stability
        return x_exponent / np.sum(x_exponent, axis=1, keepdims=True)
    
    @staticmethod
    def backward(d_out, y_pred, y_true):
        """Computes gradient of softmax loss (cross-entropy loss)."""
        return y_pred - y_true  # Only works when combined with cross-entropy loss
