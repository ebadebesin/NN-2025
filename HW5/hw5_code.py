import numpy as np

class Activation:
    """Class containing different activation functions and their derivatives."""
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def tanh(x):
        return np.tanh(x) # f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def softmax_derivative(output, y_true):
        return output - y_true  # For cross-entropy loss


class Layer:
    
    def __init__(self, num_inputs, num_neurons, activation_function):
        self.weights = np.random.randn(num_inputs, num_neurons) * 0.01
        self.biases = np.zeros((1, num_neurons))
        self.activation_function = activation_function
        self.activation_derivative = self.get_activation_derivative()
    
    def get_activation_derivative(self):
        """Returns the derivative function based on the activation function name."""
        return {
            Activation.linear: Activation.linear_derivative,
            Activation.relu: Activation.relu_derivative,
            Activation.sigmoid: Activation.sigmoid_derivative,
            Activation.tanh: Activation.tanh_derivative,
            Activation.softmax: Activation.softmax_derivative
        }[self.activation_function]
    
    def forwardProp(self, inputs):
        """Calculates the forward propagation."""
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self.activation_function(self.z)
        return self.a


class LossFunction:
    """Class for loss functions."""
    
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mean_squared_error_derivative(y_true, y_pred):
        return y_pred - y_true
    
    @staticmethod
    def cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred))

    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        return y_pred - y_true # For softmax activation


class DeepNeuralNetwork:
    """Implements a deep neural network with multiple layers."""
    
    def __init__(self, layer_sizes, activations, loss_function):
        """
        :layer_sizes: List of neurons per layer.
        :activations: List of activation functions for each layer.
        :loss_function: Loss function (Cross-Entropy).
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], activations[i]))

        self.loss_function = loss_function
        self.loss_derivative = LossFunction.cross_entropy_derivative if loss_function == LossFunction.cross_entropy else LossFunction.mean_squared_error_derivative
    
    def forwardProp(self, X):
        """Performs forward propagation through all layers."""
        for layer in self.layers:
            X = layer.forwardProp(X)
        return X


class BackPropagation:
    """Class for backpropagation and parameter updates."""
    
    @staticmethod
    def calc(model, X, y_true, learning_rate):
        """Calculates gradients and updates weights using backpropagation."""
        # Forward 
        output = model.forwardProp(X)
        
        # calculate loss derivative
        d_loss = model.loss_derivative(y_true, output)

        # Backward
        for layer in reversed(model.layers):
            d_activation = layer.activation_derivative(layer.a)
            d_loss *= d_activation

            d_weights = np.dot(layer.inputs.T, d_loss)
            d_biases = np.sum(d_loss, axis=0, keepdims=True)

            d_loss = np.dot(d_loss, layer.weights.T)

            # Update weights and biases
            layer.weights -= learning_rate * d_weights
            layer.biases -= learning_rate * d_biases


class Training:
    """Class to train the deep neural network."""
    
    @staticmethod
    def train(model, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            output = model.forwardProp(X)
            
            # calculate loss
            loss = model.loss_function(y, output)

            # Backpropagation
            BackPropagation.calc(model, X, y, learning_rate)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")


if __name__ == "__main__":
    # Example dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # Input
    y = np.array([[0], [1], [1], [0]]) # Target output

    # Define network structure
    layer_sizes = [2, 4, 4, 1]  # Input layer: 2, Hidden layers: 4 and 4 neurons, Output: 1 neuron
    activations = [Activation.relu, Activation.relu, Activation.sigmoid]  # Hidden layers use ReLU, output uses Sigmoid
    loss_function = LossFunction.cross_entropy

    # Initialize model
    dnn = DeepNeuralNetwork(layer_sizes, activations, loss_function)

    # Train model
    Training.train(dnn, X, y, epochs=10, learning_rate=0.1)

    # Test model
    predictions = dnn.forwardProp(X)
    print("Predictions:", predictions)
