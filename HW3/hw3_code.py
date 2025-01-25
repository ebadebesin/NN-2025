import numpy as np

class Activation:
    """Class for activation functions and their derivatives."""
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

class Neuron:
    """Class representing a single neuron."""
    def __init__(self, num_inputs):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand(1)

class Layer:
    """Class representing a layer of neurons."""
    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]

    def get_weights(self):
        return np.array([neuron.weights for neuron in self.neurons])

    def get_biases(self):
        return np.array([neuron.bias[0] for neuron in self.neurons]) 

class Model:
    """Class representing the entire neural network."""
    def __init__(self, input_size, hidden_layer_size, output_size):
        self.hidden_layer = Layer(hidden_layer_size, input_size)
        self.output_layer = Layer(output_size, hidden_layer_size)

class LossFunction:
    """Class for loss calculation."""
    @staticmethod
    def cross_entropy(y_true, y_pred):
        # To prevent log(0), add 1e-15 to y_pred
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        # Derivative of cross-entropy loss
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))

class ForwardProp:
    """Class for forward propagation."""
    @staticmethod
    def compute(model, inputs):
        hidden_inputs = np.dot(inputs, model.hidden_layer.get_weights().T) + model.hidden_layer.get_biases()
        hidden_outputs = Activation.sigmoid(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, model.output_layer.get_weights().T) + model.output_layer.get_biases()
        final_outputs = Activation.sigmoid(final_inputs)

        return hidden_outputs, final_outputs

class BackProp:
    """Class for backward propagation."""
    @staticmethod
    def compute(model, inputs, hidden_outputs, final_outputs, y_true, learning_rate):
        # Output layer error
        output_errors = LossFunction.cross_entropy_derivative(y_true, final_outputs)
        output_deltas = output_errors * Activation.sigmoid_derivative(final_outputs)

        # Hidden layer error
        hidden_errors = np.dot(output_deltas, model.output_layer.get_weights())
        hidden_deltas = hidden_errors * Activation.sigmoid_derivative(hidden_outputs)

        # Update output layer weights and biases
        for i, neuron in enumerate(model.output_layer.neurons):
            neuron.weights -= learning_rate * np.dot(hidden_outputs.T, output_deltas[:, i])
            neuron.bias -= learning_rate * np.sum(output_deltas[:, i])

        # Update hidden layer weights and biases
        for i, neuron in enumerate(model.hidden_layer.neurons):
            neuron.weights -= learning_rate * np.dot(inputs.T, hidden_deltas[:, i])
            neuron.bias -= learning_rate * np.sum(hidden_deltas[:, i])

class GradDescent:
    """Class for gradient descent optimization."""
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

class Training:
    """Class to train the model."""
    @staticmethod
    def train(model, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward propagation
            hidden_outputs, final_outputs = ForwardProp.compute(model, X)

            # Compute loss
            loss = LossFunction.cross_entropy(y, final_outputs)

            # Backward propagation
            BackProp.compute(model, X, hidden_outputs, final_outputs, y, learning_rate)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage
if __name__ == "__main__":
    # Input data 
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Initialize model
    nn = Model(input_size=2, hidden_layer_size=1, output_size=1)

    # Train model
    Training.train(nn, X, y, epochs=100, learning_rate=0.1)

    # Test model
    _, predictions = ForwardProp.compute(nn, X)
    print("Predictions:", predictions)

