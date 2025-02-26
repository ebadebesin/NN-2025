# Implement regularization algorithms in your neural network. 
# Implement dropout algorithms in your neural network. 

import numpy as np

class Activation:
    """Class containing different activation functions and their derivatives."""
    
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
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class Layer:
    def __init__(self, num_inputs, num_neurons, activation_function, dropout_rate=0.0, l2_lambda=0.0):
        self.weights = np.random.randn(num_inputs, num_neurons) * 0.01
        self.biases = np.zeros((1, num_neurons))
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate  # Dropout probability
        self.l2_lambda = l2_lambda  # Regularization 

    def forward(self, inputs, training=True):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self.activation_function(self.z)

        # Apply dropout during training
        if training and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=self.a.shape) / (1 - self.dropout_rate)
            self.a *= self.dropout_mask

        return self.a

class DeepNeuralNetwork:
    def __init__(self, layer_sizes, activations, dropout_rates=None, l2_lambda=0.0):
        self.layers = []
        dropout_rates = dropout_rates or [0.0] * len(layer_sizes)

        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], activations[i], dropout_rates[i], l2_lambda))

    def forward(self, X, training=True):
        for layer in self.layers:
            X = layer.forward(X, training)
        return X

class LossFunction:
    """Class for loss functions with L2 regularization."""
    
    @staticmethod
    def cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred))

    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        return y_pred - y_true  # Used with softmax

class BackPropagation:
    @staticmethod
    def calc(model, X, y_true, learning_rate, l2_lambda):
        output = model.forward(X, training=True)
        d_loss = LossFunction.cross_entropy_derivative(y_true, output)

        for layer in reversed(model.layers):
            # Correct activation derivative
            if layer.activation_function == Activation.relu:
                d_activation = Activation.relu_derivative(layer.z)
            elif layer.activation_function == Activation.softmax:
                d_activation = 1  # Softmax derivative is handled by cross-entropy
            
            d_loss *= d_activation

            d_weights = np.dot(layer.inputs.T, d_loss) + l2_lambda * layer.weights  # L2 Regularization
            d_biases = np.sum(d_loss, axis=0, keepdims=True)

            d_loss = np.dot(d_loss, layer.weights.T)

            layer.weights -= learning_rate * d_weights
            layer.biases -= learning_rate * d_biases


class Training:
    @staticmethod
    def train(model, X, y, X_val, y_val, epochs, learning_rate, l2_lambda=0.0):
        for epoch in range(epochs):
            output = model.forward(X, training=True)
            loss = LossFunction.cross_entropy(y, output)

            BackPropagation.calc(model, X, y, learning_rate, l2_lambda)

            if epoch % 10 == 0:
                train_acc = compute_accuracy(model, X, y)
                val_acc = compute_accuracy(model, X_val, y_val)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")



import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

def compute_accuracy(model, X, y_true):
    predictions = model.forward(X, training=False)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy


# Function to load images
def load_images(image_dir, labels=True):
    images = []
    labels_list = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            img = Image.open(os.path.join(image_dir, filename)).convert('L')
            img = img.resize((20, 20))
            img_array = np.array(img).flatten() / 255.0  # Normalize pixel values
            images.append(img_array)

            if labels:
                label = int(filename.split('_')[0]) # Extract the digit from the filename
                labels_list.append(label)

    images = np.array(images)
    labels_array = np.eye(10)[labels_list] if labels else None
    return images, labels_array

# Load the dataset
train_images, train_labels = load_images('handwritten_images')

# Split dataset (80% Train, 10% Validation, 10% Test)
X_train, X_temp, y_train, y_temp = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define a neural network with dropout and L2 regularization
layer_sizes = [400, 55, 36, 10]  # Output layer should match the number of classes (10)
activations = [Activation.relu, Activation.relu, Activation.softmax]
dropout_rates = [0.2, 0.2, 0.0]  # Apply dropout to first two hidden layers
l2_lambda = 0.01  # L2 Regularization coefficient

# Create model
model = DeepNeuralNetwork(layer_sizes, activations, dropout_rates, l2_lambda)

# Train the model
Training.train(model, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.01, l2_lambda=l2_lambda)
 
# test_accuracy = compute_accuracy(model, X_test, y_test)
# print(f"Test Accuracy: {test_accuracy:.4f}")

test_accuracy = compute_accuracy(model, X_test, y_test)
print(f"Final Test Accuracy: {test_accuracy:.4f}")

 