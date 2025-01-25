import numpy as np
from PIL import Image
import os

# Define the sigmoid function and its derivative
def sigmoid(x):
    # x = np.clip(x, -500, 500)  # Avoid overflow
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Cross-Entropy Loss function
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))

def cross_entropy_loss_derivative(y_true, y_pred):
    return -(y_true / (y_pred + 1e-15)) + ((1 - y_true) / (1 - y_pred + 1e-15))

class Perceptron:
    def __init__(self, input_size, output_size, activation_func=sigmoid, activation_derivative=sigmoid_derivative):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative
        
        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs):
        """Compute the forward pass."""
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self.activation_func(self.z)
        return self.a

    def backward(self, y_true, learning_rate):
        """Perform backpropagation and update weights and biases."""
        loss_grad = cross_entropy_loss_derivative(y_true, self.a)

        # Gradients of activation function
        activation_grad = self.activation_derivative(self.z) * loss_grad

        # Gradients of weights and biases
        weight_grad = np.dot(self.inputs.T, activation_grad)
        bias_grad = np.sum(activation_grad, axis=0, keepdims=True)

        # Update weights and biases
        self.weights -= learning_rate * weight_grad
        self.biases -= learning_rate * bias_grad

    def train(self, X, y, epochs, learning_rate):
        """Train the perceptron on the given dataset."""
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = cross_entropy_loss(y, predictions)
            self.backward(y, learning_rate)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        """Predict outputs for the given inputs."""
        outputs = self.forward(X)
        return np.argmax(outputs, axis=1)

# Helper function to load images as flattened arrays
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

# Load training and test datasets
train_images, train_labels = load_images('handwritten_images')
test_images, _ = load_images('test_images', labels=False)

# Initialize and train the perceptron
input_size = 20 * 20 
output_size = 10 

perceptron = Perceptron(input_size, output_size)

# Train the perceptron
perceptron.train(train_images, train_labels, epochs=10, learning_rate=0.1)

print("Training complete.")
# Test the perceptron
predictions = perceptron.predict(test_images)
print("Predictions for test images:", predictions)



