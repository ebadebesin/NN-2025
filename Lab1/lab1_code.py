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
    def sigmoid_derivative(a):
        return a * (1 - a)  # 'a' is already sigmoid(z), not raw input


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
        if activation_function == Activation.sigmoid or activation_function == Activation.tanh:
            self.weights = np.random.randn(num_inputs, num_neurons) * np.sqrt(1 / num_inputs)  # Xavier for Sigmoid/Tanh
        else:
            self.weights = np.random.randn(num_inputs, num_neurons) * np.sqrt(2 / num_inputs)  # He for ReLU
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
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-15  # Small constant to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Keep y_pred in range (epsilon, 1-epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred):
        # return y_pred - y_true
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip values
        return (y_pred - y_true) / (y_pred * (1 - y_pred) + epsilon)  # Avoid division by zero




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
        self.loss_derivative = LossFunction.binary_cross_entropy_derivative 
        
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
    def train(model, X, y, epochs, learning_rate, clip_value=1.0):
        for epoch in range(epochs):
            # Forward pass
            output = model.forwardProp(X)
            
            # calculate loss
            loss = model.loss_function(y, output)

            # Backpropagation
            BackPropagation.calc(model, X, y, learning_rate)

            # Gradient clipping
            for layer in model.layers:
                np.clip(layer.weights, -clip_value, clip_value, out=layer.weights)
                np.clip(layer.biases, -clip_value, clip_value, out=layer.biases)
            

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split

    # Load the dataset
    data = pd.read_csv('Lab1/accident.csv')
    # print(data.head())

    # Convert categorical variables to numerical values
    label_encoders = {}
    categorical_columns = ['Gender', 'Helmet_Used', 'Seatbelt_Used']

    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # Separate features and target variable
    X = data.drop('Survived', axis=1)
    y = data['Survived']


    X.fillna(X.mean(), inplace=True)  # Replace NaNs with column mean


    # Normalize numerical features
    scaler = StandardScaler()
    X[['Age', 'Speed_of_Impact']] = scaler.fit_transform(X[['Age', 'Speed_of_Impact']])

    # Convert to NumPy arrays
    X = X.values
    y = y.values.reshape(-1, 1)  # Ensure y is a column vector

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print first few rows for verification
    print("X_train:", X_train[:5])
    print("y_train:", y_train[:5])

    # Define network structure
    input_size = X_train.shape[1]
    layer_sizes = [input_size, 10, 8, 8, 4, 1]  # Adjust input layer size based on features
    activations = [Activation.relu, Activation.relu, Activation.relu, Activation.relu, Activation.sigmoid]
    loss_function = LossFunction.binary_cross_entropy

    # Initialize model
    dnn = DeepNeuralNetwork(layer_sizes, activations, loss_function)

    # Train model
    Training.train(dnn, X_train, y_train, epochs=100, learning_rate=0.001, clip_value=1.0)

    # Test model
    predictions = (dnn.forwardProp(X_test) > 0.5).astype(int)
    print("Predictions:", predictions.T)

    # Accuracy
    accuracy = np.mean(predictions == y_test)
    print("Accuracy:", accuracy)




    
    # # Example dataset (you can replace this with your actual dataset)
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0]])  # Input
    # y = np.array([[0], [1], [1], [0], [0]])  # Target output

    # # Define network structure
    # layer_sizes = [2, 10, 8, 8, 4, 1]  # Input layer: 2, Hidden layers: 10, 8, 8, 4 neurons, Output: 1 neuron
    # activations = [Activation.relu, Activation.relu, Activation.relu, Activation.relu, Activation.sigmoid]  # Activation functions
    # loss_function = LossFunction.binary_cross_entropy

    # # Initialize model
    # dnn = DeepNeuralNetwork(layer_sizes, activations, loss_function)

    # # Train model
    # Training.train(dnn, X, y, epochs=1000, learning_rate=0.01)

    # # Test model
    # predictions = (dnn.forwardProp(X) > 0.5).astype(int)
    # print("Predictions:", predictions)

    # # Accuracy
    # accuracy = np.mean(predictions == y)
    # print("Accuracy:", accuracy)  # Should be around 1.0 (100%) for this simple dataset


############ Sample output: with 100 epochs #############
# Epoch 0, Loss: 0.7215771002950686
# Epoch 10, Loss: 0.6895876353216396
# Epoch 20, Loss: 0.6690710077637243
# Epoch 30, Loss: 0.6565118345872337
# Epoch 40, Loss: 0.6375997022990776
# Epoch 50, Loss: 0.6272684820393024
# Epoch 60, Loss: 0.6195957247056054
# Epoch 70, Loss: 0.6120389092054422
# Epoch 80, Loss: 0.6045839287629298
# Epoch 90, Loss: 0.5971286395569498
# Predictions: [[0]
#  [1]
#  [1]
#  [0]
#  [0]]
# Accuracy: 1.0



############ Sample output: with 1000 epochs ##########
# Epoch 0, Loss: 0.698334498848663
# Epoch 10, Loss: 0.6959416442113326
# Epoch 20, Loss: 0.6933640077849832
# Epoch 30, Loss: 0.6908418776556723
# Epoch 40, Loss: 0.6883843890817488
# Epoch 50, Loss: 0.6871529873968563
# Epoch 60, Loss: 0.6859636125277021
# Epoch 70, Loss: 0.6848177176365441
# Epoch 80, Loss: 0.6837096233703408
# Epoch 90, Loss: 0.6826265344746143
# Epoch 100, Loss: 0.6815770401926836
# Epoch 110, Loss: 0.680551273350809
# Epoch 120, Loss: 0.6795471473131623
# Epoch 130, Loss: 0.6785547603270519
# Epoch 140, Loss: 0.6775732329142445
# Epoch 150, Loss: 0.6765980488298068
# Epoch 160, Loss: 0.6756235395670375
# Epoch 170, Loss: 0.6746475631919342
# Epoch 180, Loss: 0.6736678558390479
# Epoch 190, Loss: 0.6726781424291985
# Epoch 200, Loss: 0.6716670598180698
# Epoch 210, Loss: 0.6706328670712081
# Epoch 220, Loss: 0.669563138590604
# Epoch 230, Loss: 0.6684509481722027
# Epoch 240, Loss: 0.667313301611504
# Epoch 250, Loss: 0.6661417394955822
# Epoch 260, Loss: 0.6650160113918777
# Epoch 270, Loss: 0.663869026369181
# Epoch 280, Loss: 0.6626490456050964
# Epoch 290, Loss: 0.6613871778802054
# Epoch 300, Loss: 0.6600878346006165
# Epoch 310, Loss: 0.6586795813825344
# Epoch 320, Loss: 0.6572162380870884
# Epoch 330, Loss: 0.655647067118815
# Epoch 340, Loss: 0.6540184722100183
# Epoch 350, Loss: 0.6522693650711142
# Epoch 360, Loss: 0.6504563635654823
# Epoch 370, Loss: 0.6484975470313301
# Epoch 380, Loss: 0.6464473785037518
# Epoch 390, Loss: 0.6442130868505395
# Epoch 400, Loss: 0.6418850137219978
# Epoch 410, Loss: 0.6393515463073132
# Epoch 420, Loss: 0.6368339812184174
# Epoch 430, Loss: 0.6340802483516289
# Epoch 440, Loss: 0.6312585626871805
# Epoch 450, Loss: 0.6281953949857086
# Epoch 460, Loss: 0.6251449912467901
# Epoch 470, Loss: 0.6216679210134355
# Epoch 480, Loss: 0.6181382650903667
# Epoch 490, Loss: 0.6143928016965978
# Epoch 500, Loss: 0.6104337449396958
# Epoch 510, Loss: 0.6061227187892324
# Epoch 520, Loss: 0.6018608901828144
# Epoch 530, Loss: 0.5971988280215913
# Epoch 540, Loss: 0.5912676830603749
# Epoch 550, Loss: 0.5839357715092276
# Epoch 560, Loss: 0.5763426168546756
# Epoch 570, Loss: 0.5684848282015462
# Epoch 580, Loss: 0.5605551493260177
# Epoch 590, Loss: 0.5523563532475375
# Epoch 600, Loss: 0.5441754717323336
# Epoch 610, Loss: 0.5359981903899583
# Epoch 620, Loss: 0.5275372935961975
# Epoch 630, Loss: 0.5185884560628299
# Epoch 640, Loss: 0.5099070235674056
# Epoch 650, Loss: 0.5006396105159032
# Epoch 660, Loss: 0.4915636454853625
# Epoch 670, Loss: 0.48202587821942877
# Epoch 680, Loss: 0.47268228388534334
# Epoch 690, Loss: 0.4636227278951927
# Epoch 700, Loss: 0.45385315728199715
# Epoch 710, Loss: 0.4444299953185412
# Epoch 720, Loss: 0.4350988612550818
# Epoch 730, Loss: 0.4253536093551795
# Epoch 740, Loss: 0.4158770219490788
# Epoch 750, Loss: 0.40684308657973567
# Epoch 760, Loss: 0.3974570527548191
# Epoch 770, Loss: 0.3883726930453468
# Epoch 780, Loss: 0.37934306442494625
# Epoch 790, Loss: 0.370835760258741
# Epoch 800, Loss: 0.36236076054384075
# Epoch 810, Loss: 0.35417453515635106
# Epoch 820, Loss: 0.3464097451890643
# Epoch 830, Loss: 0.33891948037329545
# Epoch 840, Loss: 0.331450926915544
# Epoch 850, Loss: 0.32444240092300103
# Epoch 860, Loss: 0.31754211108044716
# Epoch 870, Loss: 0.31128214467386045
# Epoch 880, Loss: 0.3050772874004155
# Epoch 890, Loss: 0.2990189305501759
# Epoch 900, Loss: 0.2931292618772421
# Epoch 910, Loss: 0.28761469068425305
# Epoch 920, Loss: 0.2822242236029654
# Epoch 930, Loss: 0.2774767732527579
# Epoch 940, Loss: 0.27243639594321
# Epoch 950, Loss: 0.267385555854213
# Epoch 960, Loss: 0.26275370478746124
# Epoch 970, Loss: 0.2582607996598647
# Epoch 980, Loss: 0.25405007888444764
# Epoch 990, Loss: 0.2499327036070597
# Predictions: [[0]
#  [1]
#  [1]
#  [0]
#  [0]]
# Accuracy: 1.0


############ Sample output: with 10000 epochs ############
# X_train: [[ 0.64242306  1.         -1.18841803  1.          0.        ]
#  [ 0.97789202  1.          1.32645971  0.          1.        ]
#  [ 1.1791734   0.          1.1923329   1.          1.        ]
#  [-0.36398382  0.         -1.52373506  1.          1.        ]
#  [-1.30329691  1.         -1.28901314  0.          0.        ]]
# y_train: [[0]
#  [1]
#  [0]
#  [0]
#  [1]]
# Epoch 0, Loss: 0.6939286948673203
# Epoch 10, Loss: 0.6906351176184179
# Epoch 20, Loss: 0.6902941536732282
# Epoch 30, Loss: 0.6899225702893849
# Epoch 40, Loss: 0.6895987188040069
# Predictions: [[1 1 0 0 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#   1 1 1 1]]
# Accuracy: 0.5

# X_train: [[ 0.64242306  1.         -1.18841803  1.          0.        ]
#  [ 0.97789202  1.          1.32645971  0.          1.        ]
#  [ 1.1791734   0.          1.1923329   1.          1.        ]
#  [-0.36398382  0.         -1.52373506  1.          1.        ]
#  [-1.30329691  1.         -1.28901314  0.          0.        ]]
# y_train: [[0]
#  [1]
#  [0]
#  [0]
#  [1]]
# Epoch 0, Loss: 0.687163141106254
# Epoch 10, Loss: 0.6880993124097476
# Epoch 20, Loss: 0.6870076406746752
# Epoch 30, Loss: 0.6857835895308155
# Epoch 40, Loss: 0.6843095303318802
# Predictions: [[1 0 1 1 1 1 0 0 1 1 1 1 0 0 0 0 1 0 1 0 0 1 1 1 1 1 1 1 0 1 0 0 0 1 0 1
#   1 1 0 1]]
# Accuracy: 0.65