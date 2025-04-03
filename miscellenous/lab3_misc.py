
from nn import BackPropagation
from nn import DeepNeuralNetwork
from nn import Activation
from nn import LossFunction
import numpy as np

class NeuralNetworkTrainer:
    """A class to train and evaluate the neural network on a dataset."""
    
    def __init__(self, model, X_train, y_train, X_test, y_test, learning_rate=0.003, epochs=500, clip_value=1.0):
        """
        Initialize the trainer with a dataset and model.

        :param model: An instance of DeepNeuralNetwork.
        :param X_train: Training features.
        :param y_train: Training labels.
        :param X_test: Test features.
        :param y_test: Test labels.
        :param learning_rate: Learning rate for training.
        :param epochs: Number of epochs for training.
        :param clip_value: Clipping value for weight updates.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.clip_value = clip_value

    def train(self):
        """Train the neural network using the provided dataset."""
        for epoch in range(self.epochs):
            # Forward pass
            output = self.model.forwardProp(self.X_train)  # Enable dropout during training
            
            # Compute loss
            loss = self.model.loss_function(self.y_train, output)

            # Backpropagation
            BackPropagation.calc(self.model, self.X_train, self.y_train, self.learning_rate)

            # Gradient clipping
            for layer in self.model.layers:
                np.clip(layer.weights, -self.clip_value, self.clip_value, out=layer.weights)
                np.clip(layer.biases, -self.clip_value, self.clip_value, out=layer.biases)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def evaluate(self):
        """Evaluate the trained model on the test dataset."""
        # Get predictions
        predictions = (self.model.forwardProp(self.X_test) > 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == self.y_test)
        
        # Flatten predictions and true labels for sklearn metrics
        y_true = self.y_test.flatten()
        y_pred = predictions.flatten()
        
        # Calculate additional metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, self.model.forwardProp(self.X_test).flatten())
        cm = confusion_matrix(y_true, y_pred)
        
        # Print metrics
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print("Confusion Matrix:")
        print(cm)

    def run(self):
        """Train and evaluate the model."""
        print("Starting Training...")
        self.train()
        print("\nTraining Complete. Evaluating Model...")
        self.evaluate()

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    # Load the dataset
    data = pd.read_csv('Lab2/accident.csv')
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

    # Count the number of survived and non-survived individuals
    survival_counts = data['Survived'].value_counts()
    # Print the counts
    print("Survival Counts:")
    print(survival_counts)

    X.fillna(X.mean(), inplace=True)  # Replace NaNs with column mean

    # Normalize numerical features
    scaler = StandardScaler()
    X[['Age', 'Speed_of_Impact']] = scaler.fit_transform(X[['Age', 'Speed_of_Impact']])

    # Convert to NumPy arrays
    X = X.values
    y = y.values.reshape(-1, 1)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    input_size = X_train.shape[1]
    layer_sizes = [input_size, 256, 128, 64, 32, 1]  # Increased neurons in hidden layers
    
    # Use ReLU for hidden layers and sigmoid for output
    activations = [Activation.relu, Activation.relu, Activation.relu, Activation.relu, Activation.sigmoid]
    
    # Binary cross-entropy is appropriate for survival prediction
    loss_function = LossFunction.binary_cross_entropy

    # Initialize the model without the unsupported dropout_rate argument
    dnn = DeepNeuralNetwork(layer_sizes, activations, loss_function)  # Removed dropout_rate

    # Create a trainer instance with updated learning rate and epochs
    trainer = NeuralNetworkTrainer(dnn, X_train, y_train, X_test, y_test, learning_rate=0.0001, epochs=100, clip_value=1.0)

    # Train and evaluate the model
    trainer.run()
    


# Sample Output
# Survival Counts:
# Survived
# 1    101
# 0     99
# Name: count, dtype: int64
# Starting Training... with learning rate 0.0001 and 100 epochs.
# Epoch 0, Loss: 0.6840383660910729
# Epoch 10, Loss: 0.6822503010313802
# Epoch 20, Loss: 0.6785195128713135
# Epoch 30, Loss: 0.6750049698951222
# Epoch 40, Loss: 0.6716280915716868
# Epoch 50, Loss: 0.6684573911885148
# Epoch 60, Loss: 0.6653262495797666
# Epoch 70, Loss: 0.6623324883894626
# Epoch 80, Loss: 0.6594837166125874
# Epoch 90, Loss: 0.6566684644628404

# Training Complete. Evaluating Model... layer_sizes = [input_size, 256, 128, 64, 32, 1] 
# Model Evaluation:
# Accuracy: 0.6500
# Precision: 0.6429
# Recall: 0.5000
# F1-Score: 0.5625
# AUC-ROC: 0.6086
# Confusion Matrix:
# [[17  5]
#  [ 9  9]]