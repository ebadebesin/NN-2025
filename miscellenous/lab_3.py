from nn import BackPropagation
from nn import DeepNeuralNetwork
from nn import Activation
from nn import LossFunction
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class NeuralNetworkTrainer:
    """A class to train and evaluate the neural network on a dataset."""
    
    def __init__(self, model, X_train, y_train, X_test, y_test, learning_rate=0.003, epochs=500, 
                 clip_value=1.0, batch_size=None, early_stopping_patience=20):
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
        :param batch_size: Batch size for mini-batch training (None for full batch).
        :param early_stopping_patience: Number of epochs to wait for improvement before stopping.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.clip_value = clip_value
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train(self):
        """Train the neural network using the provided dataset."""
        n_samples = self.X_train.shape[0]
        
        # For early stopping, split training data into train and validation
        val_size = int(0.1 * n_samples)
        X_val = self.X_train[-val_size:]
        y_val = self.y_train[-val_size:]
        X_train_subset = self.X_train[:-val_size]
        y_train_subset = self.y_train[:-val_size]
        
        for epoch in range(self.epochs):
            # Full batch or mini-batch training
            if self.batch_size is None:
                # Full batch gradient descent
                output = self.model.forwardProp(X_train_subset)
                loss = self.model.loss_function(y_train_subset, output)
                BackPropagation.calc(self.model, X_train_subset, y_train_subset, self.learning_rate)
            else:
                # Mini-batch gradient descent
                indices = np.random.permutation(X_train_subset.shape[0])
                X_shuffled = X_train_subset[indices]
                y_shuffled = y_train_subset[indices]
                
                n_batches = int(np.ceil(X_train_subset.shape[0] / self.batch_size))
                total_loss = 0
                
                for batch in range(n_batches):
                    start_idx = batch * self.batch_size
                    end_idx = min((batch + 1) * self.batch_size, X_train_subset.shape[0])
                    
                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]
                    
                    output = self.model.forwardProp(X_batch)
                    batch_loss = self.model.loss_function(y_batch, output)
                    total_loss += batch_loss
                    
                    BackPropagation.calc(self.model, X_batch, y_batch, self.learning_rate)
                
                loss = total_loss / n_batches
            
            # Gradient clipping
            for layer in self.model.layers:
                np.clip(layer.weights, -self.clip_value, self.clip_value, out=layer.weights)
                np.clip(layer.biases, -self.clip_value, self.clip_value, out=layer.biases)
            
            # Early stopping check on validation set
            val_output = self.model.forwardProp(X_val)
            val_loss = self.model.loss_function(y_val, val_output)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Check for early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model weights (would need to implement this)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    def evaluate(self):
        """Evaluate the trained model on the test dataset."""
        # Get predictions
        predictions_prob = self.model.forwardProp(self.X_test)
        predictions = (predictions_prob > 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == self.y_test)
        
        # Flatten predictions and true labels for sklearn metrics
        y_true = self.y_test.flatten()
        y_pred = predictions.flatten()
        y_prob = predictions_prob.flatten()
        
        # Calculate additional metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except:
            auc_roc = 0  # In case of single class prediction
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
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': cm
        }
        
    def run(self):
        """Train and evaluate the model."""
        print("Starting Training...")
        self.train()
        print("\nTraining Complete. Evaluating Model...")
        return self.evaluate()


def create_model(input_size, hidden_layers, neurons_per_layer, activation_hidden, activation_output, loss_function):
    """
    Create a neural network model with specified architecture.
    
    :param input_size: Number of input features
    :param hidden_layers: Number of hidden layers
    :param neurons_per_layer: Number of neurons per hidden layer
    :param activation_hidden: Activation function for hidden layers
    :param activation_output: Activation function for output layer
    :param loss_function: Loss function to use
    :return: Initialized DeepNeuralNetwork model
    """
    # Create layer sizes array
    layer_sizes = [input_size]
    
    # Add hidden layers
    for _ in range(hidden_layers):
        layer_sizes.append(neurons_per_layer)
    
    # Add output layer (1 neuron for binary classification)
    layer_sizes.append(1)
    
    # Define activations
    activations = [activation_hidden] * hidden_layers + [activation_output]
    
    # Create model
    model = DeepNeuralNetwork(layer_sizes, activations, loss_function)
    model.loss_derivative = LossFunction.binary_cross_entropy_derivative  # Ensure correct derivative
    
    return model


def perform_hyperparameter_tuning(X_train, y_train, X_test, y_test, input_size):
    """
    Perform grid search for hyperparameter tuning.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param X_test: Test features
    :param y_test: Test labels
    :param input_size: Number of input features
    :return: Best model and parameters
    """
    # Define hyperparameter grid
    param_grid = {
        'hidden_layers': [1, 2, 3, 4],
        'neurons_per_layer': [16, 32, 64, 128],
        'activation_hidden': [Activation.relu, Activation.tanh],
        'learning_rate': [0.01, 0.003, 0.001, 0.0003],
        'batch_size': [16, 32, 64, None],
        'epochs': [100, 150],
        'clip_value': [1.0, 3.0, 5.0]
    }
    
    # Track best configuration
    best_score = 0
    best_accuracy = 0
    best_params = None
    best_model = None
    
    # Keep track of all results
    results = []
    
    # Reduced parameter combinations for faster execution
    hidden_layers_options = [2, 3, 4]  # Try 2 or 3 or 4 hidden layers
    neurons_options = [16, 32, 64]  # Try 16 or 32 or 64 neurons per layer
    activation_options = [Activation.relu, Activation.tanh]  # Try ReLU or tanh
    learning_rate_options = [0.003, 0.001, 0.005]  # Try 0.003 or 0.001 or 0.005 learning rate
    batch_size_options = [16, 32, None]  # Try batch size 16 or 32 or full batch
    
    print("Starting hyperparameter tuning...")
    
    # Create smaller train/validation split for faster tuning
    X_tune, X_val, y_tune, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Grid search
    for hidden_layers in hidden_layers_options:
        for neurons in neurons_options:
            for activation in activation_options:
                for lr in learning_rate_options:
                    for batch_size in batch_size_options:
                        # Create model with current parameters
                        model = create_model(
                            input_size=input_size,
                            hidden_layers=hidden_layers,
                            neurons_per_layer=neurons,
                            activation_hidden=activation,
                            activation_output=Activation.sigmoid,
                            loss_function=LossFunction.binary_cross_entropy
                        )
                        
                        # Create trainer
                        trainer = NeuralNetworkTrainer(
                            model=model,
                            X_train=X_tune,
                            y_train=y_tune,
                            X_test=X_val,
                            y_test=y_val,
                            learning_rate=lr,
                            epochs=80,  # Reduced for faster tuning
                            clip_value=3.0,
                            batch_size=batch_size,
                            early_stopping_patience=15
                        )
                        
                        # Print current configuration
                        config = {
                            'hidden_layers': hidden_layers,
                            'neurons': neurons,
                            'activation': activation.__name__ if hasattr(activation, '__name__') 
                                         else activation.__str__(),
                            'learning_rate': lr,
                            'batch_size': batch_size if batch_size is not None else 'full'
                        }
                        print(f"\nTrying configuration: {config}")
                        
                        # Train and evaluate
                        metrics = trainer.run()
                        
                        # Store results
                        result = {**config, **metrics}
                        results.append(result)
                        
                        # Update best if improved
                        if metrics['f1'] > best_score:
                            best_score = metrics['f1']
                            best_params = config
                            best_model = model
                        
                        #Update best if improved for accuracy
                        if metrics['accuracy'] > best_accuracy:
                            best_accuracy = metrics['accuracy']
                            best_params = config
                            best_model = model

    print("\n--- Hyperparameter Tuning Results ---")
    print(f"Best Parameters: {best_params}")
    print(f"Best F1 Score: {best_score:.4f}")
    print (f"Best Accuracy: {best_accuracy:.4f}")
    
    # Return best model and parameters
    return best_model, best_params


def analyze_feature_importance(model, feature_names):
    """
    Analyze feature importance based on weights of the first layer.
    
    :param model: Trained neural network model
    :param feature_names: List of feature names
    :return: Dictionary of feature importance scores
    """
    # Get weights from first layer
    first_layer_weights = model.layers[0].weights
    
    # Calculate importance as mean absolute weight
    importance = np.mean(np.abs(first_layer_weights), axis=1)
    
    # Create dictionary mapping features to importance
    feature_importance = {feature: float(score) for feature, score in zip(feature_names, importance)}
    
    # Sort by importance
    sorted_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
    
    return sorted_importance


if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv('Lab2/accident.csv')
    
    # Convert categorical variables to numerical values
    label_encoders = {}
    categorical_columns = ['Gender', 'Helmet_Used', 'Seatbelt_Used']
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    
    # Store feature names for later use
    feature_names = data.drop('Survived', axis=1).columns.tolist()
    
    # Separate features and target variable
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    
    # Count the number of survived and non-survived individuals
    survival_counts = data['Survived'].value_counts()
    print("Survival Counts:")
    print(survival_counts)
    
    # Handle missing values
    X.fillna(X.mean(), inplace=True)  # Replace NaNs with column mean
    
    # Normalize numerical features
    scaler = StandardScaler()
    X[['Age', 'Speed_of_Impact']] = scaler.fit_transform(X[['Age', 'Speed_of_Impact']])
    
    # Convert to NumPy arrays
    X = X.values
    y = y.values.reshape(-1, 1)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Get input size
    input_size = X_train.shape[1]
    
    # Option 1: Use Hyperparameter Tuning
    use_hyperparameter_tuning = True
    
    if use_hyperparameter_tuning:
        # Perform hyperparameter tuning
        best_model, best_params = perform_hyperparameter_tuning(X_train, y_train, X_test, y_test, input_size)
        
        # Create new model with best parameters and train on full training set
        final_model = create_model(
            input_size=input_size,
            hidden_layers=best_params['hidden_layers'],
            neurons_per_layer=best_params['neurons'],
            activation_hidden=Activation.relu if best_params['activation'] == 'relu' else Activation.tanh,
            activation_output=Activation.sigmoid,
            loss_function=LossFunction.binary_cross_entropy
        )
        
        # Train final model
        final_trainer = NeuralNetworkTrainer(
            model=final_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            learning_rate=best_params['learning_rate'],
            epochs=200,  # Train longer for final model
            clip_value=3.0,
            batch_size=best_params['batch_size'] if best_params['batch_size'] != 'full' else None,
            early_stopping_patience=25
        )
        
        print("\n--- Training Final Model with Best Parameters ---")
        final_trainer.run()
        
        # Analyze feature importance
        importance = analyze_feature_importance(final_model, feature_names)
        print("\nFeature Importance:")
        for feature, score in importance.items():
            print(f"{feature}: {score:.4f}")
        
    else:
        # Option 2: Use predefined architecture (original approach)
        layer_sizes = [input_size, 64, 32, 16, 1]  # More reasonable architecture
        activations = [Activation.relu, Activation.relu, Activation.relu, Activation.sigmoid]
        dnn = DeepNeuralNetwork(layer_sizes, activations, LossFunction.binary_cross_entropy)
        
        # Train with improved parameters
        trainer = NeuralNetworkTrainer(
            dnn, X_train, y_train, X_test, y_test, 
            learning_rate=0.001, 
            epochs=150, 
            clip_value=3.0,
            batch_size=32,
            early_stopping_patience=20
        )
        
        trainer.run()
        
        # Analyze feature importance
        importance = analyze_feature_importance(dnn, feature_names)
        print("\nFeature Importance:")
        for feature, score in importance.items():
            print(f"{feature}: {score:.4f}")




# --- Hyperparameter Tuning Results ---
# Best Parameters: {'hidden_layers': 3, 'neurons': 16, 'activation': 'relu', 'learning_rate': 0.003, 'batch_size': 'full'}
# Best F1 Score: 0.6667
# Best Accuracy: 0.6875

# --- Training Final Model with Best Parameters ---
# Starting Training...
# Epoch 0, Train Loss: 0.756486, Val Loss: 0.689623
# Epoch 10, Train Loss: 0.646818, Val Loss: 0.674650
# Epoch 20, Train Loss: 0.631974, Val Loss: 0.675871
# Epoch 30, Train Loss: 0.658495, Val Loss: 0.689792
# Epoch 40, Train Loss: 0.595063, Val Loss: 0.695366
# Epoch 50, Train Loss: 0.621457, Val Loss: 0.666939
# Early stopping at epoch 57

# Training Complete. Evaluating Model...

# Model Evaluation:
# Accuracy: 0.5250
# Precision: 0.5333
# Recall: 0.4000
# F1-Score: 0.4571
# AUC-ROC: 0.5225
# Confusion Matrix:
# [[13  7]
#  [12  8]]

# Feature Importance:
# Seatbelt_Used: 0.7004
# Age: 0.6328
# Gender: 0.6019
# Speed_of_Impact: 0.5002
# Helmet_Used: 0.4503


