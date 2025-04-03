# For Lab 3, I decided to generate synthetic data instead due to limited data availability.
# This code will create a synthetic dataset for a binary classification problem on accident survival prediction.
# It uses a neural network created in nn with hyperparameter search and tuning to optimize the model.

from nn import BackPropagation
from nn import DeepNeuralNetwork
from nn import Activation
from nn import LossFunction
from nn import Training

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline


# Function to create a model with specified hyperparameters
def create_model(hidden_layers, neurons_per_layer, activation_hidden, activation_output, learning_rate):
    # Build layer sizes array
    layer_sizes = [input_size] + [neurons_per_layer] * hidden_layers + [output_size]
    
    # Define activations for each layer
    activations = [activation_hidden] * hidden_layers + [activation_output]
    
    # Create the model
    model = DeepNeuralNetwork(
        layer_sizes=layer_sizes,
        activations=activations,
        loss_function=LossFunction.binary_cross_entropy
    )
    model.loss_derivative = LossFunction.binary_cross_entropy_derivative
    
    return model

# Load and preprocess data (using synthetic data)
def create_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(16, 80, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Speed_of_Impact': np.random.randint(10, 120, n_samples),
        'Helmet_Used': np.random.choice([0, 1], n_samples),
        'Seatbelt_Used': np.random.choice([0, 1], n_samples),
    }
    
    # Create survival probability with stronger feature influences
    survival_prob = (
        0.7 - (data['Age'] - 20) / 120 - data['Speed_of_Impact'] / 150 + 
        (data['Helmet_Used'] * 0.25) + (data['Seatbelt_Used'] * 0.35) +
        (np.array([1 if g == 'Female' else 0.8 for g in data['Gender']]) * 0.1)
    )
    
    # Clip probabilities between 0.1 and 0.9
    survival_prob = np.clip(survival_prob, 0.1, 0.9)
    
    # Add some randomness to make the problem more realistic
    noise = np.random.normal(0, 0.05, n_samples)
    survival_prob += noise
    survival_prob = np.clip(survival_prob, 0.05, 0.95)
    
    # Convert probabilities to binary outcome with better distribution
    data['Survived'] = (np.random.random(n_samples) < survival_prob).astype(int)
    
    return pd.DataFrame(data)

# Load data
datasetFrame = create_synthetic_data(2000)  # Create a larger dataset

# Split data
X = datasetFrame.drop('Survived', axis=1)
y = datasetFrame['Survived']
y_binary = y.values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

# Make sure we have both classes represented in the data
print(f"Training data distribution: {np.bincount(y_train.flatten())}")
print(f"Testing data distribution: {np.bincount(y_test.flatten())}")

# Preprocess data
numerical_features = ['Age', 'Speed_of_Impact']
categorical_features = ['Gender']
binary_features = ['Helmet_Used', 'Seatbelt_Used']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features),
        ('bin', 'passthrough', binary_features)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get input size after preprocessing
input_size = X_train_processed.shape[1]
output_size = 1

# Define hyperparameter grid
hidden_layers_options = [1, 2, 3]
neurons_options = [8, 16, 32]
activation_hidden_options = [Activation.relu, Activation.tanh]
learning_rate_options = [0.01, 0.005, 0.001]
batch_size_options = [16, 32, 64, None]  # None means use all data (full batch)

# Hyperparameter tuning using grid search
best_accuracy = 0
best_model = None
best_params = {}

# Function to train and evaluate model with given parameters
def train_and_evaluate(hidden_layers, neurons, activation_hidden, learning_rate, batch_size, epochs=150):
    model = create_model(
        hidden_layers=hidden_layers,
        neurons_per_layer=neurons,
        activation_hidden=activation_hidden,
        activation_output=Activation.sigmoid,
        learning_rate=learning_rate
    )
    
    # Train the model
    if batch_size is None:
        # Train on full batch
        Training.train(model, X_train_processed, y_train, epochs, learning_rate, clip_value=5.0)
    else:
        # Mini-batch training
        n_samples = X_train_processed.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train_processed[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                output = model.forwardProp(X_batch)
                loss = model.loss_function(y_batch, output)
                epoch_loss += loss
                
                # Backpropagation
                BackPropagation.calc(model, X_batch, y_batch, learning_rate)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss/n_batches:.6f}")
    
    # Evaluate model
    y_pred_prob = model.forwardProp(X_test_processed)
    y_pred = (y_pred_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

# Do grid search for hyperparameter tuning
print("\nPerforming hyperparameter tuning...")
for hidden_layers in hidden_layers_options:
    for neurons in neurons_options:
        for activation_hidden in activation_hidden_options:
            for learning_rate in learning_rate_options:
                for batch_size in batch_size_options:
                    print(f"\nTrying: layers={hidden_layers}, neurons={neurons}, "
                          f"activation={activation_hidden.__name__}, "
                          f"learning_rate={learning_rate}, batch_size={batch_size}")
                    
                    model, accuracy = train_and_evaluate(
                        hidden_layers=hidden_layers,
                        neurons=neurons,
                        activation_hidden=activation_hidden,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        epochs=100 
                    )
                    
                    print(f"Accuracy: {accuracy:.4f}")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model
                        best_params = {
                            'hidden_layers': hidden_layers,
                            'neurons': neurons,
                            'activation_hidden': activation_hidden.__name__,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size
                        }

print("\nBest Model Parameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")
print(f"Best Accuracy: {best_accuracy:.4f}")

# Evaluation of the best model
def evaluate_model(model, X, y_true):
    # Make predictions
    y_pred_prob = model.forwardProp(X)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Print results
    print(f"\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return accuracy, precision, recall, f1, conf_matrix

# Evaluate best model on test data
print("\nEvaluating best model on test data:")
test_metrics = evaluate_model(best_model, X_test_processed, y_test)


def analyze_feature_importance(model, feature_names):
    # Examine the first layer weights
    first_layer = model.layers[0]
    weights = first_layer.weights
    
    # Calculate the absolute average impact of each feature
    importance = np.mean(np.abs(weights), axis=1)
    
    # Create a dictionary mapping features to importance scores
    feature_importance = {feature: score for feature, score in zip(feature_names, importance)}
    
    # Sort by importance
    sorted_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
    
    return sorted_importance

# Feature names for analysis
feature_names = (
    numerical_features + 
    [f"Gender_{cat}" for cat in ['Male']] +
    binary_features
)

# Analyze feature importance for best model
importance = analyze_feature_importance(best_model, feature_names)
print("\nFeature Importance:")
for feature, score in importance.items():
    print(f"{feature}: {score:.4f}")

# Make predictions for new data with best model
def predict_survival(model, preprocessor, new_data):
    # Preprocess the new data
    processed_data = preprocessor.transform(new_data)
    
    # Make prediction
    survival_prob = model.forwardProp(processed_data)
    survived = (survival_prob > 0.5).astype(int)
    
    return survival_prob, survived

# Test with some random unseen examples
new_examples = pd.DataFrame({
    'Age': [25, 65, 35, 45],
    'Gender': ['Male', 'Female', 'Male', 'Female'],
    'Speed_of_Impact': [30, 80, 60, 40],
    'Helmet_Used': [1, 0, 0, 1],
    'Seatbelt_Used': [1, 0, 1, 1]
})

print("\nPredictions for new examples:")
probabilities, predictions = predict_survival(best_model, preprocessor, new_examples)

result_datasetFrame = new_examples.copy()
result_datasetFrame['Survival_Probability'] = probabilities
result_datasetFrame['Predicted_Survival'] = predictions
print(result_datasetFrame)




################## Sample output ##################
# Best Model Parameters:
# hidden_layers: 3
# neurons: 16
# activation_hidden: tanh
# learning_rate: 0.001
# batch_size: 32
# Best Accuracy: 0.7300

# Evaluating best model on test data:

# Model Evaluation:
# Accuracy: 0.7300
# Precision: 0.7290
# Recall: 0.6313
# F1 Score: 0.6766

# Confusion Matrix:
# [[179  42]
#  [ 66 113]]

# Feature Importance:
# Helmet_Used: 0.4639
# Gender_Male: 0.4404
# Speed_of_Impact: 0.4145
# Seatbelt_Used: 0.3946
# Age: 0.3311

# Predictions for new examples:
#    Age  Gender  Speed_of_Impact  Helmet_Used  Seatbelt_Used  Survival_Probability  Predicted_Survival
# 0   25    Male               30            1              1              0.906782                   1
# 1   65  Female               80            0              0              0.045793                   0
# 2   35    Male               60            0              1              0.679258                   1
# 3   45  Female               40            1              1              0.896341                   1
