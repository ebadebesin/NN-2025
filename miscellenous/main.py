import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Import your classes
from nn import Activation
from nn import LossFunction
from nn import Layer
from nn import DeepNeuralNetwork
from nn import BackPropagation
from miscellenous.utility import NeuralNetworkUtility

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def load_accident_dataset(filepath=None):
    # Sample data
    sample_data = """Age,Gender,Speed_of_Impact,Helmet_Used,Seatbelt_Used,Survived
56,Female,27,No,No,1
69,Female,46,No,Yes,1
46,Male,46,Yes,Yes,0
32,Male,117,No,Yes,0
45,Female,32,No,Yes,1
22,Male,87,No,No,0
31,Female,54,No,Yes,1
18,Male,104,Yes,No,0
62,Female,38,No,Yes,1
51,Male,71,No,No,0
27,Female,26,No,Yes,1
42,Male,92,Yes,No,0
39,Female,45,No,Yes,1
55,Male,36,No,Yes,1
33,Male,98,No,No,0
48,Female,42,No,Yes,1
28,Male,74,Yes,No,0
61,Female,28,No,Yes,1
25,Male,115,No,No,0
37,Female,39,No,Yes,1"""
    
    try:
        # Try to load from file if provided
        if filepath:
            df = pd.read_csv(filepath)
        else:
            # Use sample data if no file provided
            from io import StringIO
            df = pd.read_csv(StringIO(sample_data))
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using sample data instead.")
        from io import StringIO
        df = pd.read_csv(StringIO(sample_data))
    
    # Extract features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived'].values.reshape(-1, 1)
    
    return X, y, df.columns.tolist()

def preprocess_accident_data(X):

    # Handle categorical features
    X_processed = X.copy()
    
    # Convert Gender to numerical (0 for Male, 1 for Female)
    X_processed['Gender'] = X_processed['Gender'].map({'Male': 0, 'Female': 1})
    
    # Convert Yes/No columns to numerical (0 for No, 1 for Yes)
    X_processed['Helmet_Used'] = X_processed['Helmet_Used'].map({'No': 0, 'Yes': 1})
    X_processed['Seatbelt_Used'] = X_processed['Seatbelt_Used'].map({'No': 0, 'Yes': 1})
    
    # Convert to numpy array
    return X_processed.values.astype(float)

def create_model(input_size):

    # Define network architecture for binary classification
    # Smaller network to reduce overfitting
    layer_sizes = [input_size, 21, 14, 7, 1]  # Input -> smaller hidden layers -> binary output
    
    # Use ReLU for hidden layers and sigmoid for output
    activations = [Activation.relu, Activation.relu, Activation.relu, Activation.sigmoid]
    
    # Binary cross-entropy is appropriate for survival prediction
    loss_function = LossFunction.binary_cross_entropy
    
    # Create model
    model = DeepNeuralNetwork(layer_sizes, activations, loss_function)
    model.loss_derivative = LossFunction.binary_cross_entropy_derivative
    
    return model

def plot_training_history(history):

    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], 'b-', label='Training Loss')
    plt.plot(history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(history['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(conf_matrix):

    class_names = ["Not Survived", "Survived"]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def main():
    # Load dataset
    print("Loading and preprocessing dataset...")
    X_df, y, feature_names = load_accident_dataset("Lab2/accident.csv")  # Load sample data
    X = preprocess_accident_data(X_df)
    
    print(f"Dataset shape: {X.shape} features, {y.shape} labels")
    print(f"Features: {feature_names[:-1]}")  # Exclude 'Survived' from features list
    
    # Create model
    print("\nCreating neural network model...")
    model = create_model(X.shape[1])
    
    # Create the utility and preprocess data
    print("\nPreparing data...")
    nn_utility = NeuralNetworkUtility(model)
    X_train, X_test, y_train, y_test = nn_utility.preprocess_data(X, y)
    
    # Train the model with validation
    print("\nTraining model...")
    history = nn_utility.train_with_validation(
        X_train, y_train, 
        X_test, y_test,
        epochs=50,                # Reduced number of epochs
        learning_rate=0.001,      # Lower learning rate for stability
        batch_size=4,             # Small batch size for small dataset
        clip_value=0.5,           # Lower clip value to prevent large gradients
        verbose=True,
        verbose_interval=10
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    eval_results = nn_utility.evaluate(X_test, y_test)
    print(f"Test Loss: {eval_results['loss']:.4f}")
    print(f"Test Accuracy: {eval_results['accuracy']:.4f}")
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(nn_utility.history)
    
    # Plot confusion matrix
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(eval_results['confusion_matrix'])
    
    # Make predictions on new cases
    print("\nExample predictions:")
    example_cases = [
        {"Age": 25, "Gender": "Male", "Speed_of_Impact": 100, "Helmet_Used": "No", "Seatbelt_Used": "No"},
        {"Age": 45, "Gender": "Female", "Speed_of_Impact": 35, "Helmet_Used": "No", "Seatbelt_Used": "Yes"},
        {"Age": 30, "Gender": "Male", "Speed_of_Impact": 60, "Helmet_Used": "Yes", "Seatbelt_Used": "Yes"}
    ]
    
    for case in example_cases:
        example_df = pd.DataFrame([case])
        example_processed = preprocess_accident_data(example_df)
        
        # Make prediction
        prediction = nn_utility.predict(example_processed)
        survival_prob = prediction[0][0]
        survived = survival_prob > 0.5
        
        print(f"\nCase: {case}")
        print(f"Survival probability: {survival_prob:.2%}")
        print(f"Prediction: {'Survived' if survived else 'Not Survived'}")

if __name__ == "__main__":
    main()