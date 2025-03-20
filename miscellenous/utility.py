import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nn import BackPropagation

class NeuralNetworkUtility:
    """Utility class for neural network training, evaluation and visualization."""
    
    def __init__(self, model):
        self.model = model
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        self.scaler = StandardScaler()
    
    def preprocess_data(self, X, y, test_size=0.2, random_state=42):
        # Split data first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Normalize input features using StandardScaler
        # Fit only on training data to prevent data leakage
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def train_with_validation(self, X_train, y_train, X_val, y_val, 
                             epochs=100, learning_rate=0.001, 
                             batch_size=32, clip_value=1.0, 
                             verbose=True, verbose_interval=10):

        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Mini-batch training
            if batch_size:
                indices = np.random.permutation(n_samples)
                num_batches = int(np.ceil(n_samples / batch_size))
                
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, n_samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    X_batch = X_train[batch_indices]
                    y_batch = y_train[batch_indices]
                    
                    # Forward pass with safety checks
                    try:
                        output_batch = self.model.forwardProp(X_batch)
                        batch_loss = self.model.loss_function(y_batch, output_batch)
                        epoch_loss += batch_loss * len(batch_indices)
                        
                        
                        # Backpropagation with lower learning rate
                        BackPropagation.calc(self.model, X_batch, y_batch, learning_rate)
                        
                        # Gradient clipping
                        for layer in self.model.layers:
                            np.clip(layer.weights, -clip_value, clip_value, out=layer.weights)
                            np.clip(layer.biases, -clip_value, clip_value, out=layer.biases)
                            
                            # Replace any NaN weights or biases with zeros
                            layer.weights = np.nan_to_num(layer.weights, nan=0.0)
                            layer.biases = np.nan_to_num(layer.biases, nan=0.0)
                            
                    except Exception as e:
                        print(f"Error in batch {i}: {e}")
                        continue
                
                # Calculate average loss for the epoch
                epoch_loss /= n_samples
                
            else:
                # Full batch update (using the same safety measures)
                try:
                    output_train = self.model.forwardProp(X_train)
                    epoch_loss = self.model.loss_function(y_train, output_train)
                    
                    if not np.isnan(epoch_loss):
                        BackPropagation.calc(self.model, X_train, y_train, learning_rate)
                        
                        # Gradient clipping
                        for layer in self.model.layers:
                            np.clip(layer.weights, -clip_value, clip_value, out=layer.weights)
                            np.clip(layer.biases, -clip_value, clip_value, out=layer.biases)
                            
                            # Replace any NaN values
                            layer.weights = np.nan_to_num(layer.weights, nan=0.0)
                            layer.biases = np.nan_to_num(layer.biases, nan=0.0)
                    
                except Exception as e:
                    print(f"Error in full batch update: {e}")
            
            # Calculate and record metrics
            try:
                # Safe evaluation
                train_output = self.model.forwardProp(X_train)
                val_output = self.model.forwardProp(X_val)
                
                train_loss = self.safe_loss_calculation(y_train, train_output)
                val_loss = self.safe_loss_calculation(y_val, val_output)
                
                # Calculate accuracy
                train_accuracy = self.calculate_accuracy(train_output, y_train)
                val_accuracy = self.calculate_accuracy(val_output, y_val)
                
                # Record history
                self.history['loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['accuracy'].append(train_accuracy)
                self.history['val_accuracy'].append(val_accuracy)
                
                if verbose and (epoch % verbose_interval == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch}/{epochs}, "
                          f"Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                          f"Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
                    
            except Exception as e:
                print(f"Error calculating metrics for epoch {epoch}: {e}")
                continue
                
        return self.history
    
    def safe_loss_calculation(self, y_true, y_pred):
        try:
            # Clip prediction values to avoid log(0) or division by zero
            epsilon = 1e-15
            y_pred_safe = np.clip(y_pred, epsilon, 1 - epsilon)
            
            # Calculate loss
            loss = self.model.loss_function(y_true, y_pred_safe)
            
            # Return a default high value if NaN occurs
            return 999.9 if np.isnan(loss) else loss
        except:
            return 999.9  # Return a default value on error
    
    def calculate_accuracy(self, outputs, targets):
        try:
            # Handle NaN values
            outputs = np.nan_to_num(outputs, nan=0.5)
            
            if outputs.shape[1] > 1:  # Multi-class
                predicted_classes = np.argmax(outputs, axis=1)
                true_classes = np.argmax(targets, axis=1)
            else:  # Binary
                predicted_classes = (outputs > 0.5).astype(int).flatten()
                true_classes = targets.flatten().astype(int)
                
            return np.mean(predicted_classes == true_classes)
        except:
            return 0.5  # Return default accuracy on error
    
    def evaluate(self, X_test, y_test):
        try:
            output = self.model.forwardProp(X_test)
            output = np.nan_to_num(output, nan=0.5)  # Replace NaNs
            
            loss = self.safe_loss_calculation(y_test, output)
            accuracy = self.calculate_accuracy(output, y_test)
            
            # Calculate predictions
            if output.shape[1] > 1:  # Multi-class
                predictions = np.argmax(output, axis=1)
                true_labels = np.argmax(y_test, axis=1)
            else:  # Binary
                predictions = (output > 0.5).astype(int).flatten()
                true_labels = y_test.flatten().astype(int)
            
            # Create confusion matrix
            from sklearn.metrics import confusion_matrix
            conf_matrix = confusion_matrix(true_labels, predictions)
            
            # Return evaluation metrics
            return {
                'loss': loss,
                'accuracy': accuracy,
                'predictions': predictions,
                'confusion_matrix': conf_matrix
            }
        except Exception as e:
            print(f"Error in evaluation: {e}")
            # Return default values
            return {
                'loss': 999.9,
                'accuracy': 0.5,
                'predictions': np.zeros(len(y_test)),
                'confusion_matrix': np.array([[0, 0], [0, 0]])
            }
    
    def predict(self, X_new):
        try:
            # Apply the same preprocessing as training data
            X_scaled = self.scaler.transform(X_new)
            
            # Make prediction
            output = self.model.forwardProp(X_scaled)
            
            # Handle NaN values
            output = np.nan_to_num(output, nan=0.5)
            
            return output
        except Exception as e:
            print(f"Error in prediction: {e}")
            # Return default prediction
            return np.full((X_new.shape[0], 1), 0.5)