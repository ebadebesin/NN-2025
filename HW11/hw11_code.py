import numpy as np

def create_mini_batches(X, y, batch_size):
    m = X.shape[0]
    indices = np.random.permutation(m)
    X_ = X[indices]
    Y_ = y[indices]
    
    mini_batches = []
    for i in range(0, m, batch_size):
        X_mini = X_[i:i + batch_size]
        y_mini = Y_[i:i + batch_size]
        mini_batches.append((X_mini, y_mini))
    
    return mini_batches

# For Example
class MiniBatchTrainer:
    def __init__(self, model, loss_function, learning_rate=0.01, batch_size=32, epochs=100):
        self.model = model
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, X, y):
        for epoch in range(self.epochs):
            mini_batches = create_mini_batches(X, y, self.batch_size)
            for X_mini, y_mini in mini_batches:
                # Forward propagation
                predictions = self.model.forwardProp(X_mini)
                
                # Compute loss
                loss = self.loss_function(y_mini, predictions)
                
                # Backward propagation and update weights
                self.model.backwardProp(y_mini, self.learning_rate)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

