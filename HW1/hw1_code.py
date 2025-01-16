# Class to simulate neuron with McCulloch-Pitts model

class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def activation_function(self, x):
        return 1 if x >= self.threshold else 0

    def output(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("Number of inputs must match number of weights")
        
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs))
        return self.activation_function(weighted_sum)

# Example
if __name__ == "__main__":
    # Initialize weights and threshold
    weights = [0.5, 0.5]
    threshold = 1

    # Create neuron
    neuron = McCullochPittsNeuron(weights, threshold)

    # Define inputs
    inputs = [1, 1]

    # Output
    output = neuron.output(inputs)
    print(f"Inputs: {inputs}") 
    print (f"Weights: {weights}")
    print(f"Threshold: {threshold}")

    print(f"Output: {output}")