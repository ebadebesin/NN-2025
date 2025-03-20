import numpy as np

# Convolution function
def convolution(input, filter):
    # Get dimensions of input and filter
    inputHeight, inputWidth = input.shape
    filterHeight, filterWidth = filter.shape

    # Calculate dimensions of output
    outputHeight = inputHeight - filterHeight + 1
    outputWidth = inputWidth - filterWidth + 1
    
    # Initialize output
    output = np.zeros((outputHeight, outputWidth))
    
    # Loop through the input and apply convolution at each position 
    for i in range(outputHeight):
        for j in range(outputWidth):
            # Get the position in the input that corresponds to the current output position
            position = input[i:i+filterHeight, j:j+filterWidth]
            # Apply convolution at the current position
            output[i, j] = np.sum(position * filter)  # Hadamard product and sum
    
    return output

# Example 6x5 input
input = np.array([
                [2,4,7,6,5],
                [9,7,1,2,6],
                [8,3,4,5,7],
                [4,3,3,8,4],
                [5,2,1,1,2],
                [6,7,8,9,3]
                ])
# Example 3x3 filter
filter = np.array([
                [1, 0, -1],
                [1, 0, -1],
                [1, 0, -1]
                ])


# Apply convolution
output = convolution(input, filter)
print("Convoluted Output:")
print(output)


# Sample output:

# Convoluted Output:
# [[ 7.  1. -6.]
#  [13. -2. -9.]
#  [ 9. -6. -5.]
#  [ 3. -6.  3.]]