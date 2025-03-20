import numpy as np

def depthwise_convolution(inputImage, kernels):
    # Get dimensions
    imgHeight, imgWidth, channels = inputImage.shape
    kernelHeight, kernelWidth, kernelChannels = kernels.shape
    
    # Check if kernel channels match input channels
    if channels != kernelChannels:
        raise ValueError("Number of channels in input and kernel must match for depthwise convolution")
    
    # Calculate output dimensions
    outputHeight = imgHeight - kernelHeight + 1
    outputWidth = imgWidth - kernelWidth + 1
    
    # Initialize output array
    output = np.zeros((outputHeight, outputWidth, channels))
    
    # Perform depthwise convolution (one filter per channel)
    for c in range(channels):
        for i in range(outputHeight):
            for j in range(outputWidth):
                # Extract patch from input channel
                patch = inputImage[i:i+kernelHeight, j:j+kernelWidth, c]
                # Apply corresponding kernel
                output[i, j, c] = np.sum(patch * kernels[:, :, c])
    
    return output

def pointwise_convolution(inputImage, filters):
    # Get dimensions
    imgHeight, imgWidth, inChannels = inputImage.shape
    filter_inChannels, outChannels = filters.shape
    
    # Check if filter input channels match input image channels
    if inChannels != filter_inChannels:
        raise ValueError("Number of input channels in filters must match input image channels")
    
    # Initialize output array
    output = np.zeros((imgHeight, imgWidth, outChannels))
    
    # Perform pointwise convolution
    for i in range(imgHeight):
        for j in range(imgWidth):
            # Extract point from input image (all channels)
            point = inputImage[i, j, :]
            # Apply all filters to this point
            for f in range(outChannels):
                output[i, j, f] = np.sum(point * filters[:, f])
    
    return output

def convolve(inputImage, kernel_or_filters, mode='depthwise'):
    if mode.lower() == 'depthwise':
        return depthwise_convolution(inputImage, kernel_or_filters)
    elif mode.lower() == 'pointwise':
        return pointwise_convolution(inputImage, kernel_or_filters)
    else:
        raise ValueError("Mode must be either 'depthwise' or 'pointwise'")

# Example usage
# if __name__ == "__main__":
#     # Create a sample 6x6x3 image (RGB)
#     inputImage = np.random.rand(6, 6, 3)
    
#     # Create a 3x3x3 kernel for depthwise convolution
#     depthwise_kernel = np.random.rand(3, 3, 3)
    
#     # Create filters for pointwise convolution (3 input channels, 5 output channels)
#     pointwise_filters = np.random.rand(3, 5)
    
#     # Perform depthwise convolution
#     depthwise_output = convolve(inputImage, depthwise_kernel, mode='depthwise')
#     print("Depthwise Convolution Output Shape:", depthwise_output.shape)
    
#     # Perform pointwise convolution
#     pointwise_output = convolve(inputImage, pointwise_filters, mode='pointwise')
#     print("Pointwise Convolution Output Shape:", pointwise_output.shape)
    
#     # Example of depthwise followed by pointwise (depthwise separable convolution)
#     depthwise_separable_output = convolve(
#         convolve(inputImage, depthwise_kernel, mode='depthwise'),
#         pointwise_filters, 
#         mode='pointwise'
#     )
#     print("Depthwise Separable Convolution Output Shape:", depthwise_separable_output.shape)
    
#     # Show first channel of each result
#     print("\nSample of Depthwise Output (first channel):")
#     print(depthwise_output[:,:,0])
    
#     print("\nSample of Pointwise Output (first channel):")
#     print(pointwise_output[:,:,0])