import numpy as np

def min_max_normalization(data):
    # scale data between 0 and 1
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val + 1e-8)

def z_score_normalization(data):
    # scale data to have a mean of 0 and standard deviation of 1
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-8)

# # Example usage
# data = np.array([[150, 300, 200], [100, 200, 450], [500, 520, 540]])

# # Min-Max Normalization
# normalized_data_min_max = min_max_normalization(data)
# print("Min-Max Normalized Data:")
# print(normalized_data_min_max)

# # Z-score Normalization
# normalized_data_z_score = z_score_normalization(data)
# print("\nZ-score Normalized Data:")
# print(normalized_data_z_score)
