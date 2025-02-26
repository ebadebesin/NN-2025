# Develop a prototype of the training, validation, and testing sets of your choice for the future training of your neural network.
# The term “prototype” means that the sets may be quite limited by number of images, but the proportions of images in them should be maintained as required

import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# Function to load images
def load_images(image_dir, labels=True):
    images = []
    labels_list = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            img = Image.open(os.path.join(image_dir, filename)).convert('L')
            img = img.resize((20, 20))
            img_array = np.array(img).flatten() / 255.0  # Normalize pixel values
            images.append(img_array)

            if labels:
                label = int(filename.split('_')[0]) # Extract the digit from the filename
                labels_list.append(label)

    images = np.array(images)
    labels_array = np.eye(10)[labels_list] if labels else None
    return images, labels_array

# Load the dataset
train_images, train_labels = load_images('handwritten_images')

# Split dataset (80% Train, 10% Validation, 10% Test)
X_train, X_temp, y_train, y_temp = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print dataset sizes to verify % proportions 
print(f"Training Set: {len(X_train)} images")
print(f"Validation Set: {len(X_val)} images")
print(f"Testing Set: {len(X_test)} images")
print("Dataset split successfully.")
