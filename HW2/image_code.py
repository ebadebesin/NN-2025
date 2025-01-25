import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Directory to save generated images
output_dir = 'handwritten_images'
os.makedirs(output_dir, exist_ok=True)

# Function to generate a single image of a number
def generate_image_of_number(number, size=(20, 20)):
    # Create a blank image with white background
    image = Image.new('L', size, color=255)  # 'L' mode for grayscale
    draw = ImageDraw.Draw(image)
    
    # Load a font (or use a simple built-in font)
    font = ImageFont.load_default()

    # Get bounding box of the text to center it
    bbox = draw.textbbox((0, 0), str(number), font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    text_position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    
    # Draw the number on the image
    draw.text(text_position, str(number), fill=0, font=font)

    return image

# Generate 10 different images for each number (0-9)
for number in range(10):
    for i in range(10):
        img = generate_image_of_number(number)
        img.save(f'{output_dir}/{number}_{i}.png')

# Code to create test images (unlabeled) for testing the perceptron
test_images_dir = 'test_images'
os.makedirs(test_images_dir, exist_ok=True)

for i in range(10):  # Create 10 random test images
    random_number = np.random.randint(0, 10)
    img = generate_image_of_number(random_number)
    img.save(f'{test_images_dir}/test_image_{i}.png')



# from PIL import Image, ImageDraw, ImageFont
# import random
# import os

# def generate_images(output_dir, num_images=10, image_size=(20, 20)):
#     """
#     Generate grayscale images of handwritten digits 0-9.

#     :param output_dir: Directory to save the images.
#     :param num_images: Number of images to generate per digit.
#     :param image_size: Size of the images (width, height).
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     font = ImageFont.load_default()  # Default font (adjustable for handwritten-like appearance)

#     for digit in range(10):
#         digit_dir = os.path.join(output_dir, str(digit))
#         os.makedirs(digit_dir, exist_ok=True)

#         for i in range(num_images):
#             # Create blank image
#             img = Image.new("L", image_size, color=255)
#             draw = ImageDraw.Draw(img)

#             # Randomize position slightly
#             x_offset = random.randint(0, 5)
#             y_offset = random.randint(0, 5)

#             # Draw digit
#             draw.text((x_offset, y_offset), str(digit), font=font, fill=random.randint(0, 150))

#             # Save the image
#             img.save(os.path.join(digit_dir, f"{digit}_{i}.png"))

# def generate_test_images(output_dir, num_images=10, image_size=(20, 20)):
#     """
#     Generate unlabeled grayscale test images.

#     :param output_dir: Directory to save the test images.
#     :param num_images: Number of test images to generate.
#     :param image_size: Size of the images (width, height).
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     for i in range(num_images):
#         img = Image.new("L", image_size, color=255)
#         draw = ImageDraw.Draw(img)

#         # Randomly choose a digit to draw
#         digit = random.randint(0, 9)

#         # Randomize position slightly
#         x_offset = random.randint(0, 5)
#         y_offset = random.randint(0, 5)

#         # Draw digit
#         draw.text((x_offset, y_offset), str(digit), font=ImageFont.load_default(), fill=random.randint(0, 150))

#         # Save the image
#         img.save(os.path.join(output_dir, f"test_{i}.png"))


# # Generate the images
# generate_images("training_images", num_images=10)
# generate_test_images("test_images", num_images=10)
