import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
from PIL import Image

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define data directory on Google Drive
data_dir = '/content/drive/MyDrive/footcontact'

# Check if the data directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"The directory '{data_dir}' does not exist. Please ensure you have provided the correct path.")

# Define image dimensions and number of channels
image_height = 100
image_width = 100
num_channels = 3  # Assuming RGB images

# Load and preprocess training data
image_paths = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir) if file_name.endswith('.jpg')]

# Load and preprocess images
images = []
for path in image_paths:
    img = Image.open(path)
    img = img.convert('RGB')  # Convert to RGB mode (in case image is grayscale or has alpha channel)
    img = img.resize((image_height, image_width))
    img = np.array(img) / 255.0  # Normalize pixel values
    images.append(img)

# Convert list of images to numpy array
images = np.array(images)

# Define labels (assuming all images represent instances of a "no ball")
labels = np.zeros(len(images))

# Define your CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(images, labels, epochs=10)

# Save the model
model.save('/content/drive/MyDrive/bowling_model.h5')
