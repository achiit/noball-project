import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model from local system
model_path = "/Users/achintya/Desktop/noball project/bowling_model.h5"
model = load_model(model_path)

# Define image dimensions and number of channels
image_height = 100
image_width = 100

def preprocess_image(image):
    # Convert the image to the required dimensions and normalize
    image = image.resize((image_height, image_width))
    image = np.array(image) / 255.0
    return image

def classify_ball(image):
    # Preprocess the uploaded image
    processed_image = preprocess_image(image)
    # Add an extra dimension to match the input shape expected by the model
    processed_image = np.expand_dims(processed_image, axis=0)
    # Make prediction using the loaded model
    prediction = model.predict(processed_image)
    # Map prediction to class label
    if prediction > 0.5:
        return "Normal Ball"
    else:
        return "No Ball"

# Streamlit interface
st.title("Bowling Ball Classifier")
st.write("Upload an image of a bowling ball to classify whether it's a no ball or a normal ball.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    prediction = classify_ball(image)
    st.write("Prediction:", prediction)
