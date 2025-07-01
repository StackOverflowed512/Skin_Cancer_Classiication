# app/app.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

# --- Page Configuration ---
st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🔬",
    layout="centered"
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Load the trained model from the specified path."""
    model_path = os.path.join(config.SAVED_MODELS_DIR, "best_model.h5")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please train the model first.")
        return None
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# --- UI Elements ---
st.title("🔬 Skin Cancer Classification")
st.write(
    "Upload a dermoscopy image of a skin lesion, and the model will predict "
    "the most likely type of skin cancer based on the HAM10000 dataset categories."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # 1. Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # 2. Preprocess the image for the model
    image_resized = image.resize((config.IMG_SIZE, config.IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = tf.keras.applications.resnet50.preprocess_input(img_array)

    # 3. Make prediction
    prediction = model.predict(processed_img)
    predicted_class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    predicted_class_label = config.INT_TO_CLASS[predicted_class_index]
    full_class_name = config.CLASSES[predicted_class_label]

    # 4. Display the result
    st.subheader("Prediction Result")
    st.success(f"**Predicted Class:** {full_class_name} (`{predicted_class_label}`)")
    st.info(f"**Confidence:** {confidence:.2f}%")

    st.markdown("""
    ---
    **Disclaimer:** This is an academic project and **not a medical diagnosis tool**.
    The predictions are based on a machine learning model and should be verified by a qualified dermatologist.
    """)