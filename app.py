import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("🧠 MNIST Digit Recognition")
st.markdown("Upload a **28x28 grayscale image** of a handwritten digit and let the model predict it.")

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")  # or use "sgd_model.pkl"
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Image upload
uploaded_file = st.file_uploader("Upload a digit image (28x28 PNG or JPEG)", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    image = ImageOps.invert(image)  # MNIST digits are white on black
    st.image(image, caption="Your Input", width=150)

    # Convert image to model-ready format
    img_array = np.array(image).reshape(1, -1)
    img_scaled = scaler.transform(img_array.astype(np.float64))

    prediction = model.predict(img_scaled)[0]
    st.success(f"✅ Predicted Digit: **{prediction}**")
else:
    st.warning("Please upload an image to get a prediction.")   