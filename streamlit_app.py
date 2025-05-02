# streamlit_app.py

import streamlit as st
import pickle
import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load model and processor
@st.cache_resource
def load_blip_and_xgb():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")
    with open("xgb_bean_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    return processor, blip_model.to(device), xgb_model

# Constants
label_map = {
    0: "a leaf with angular leaf spot",
    1: "a leaf with bean rust disease",
    2: "a healthy bean leaf"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
processor, blip_model, xgb_model = load_blip_and_xgb()

# Streamlit UI
st.title("🌿 Bean Disease Classifier with BLIP + XGBoost")
st.write("Upload a bean leaf image to predict its disease class.")

uploaded_file = st.file_uploader("Upload an image of a bean leaf", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process image and extract features
    question = "What disease does this bean plant have?"
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)

    with torch.no_grad():
        vision_outputs = blip_model.vision_model(inputs.pixel_values)
        pooled_output = vision_outputs.pooler_output.squeeze().cpu().numpy()  # (768,)

    # Predict with XGBoost
    pred = xgb_model.predict([pooled_output])[0]
    pred_label = label_map[pred]

    st.success(f"🧠 Prediction: **{pred_label}**")
