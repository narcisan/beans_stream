# generate_xgb_model.py

import os
import pickle
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import torch

# Step 1: Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 2: Load Beans dataset
dataset = load_dataset("beans", split="train[:500]")

# Step 3: Feature Extraction
def extract_features(data, processor, model, device):
    model.eval()
    X, y = [], []
    with torch.no_grad():
        for sample in tqdm(data, desc="Extracting features"):
            image = sample['image']
            label = sample['labels']
            inputs = processor(images=image, return_tensors="pt").to(device)
            vision_outputs = model.vision_model(inputs.pixel_values)
            pooled_output = vision_outputs.pooler_output
            X.append(pooled_output.squeeze().cpu().numpy())
            y.append(label)
    return np.array(X), np.array(y)

X, y = extract_features(dataset, processor, model, device)

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 5: Train XGBoost Model for Multi-Class Classification
xgb_model = XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Step 6: Save model to disk
with open("xgb_bean_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

print("✅ Model saved as xgb_bean_model.pkl")