import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from timm import create_model
from torchvision import datasets
import os

# App Title
st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf image and get a disease classification result.")

# Dataset path
dataset_path = "data/PlantVillage-Dataset/raw/color"

# Load class names
try:
    dataset = datasets.ImageFolder(dataset_path)
    num_classes = len(dataset.classes)
    class_names = dataset.classes
    st.sidebar.success(f"✅ Detected {num_classes} classes.")
except Exception as e:
    num_classes = 1
    class_names = ["Unknown"]
    st.sidebar.error(f"❌ Failed to load class labels: {e}")

# Define image preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model
model_path = "models/epoch_5.pt"
model = create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=num_classes)

model_loaded = False
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        model_loaded = True
        st.sidebar.success("✅ Model loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model: {e}")
else:
    st.sidebar.warning("⚠️ Model file not found. Running in dummy mode.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    input_tensor = transform(image).unsqueeze(0)  # add batch dimension

    if model_loaded:
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_probs, top_idxs = torch.topk(probabilities, 3)

            st.success("🧠 Top Predictions:")
            for i in range(3):
                st.write(f"{class_names[top_idxs[i]]}: {top_probs[i].item():.2%}")
    else:
        st.warning("🧪 Dummy Mode: No model loaded. Prediction unavailable.")
