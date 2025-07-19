import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
from PIL import Image
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model architecture
    model = create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=38)
    
    # Load trained weights
    model_path = "models/epoch_5.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    else:
        st.error(f"Model file not found: {model_path}")
        return None, device

def get_class_names():
    """Return the class names for the PlantVillage dataset"""
    return [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

def preprocess_image(image):
    """Preprocess the uploaded image"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_disease(model, device, image_tensor):
    """Predict disease from preprocessed image"""
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        confidence = torch.max(probabilities).item()
    return predicted_class.item(), probabilities.cpu().numpy()[0], confidence

def main():
    # Header
    st.markdown('<h1 class="main-header">🌱 Plant Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Upload a leaf image to detect plant diseases using AI</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, device = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        return
    
    # Sidebar
    st.sidebar.markdown("## 📋 About")
    st.sidebar.markdown("""
    This application uses a **Swin Transformer** model trained on the PlantVillage dataset to detect plant diseases.
    
    **Supported Plants:**
    - Apple, Blueberry, Cherry
    - Corn, Grape, Orange, Peach
    - Pepper, Potato, Raspberry
    - Soybean, Squash, Strawberry, Tomato
    
    **Model Details:**
    - Architecture: Swin Transformer Base
    - Parameters: 86.8M
    - Classes: 38 disease categories
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a leaf image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a plant leaf"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess and predict
            if st.button("🔍 Detect Disease", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    image_tensor = preprocess_image(image)
                    
                    # Get prediction
                    predicted_class, probabilities, confidence = predict_disease(model, device, image_tensor)
                    
                    # Get class names
                    class_names = get_class_names()
                    predicted_disease = class_names[predicted_class]
                    
                    # Display results
                    with col2:
                        st.markdown('<h2 class="sub-header">🔬 Analysis Results</h2>', unsafe_allow_html=True)
                        
                        # Result box
                        st.markdown(f"""
                        <div class="result-box">
                            <h3>🌿 Detected: {predicted_disease}</h3>
                            <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Top 5 predictions
                        st.markdown("### 📊 Top 5 Predictions")
                        top_5_indices = np.argsort(probabilities)[-5:][::-1].tolist()
                        
                        for i, idx in enumerate(top_5_indices):
                            prob = float(probabilities[idx])
                            disease_name = class_names[idx]
                            bar_color = "#2E8B57" if i == 0 else "#90EE90"
                            
                            st.markdown(f"""
                            <div style="margin: 10px 0;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                    <span>{disease_name}</span>
                                    <span>{prob:.2%}</span>
                                </div>
                                <div style="background-color: #f0f0f0; height: 20px; border-radius: 10px; overflow: hidden;">
                                    <div style="background-color: {bar_color}; height: 100%; width: {prob*100}%; transition: width 0.3s;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Health status
                        if "healthy" in predicted_disease.lower():
                            st.success("✅ This plant appears to be healthy!")
                        else:
                            st.warning("⚠️ Disease detected. Consider consulting a plant expert.")
    
    # Instructions
    if uploaded_file is None:
        with col2:
            st.markdown("### 📖 How to Use")
            st.markdown("""
            1. **Upload an image** of a plant leaf in the left panel
            2. **Click 'Detect Disease'** to analyze the image
            3. **View results** including:
               - Detected disease or health status
               - Confidence level
               - Top 5 predictions
            
            **Tips for best results:**
            - Use clear, well-lit images
            - Focus on the leaf area
            - Avoid shadows or reflections
            - Ensure the leaf is clearly visible
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Built with Streamlit • Powered by Swin Transformer • PlantVillage Dataset</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 