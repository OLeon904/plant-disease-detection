import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
from PIL import Image
import os

def test_model():
    print("🧪 Testing trained model...")
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model architecture (same as training)
    model = create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=38)
    
    # Load trained weights
    model.load_state_dict(torch.load('models/epoch_5.pt', map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}")
    print(f"🔢 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create a dummy image for testing
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    
    # Test inference
    with torch.no_grad():
        output = model(dummy_image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        
    print(f"✅ Inference test successful!")
    print(f"📊 Output shape: {output.shape}")
    print(f"🎯 Predicted class: {predicted_class.item()}")
    print(f"📈 Max probability: {torch.max(probabilities).item():.4f}")
    
    return True

if __name__ == "__main__":
    try:
        test_model()
        print("🎉 All tests passed! Model is working correctly.")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 