import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
import time

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Paths
data_dir = "data/PlantVillage-Dataset/raw/color"  # adjust this to your dataset location
model_save_path = "models/epoch_5.pt"

print(f"📁 Loading dataset from: {data_dir}")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset
print("🔄 Loading dataset...")
start_time = time.time()
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
print(f"✅ Dataset loaded in {time.time() - start_time:.2f}s")
print(f"📊 Dataset size: {len(dataset)} images")
print(f"🏷️  Classes: {dataset.classes}")
print(f"📈 Number of classes: {len(dataset.classes)}")

train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
print(f"📦 DataLoader created with batch size 32")

# Create model
print("🔄 Creating model...")
start_time = time.time()
model = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=len(dataset.classes))
model.to(device)
print(f"✅ Model created in {time.time() - start_time:.2f}s")
print(f"🔢 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

# Training loop (5 epochs for demo)
print("\n🎯 Starting training...")
model.train()
for epoch in range(5):
    epoch_start = time.time()
    total_loss = 0
    num_batches = 0
    
    print(f"\n📅 Epoch [{epoch+1}/5]")
    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_start = time.time()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:  # Print every 10 batches
            batch_time = time.time() - batch_start
            print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f} - Time: {batch_time:.2f}s")
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / num_batches
    print(f"✅ Epoch [{epoch+1}/5] completed in {epoch_time:.2f}s - Avg Loss: {avg_loss:.4f}")

# Save the model
print("\n💾 Saving model...")
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"✅ Model saved to {model_save_path}")
print("🎉 Training completed successfully!")
