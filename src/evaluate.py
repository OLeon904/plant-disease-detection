import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import pandas as pd
import os
from tqdm import tqdm

def load_model_and_data():
    """Load the trained model and test dataset"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Load model
    print("🔄 Loading model...")
    model = create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=38)
    model.load_state_dict(torch.load('models/epoch_5.pt', map_location=device))
    model.to(device)
    model.eval()
    
    # Load dataset
    print("🔄 Loading dataset...")
    data_dir = "data/PlantVillage-Dataset/raw/color"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Split dataset into train/test (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"✅ Model and data loaded")
    print(f"📊 Test dataset size: {len(test_dataset)} images")
    print(f"🏷️  Number of classes: {len(dataset.classes)}")
    
    return model, device, test_loader, dataset.classes

def evaluate_model(model, device, test_loader, class_names):
    """Evaluate the model and return predictions and metrics"""
    print("🔍 Evaluating model...")
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    print(f"✅ Evaluation completed")
    print(f"📈 Accuracy: {accuracy:.4f}")
    print(f"📊 Precision: {precision:.4f}")
    print(f"📊 Recall: {recall:.4f}")
    print(f"📊 F1-Score: {f1:.4f}")
    
    return all_predictions, all_labels, all_probabilities, accuracy, precision, recall, f1

def create_confusion_matrix(all_labels, all_predictions, class_names):
    """Create and save confusion matrix visualization"""
    print("📊 Creating confusion matrix...")
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Create visualization
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Plant Disease Detection', fontsize=16, pad=20)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Confusion matrix saved to results/confusion_matrix.png")

def create_classification_report(all_labels, all_predictions, class_names):
    """Create detailed classification report"""
    print("📋 Creating classification report...")
    
    report = classification_report(all_labels, all_predictions, 
                                  target_names=class_names, output_dict=True)
    
    # Convert to DataFrame for better visualization
    df_report = pd.DataFrame(report).transpose()
    
    # Save report
    os.makedirs('results', exist_ok=True)
    df_report.to_csv('results/classification_report.csv')
    
    print("✅ Classification report saved to results/classification_report.csv")
    
    # Print summary
    print("\n📊 Classification Report Summary:")
    print(df_report.loc['weighted avg'])

def create_performance_visualizations(all_probabilities, all_labels, class_names):
    """Create additional performance visualizations"""
    print("📈 Creating performance visualizations...")
    
    # Convert to numpy arrays
    probabilities = np.array(all_probabilities)
    labels = np.array(all_labels)
    
    # Calculate confidence scores
    confidence_scores = np.max(probabilities, axis=1)
    predicted_classes = np.argmax(probabilities, axis=1)
    correct_predictions = (predicted_classes == labels)
    
    # Create confidence distribution plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(confidence_scores[correct_predictions], bins=50, alpha=0.7, label='Correct', color='green')
    plt.hist(confidence_scores[~correct_predictions], bins=50, alpha=0.7, label='Incorrect', color='red')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution')
    plt.legend()
    
    # Create accuracy vs confidence plot
    plt.subplot(2, 2, 2)
    confidence_bins = np.linspace(0, 1, 11)
    accuracies = []
    for i in range(len(confidence_bins)-1):
        mask = (confidence_scores >= confidence_bins[i]) & (confidence_scores < confidence_bins[i+1])
        if np.sum(mask) > 0:
            accuracies.append(np.mean(correct_predictions[mask]))
        else:
            accuracies.append(0)
    
    plt.plot(confidence_bins[:-1], accuracies, 'bo-')
    plt.xlabel('Confidence Score')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Confidence')
    plt.grid(True)
    
    # Create class-wise accuracy
    plt.subplot(2, 2, 3)
    class_accuracies = []
    for i in range(len(class_names)):
        mask = (labels == i)
        if np.sum(mask) > 0:
            class_accuracies.append(np.mean(correct_predictions[mask]))
        else:
            class_accuracies.append(0)
    
    plt.bar(range(len(class_names)), class_accuracies)
    plt.xlabel('Class Index')
    plt.ylabel('Accuracy')
    plt.title('Class-wise Accuracy')
    plt.xticks(range(0, len(class_names), 5))
    
    # Create top-5 accuracy
    plt.subplot(2, 2, 4)
    top5_correct = 0
    for i, (prob, label) in enumerate(zip(probabilities, labels)):
        top5_indices = np.argsort(prob)[-5:]
        if label in top5_indices:
            top5_correct += 1
    
    top5_accuracy = top5_correct / len(labels)
    plt.bar(['Top-1', 'Top-5'], [np.mean(correct_predictions), top5_accuracy])
    plt.ylabel('Accuracy')
    plt.title('Top-1 vs Top-5 Accuracy')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('results/performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Performance visualizations saved to results/performance_analysis.png")
    print(f"📊 Top-5 Accuracy: {top5_accuracy:.4f}")

def save_results(accuracy, precision, recall, f1):
    """Save evaluation results to a summary file"""
    print("💾 Saving evaluation results...")
    
    results = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [accuracy, precision, recall, f1]
    }
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/evaluation_summary.csv', index=False)
    
    print("✅ Evaluation summary saved to results/evaluation_summary.csv")
    
    # Print final summary
    print("\n" + "="*50)
    print("🎯 FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"📈 Accuracy:  {accuracy:.4f}")
    print(f"📊 Precision: {precision:.4f}")
    print(f"📊 Recall:    {recall:.4f}")
    print(f"📊 F1-Score:  {f1:.4f}")
    print("="*50)

def main():
    """Main evaluation function"""
    print("🔍 Starting Model Evaluation")
    print("="*50)
    
    # Load model and data
    model, device, test_loader, class_names = load_model_and_data()
    
    # Evaluate model
    all_predictions, all_labels, all_probabilities, accuracy, precision, recall, f1 = evaluate_model(
        model, device, test_loader, class_names
    )
    
    # Create visualizations and reports
    create_confusion_matrix(all_labels, all_predictions, class_names)
    create_classification_report(all_labels, all_predictions, class_names)
    create_performance_visualizations(all_probabilities, all_labels, class_names)
    save_results(accuracy, precision, recall, f1)
    
    print("\n🎉 Evaluation completed successfully!")
    print("📁 Check the 'results/' directory for all evaluation files.")

if __name__ == "__main__":
    main()
