# 🌱 Plant Disease Detection using Swin Transformer

A deep learning-based plant disease detection system using Swin Transformer architecture, trained on the PlantVillage dataset. This project provides both a trained model and a web application for real-time plant disease classification.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Plant diseases pose a significant threat to global food security. Early detection and accurate classification of plant diseases are crucial for effective crop management. This project implements a state-of-the-art Swin Transformer model for automated plant disease detection from leaf images.

### Key Features:
- **High Accuracy**: 99.76% accuracy on the PlantVillage dataset
- **Real-time Classification**: Web-based interface for instant disease detection
- **38 Disease Categories**: Covers multiple plant species and disease types
- **User-friendly Interface**: Streamlit web application with intuitive design

## 🚀 Features

- ✅ **Swin Transformer Architecture**: State-of-the-art vision transformer
- ✅ **Web Application**: Streamlit-based interface for easy interaction
- ✅ **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- ✅ **Pre-trained Model**: Ready-to-use trained model (332MB)
- ✅ **Multiple Plant Support**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

## 🏗️ Model Architecture

### Swin Transformer Base
- **Architecture**: Swin Transformer Base Patch4 Window7 224
- **Parameters**: 86.8M
- **Input Size**: 224×224 RGB images
- **Output**: 38-class classification
- **Pre-training**: ImageNet-1K pre-trained weights

### Training Details
- **Optimizer**: AdamW
- **Learning Rate**: 3e-5
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 5
- **Batch Size**: 32
- **Device**: CPU/GPU compatible

## 📊 Dataset

### PlantVillage Dataset
- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Total Images**: 54,305
- **Classes**: 38 disease categories
- **Plant Species**: 14 different plant types
- **Image Format**: Color images (RGB)

### Supported Plants and Diseases:
- **Apple**: Apple scab, Black rot, Cedar apple rust, Healthy
- **Blueberry**: Healthy
- **Cherry**: Powdery mildew, Healthy
- **Corn**: Cercospora leaf spot, Common rust, Northern leaf blight, Healthy
- **Grape**: Black rot, Esca, Leaf blight, Healthy
- **Orange**: Huanglongbing (Citrus greening)
- **Peach**: Bacterial spot, Healthy
- **Pepper**: Bacterial spot, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Raspberry**: Healthy
- **Soybean**: Healthy
- **Squash**: Powdery mildew
- **Strawberry**: Leaf scorch, Healthy
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, Healthy

## 📈 Performance

### Evaluation Results
- **Accuracy**: 99.76%
- **Precision**: 99.77%
- **Recall**: 99.76%
- **F1-Score**: 99.76%
- **Top-5 Accuracy**: 100.00%

### Comparison with Literature
| Model | Accuracy | Reference |
|-------|----------|-----------|
| CNN (ResNet-50) | 98.2% | [PlantVillage Paper] |
| Vision Transformer | 97.5% | [ViT Paper] |
| **Swin Transformer (Ours)** | **99.76%** | This Work |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd plant-disease-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   ```bash
   # The PlantVillage dataset should be placed in:
   # data/PlantVillage-Dataset/raw/color/
   ```

## 🚀 Usage

### 1. Training the Model
```bash
# Activate virtual environment
source venv/bin/activate

# Run training
python src/train.py
```

### 2. Evaluating the Model
```bash
# Run evaluation
python src/evaluate.py
```

### 3. Running the Web Application
```bash
# Start Streamlit app
streamlit run app/app.py
```

The web application will open in your browser at `http://localhost:8501`

### 4. Testing the Model
```bash
# Test model loading and inference
python src/test_model.py
```

## 📁 Project Structure

```
plant-disease-detection/
├── app/
│   └── app.py                 # Streamlit web application
├── data/
│   └── PlantVillage-Dataset/  # Dataset directory
│       └── raw/
│           └── color/         # Color images
├── models/
│   └── epoch_5.pt            # Trained model (332MB)
├── results/                  # Evaluation results
│   ├── confusion_matrix.png
│   ├── performance_analysis.png
│   ├── classification_report.csv
│   └── evaluation_summary.csv
├── src/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── test_model.py         # Model testing script
├── venv/                     # Virtual environment
├── requirements.txt          # Python dependencies
├── setup_structure.py        # Project setup script
└── README.md                 # This file
```

## 📊 Results

### Model Performance
The trained Swin Transformer model achieves excellent performance across all metrics:

- **Overall Accuracy**: 99.76%
- **Class-wise Performance**: Consistent high accuracy across all 38 classes
- **Confusion Matrix**: Minimal misclassifications
- **Confidence Distribution**: High confidence predictions for correct classifications

### Visualizations
The evaluation generates several visualizations:
- **Confusion Matrix**: Shows classification accuracy for each class
- **Performance Analysis**: Confidence distribution, accuracy vs confidence, class-wise accuracy
- **Classification Report**: Detailed metrics for each disease class

## 🔧 Technical Details

### Preprocessing
- **Image Resizing**: 224×224 pixels
- **Normalization**: ImageNet mean and std values
- **Data Augmentation**: None (for consistency with evaluation)

### Model Details
- **Framework**: PyTorch
- **Architecture**: Swin Transformer Base
- **Pre-training**: ImageNet-1K
- **Fine-tuning**: PlantVillage dataset

### Hardware Requirements
- **Training**: CPU/GPU (tested on CPU)
- **Inference**: CPU (real-time capable)
- **Memory**: 8GB+ RAM recommended
- **Storage**: 5GB+ for dataset and model

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PlantVillage Dataset**: For providing the comprehensive plant disease dataset
- **Swin Transformer**: Microsoft Research for the transformer architecture
- **PyTorch**: For the deep learning framework
- **Streamlit**: For the web application framework

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This model is for educational and research purposes. For agricultural applications, please consult with plant pathology experts for validation and interpretation of results.