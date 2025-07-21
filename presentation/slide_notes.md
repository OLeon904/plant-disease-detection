# Slide Templates for Video Presentation
## Plant Disease Detection using Swin Transformer

---

## 🎨 **SLIDE DESIGN GUIDELINES**

### **Color Scheme:**
- **Primary Green:** #2E8B57 (Sea Green)
- **Secondary Green:** #228B22 (Forest Green)
- **Light Green:** #90EE90 (Light Green)
- **Background:** #F8FFF8 (Very Light Green)
- **Text:** #333333 (Dark Gray)
- **Accent:** #FF6B35 (Orange for highlights)

### **Font Recommendations:**
- **Title:** Arial Bold, 36pt
- **Subtitle:** Arial, 24pt
- **Body Text:** Arial, 18pt
- **Captions:** Arial, 14pt

---

## 📊 **SLIDE TEMPLATES**

### **Slide 1: Title Slide**
```
┌─────────────────────────────────────┐
│                                     │
│    🌱 Plant Disease Detection       │
│    using Swin Transformer           │
│                                     │
│    Deep Learning for Computer Vision│
│    Octavia Leon                     │
│    July 2024                        │
│                                     │
└─────────────────────────────────────┘
```

### **Slide 2: Problem Overview**
```
┌─────────────────────────────────────┐
│  ❌ The Problem                     │
│                                     │
│  Plant diseases threaten global     │
│  food security                      │
│                                     │
│  Traditional detection methods:     │
│  • Time-consuming (manual)          │
│  • Labor-intensive (experts)        │
│  • Subjective (human error)         │
│  • Expensive (specialized)          │
└─────────────────────────────────────┘
```

### **Slide 3: Impact Statistics**
```
┌─────────────────────────────────────┐
│  📈 Global Impact                   │
│                                     │
│  • 20-40% crop losses annually      │
│  • $220 billion economic impact     │
│  • Need for early detection         │
│  • Rapid response required          │
│                                     │
│  [Include relevant chart/graph]     │
└─────────────────────────────────────┘
```

### **Slide 4: Deep Learning Solution**
```
┌─────────────────────────────────────┐
│  🤖 Our Solution                    │
│                                     │
│  Traditional vs. Deep Learning:     │
│                                     │
│  Traditional:                       │
│  • Rule-based features              │
│  • Hand-crafted                     │
│                                     │
│  Deep Learning:                     │
│  • Automatic feature learning       │
│  • End-to-end processing            │
└─────────────────────────────────────┘
```

### **Slide 5: Swin Transformer Architecture**
```
┌─────────────────────────────────────┐
│  🏗️ Swin Transformer                │
│                                     │
│  Why Swin Transformer?              │
│                                     │
│  ✅ Hierarchical feature learning   │
│  ✅ Shifted window attention        │
│  ✅ Linear computational complexity │
│  ✅ State-of-the-art performance    │
│                                     │
│  [Include architecture diagram]     │
└─────────────────────────────────────┘
```

### **Slide 6: Model Specifications**
```
┌─────────────────────────────────────┐
│  ⚙️ Model Details                   │
│                                     │
│  Architecture: Swin Transformer Base│
│  Parameters: 86.8M                  │
│  Input: 224×224 RGB images          │
│  Output: 38 disease categories      │
│  Pre-training: ImageNet-1K          │
│                                     │
│  [Include model diagram]            │
└─────────────────────────────────────┘
```

### **Slide 7: PlantVillage Dataset**
```
┌─────────────────────────────────────┐
│  📊 Dataset: PlantVillage           │
│                                     │
│  • Images: 54,305 total             │
│  • Classes: 38 disease categories   │
│  • Plants: 14 different species     │
│  • Format: Color images (RGB)       │
│                                     │
│  [Include sample images grid]       │
└─────────────────────────────────────┘
```

### **Slide 8: Supported Plants**
```
┌─────────────────────────────────────┐
│  🌿 Supported Plants                │
│                                     │
│  Apple    Blueberry   Cherry        │
│  Corn     Grape       Orange        │
│  Peach    Pepper      Potato        │
│  Raspberry Soybean    Squash        │
│  Strawberry Tomato                  │
│                                     │
│  [Include plant icons]              │
└─────────────────────────────────────┘
```

### **Slide 9: Preprocessing Pipeline**
```
┌─────────────────────────────────────┐
│  🔧 Preprocessing                   │
│                                     │
│  Input Image → Resize → Normalize   │
│                                     │
│  • Image resizing: 224×224 pixels   │
│  • Normalization: ImageNet stats    │
│  • Data split: 80% train, 20% test  │
│                                     │
│  [Include pipeline diagram]         │
└─────────────────────────────────────┘
```

### **Slide 10: Performance Metrics**
```
┌─────────────────────────────────────┐
│  📈 Key Results                     │
│                                     │
│  Accuracy:    99.76%                │
│  Precision:   99.77%                │
│  Recall:      99.76%                │
│  F1-Score:    99.76%                │
│  Top-5 Acc:   100.00%               │
│                                     │
│  [Include performance chart]        │
└─────────────────────────────────────┘
```

### **Slide 11: Literature Comparison**
```
┌─────────────────────────────────────┐
│  📊 Comparison with Literature      │
│                                     │
│  Model              | Acc  | Params │
│  ────────────────────────────────── │
│  CNN (ResNet-50)    | 98.2%| 25.6M  │
│  Vision Transformer | 97.5%| 86.4M  │
│  Swin Transformer   |99.76%| 86.8M  │
│  (Ours)             |      |        │
│                                     │
└─────────────────────────────────────┘
```

### **Slide 12: Training Results**
```
┌─────────────────────────────────────┐
│  ⏱️ Training Results                │
│                                     │
│  • Training time: ~16 minutes       │
│  • Model size: 332MB                │
│  • Convergence: 5 epochs            │
│  • Device: CPU/GPU compatible       │
│                                     │
│  [Include training curve]           │
└─────────────────────────────────────┘
```

### **Slide 13: Architecture Overview**
```
┌─────────────────────────────────────┐
│  🏗️ Technical Implementation        │
│                                     │
│  Framework: PyTorch                 │
│  Optimizer: AdamW (lr=3e-5)         │
│  Loss Function: CrossEntropyLoss    │
│  Preprocessing: ImageNet norm       │
│  Deployment: Streamlit web app      │
│                                     │
└─────────────────────────────────────┘
```

### **Slide 14: Key Features**
```
┌─────────────────────────────────────┐
│  ✨ Key Features                    │
│                                     │
│  ✅ Real-time inference             │
│  ✅ User-friendly interface         │
│  ✅ Confidence scoring              │
│  ✅ Top-5 predictions               │
│  ✅ Health status indicators        │
│  ✅ Mobile-responsive design        │
│                                     │
└─────────────────────────────────────┘
```

### **Slide 15: Areas for Improvement**
```
┌─────────────────────────────────────┐
│  🔧 Areas for Improvement           │
│                                     │
│  1. Data Augmentation               │
│  2. Multi-modal Input               │
│  3. Real-world Testing              │
│  4. Model Compression               │
│  5. Continuous Learning             │
│                                     │
└─────────────────────────────────────┘
```

### **Slide 16: Future Applications**
```
┌─────────────────────────────────────┐
│  🚀 Future Applications             │
│                                     │
│  1. Mobile Deployment               │
│  2. Multi-language Support          │
│  3. Disease Severity Prediction     │
│  4. Treatment Recommendations       │
│  5. Agricultural Integration        │
│                                     │
└─────────────────────────────────────┘
```

### **Slide 17: Project Impact**
```
┌─────────────────────────────────────┐
│  🎯 Project Impact                  │
│                                     │
│  ✅ 99.76% accuracy achieved        │
│  ✅ State-of-the-art implementation │
│  ✅ Real-time web application       │
│  ✅ User-friendly interface         │
│  ✅ Ready for deployment            │
│                                     │
└─────────────────────────────────────┘
```

### **Slide 18: Final Message**
```
┌─────────────────────────────────────┐
│  🙏 Thank You                       │
│                                     │
│  This project demonstrates the      │
│  potential of deep learning in      │
│  agricultural applications          │
│                                     │
│  Practical solution for automated   │
│  plant disease detection            │
│                                     │
│  Questions?                         │
└─────────────────────────────────────┘
```

---

## 🎬 **DEMO SCREENSHOTS TO INCLUDE**

### **Streamlit App Screenshots:**
1. **Main Interface** - Clean upload area
2. **Image Upload** - Drag and drop functionality
3. **Processing** - Loading spinner
4. **Results** - Disease detection with confidence
5. **Top 5 Predictions** - Multiple disease possibilities
6. **Health Status** - Clear indicators

### **Technical Screenshots:**
1. **Training Progress** - Loss curves
2. **Evaluation Results** - Confusion matrix
3. **Performance Metrics** - Detailed statistics
4. **Code Structure** - Clean, organized files

---

## 📝 **PRESENTATION TIPS**

### **Visual Design:**
- Use consistent colors and fonts
- Include relevant icons and graphics
- Keep text concise and readable
- Use bullet points for clarity
- Include visual hierarchy

### **Content Flow:**
- Start with problem → solution → results
- Use transitions between sections
- Include live demo in middle
- End with impact and future work
- Keep audience engaged throughout

### **Technical Details:**
- Explain complex concepts simply
- Use analogies when helpful
- Show real examples and results
- Demonstrate practical applications
- Address potential questions proactively 