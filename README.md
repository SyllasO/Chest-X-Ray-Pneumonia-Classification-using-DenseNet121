# Chest-X-Ray-Pneumonia-Classification-using-DenseNet121
A DenseNet121-based deep learning model trained in a VM environment to classify chest X-rays as NORMAL or PNEUMONIA, achieving strong accuracy and a high AUC score.

# 🩺 Chest X-Ray Pneumonia Classification using DenseNet121
**Deep Learning Model for Detecting Pneumonia from Chest X-rays (VM-Based Implementation)**

## 📌 Project Overview
This project applies a **DenseNet121-based Convolutional Neural Network (CNN)** to classify chest X-ray images into **NORMAL** and **PNEUMONIA** categories. The entire model was trained and evaluated inside a **Virtual Machine (VM)** environment, demonstrating that high-performance medical imaging models can be built even with limited computational resources.

## 🚀 Key Features
- DenseNet121 transfer learning architecture  
- Image preprocessing and augmentation  
- VM-based execution  
- Evaluation using accuracy, confusion matrix, classification report  
- ROC/AUC-based performance measurement  
- Reproducible TensorFlow + Keras pipeline  

## 🛠️ Model Architecture

### 1. DenseNet121 Backbone (Frozen)
- Pretrained on ImageNet  
- All convolutional layers frozen  
- Extracts rich hierarchical visual features suitable for medical images  

### 2. Custom Classification Head
- GlobalAveragePooling2D  
- Dropout(0.4)  
- Dense(256, ReLU)  
- Dense(1, Sigmoid)  

## 📊 Performance Summary
- **Training Accuracy:** 0.9866  
- **Validation Accuracy:** 1.0000  
- **Test Accuracy:** 0.9199  
- **AUC Score:** 0.9693  

## 📈 Key Results (Text Overview)
- Very high accuracy during training and validation  
- Strong test generalization  
- Minimal misclassifications  
- Excellent precision, recall, and F1-scores  
- Strong ROC curve separation  

## 📂 Project Structure
```
.
├── train/
├── val/
├── test/
├── models/
├── Chest-Xray.ipynb
└── README.md
```

## ⚙️ Installation
```
pip install tensorflow keras numpy matplotlib seaborn scikit-learn
```

Recommended VM setup:
- Python 3.8+
- ≥ 8GB RAM

## ▶️ How to Run the Project
```
git clone https://github.com/yourusername/Chest-Xray-DenseNet121.git
cd Chest-Xray-DenseNet121
jupyter notebook
```

## 🧪 Evaluation Metrics (Text Summary)
- Accuracy  
- Confusion matrix  
- Precision, Recall, F1-score  
- ROC Curve & AUC Score  

## 💡 My Perspective
Training the model in a VM environment highlighted the importance of efficient deep learning workflows and showed that high-performing medical AI can be developed without GPU hardware.

## 🔮 Future Improvements
- Fine-tuning DenseNet layers  
- Learning rate schedulers  
- Grad-CAM explainability  
- Larger validation set  
- GPU-enabled VM training  

## 👤 Author
**Syllas Otutey**  
MS, Health Informatics  
Michigan Technological University
