# Chest-X-Ray-Pneumonia-Classification-using-DenseNet121
A DenseNet121-based deep learning model trained in a VM environment to classify chest X-rays as NORMAL or PNEUMONIA, achieving strong accuracy and a high AUC score.

🩺 Chest X-Ray Pneumonia Classification using DenseNet121
Deep Learning Model for Detecting Pneumonia from Chest X-rays (VM-Based Implementation)
📌 Project Overview
This project applies a DenseNet121-based Convolutional Neural Network (CNN) to classify chest X-ray images into NORMAL and PNEUMONIA categories. The model was trained and evaluated inside a Virtual Machine (VM) environment, demonstrating that effective medical imaging models can be developed even with limited computational resources.
Using transfer learning, the model takes advantage of DenseNet121’s powerful pretrained feature extraction capabilities to achieve high accuracy on the pneumonia detection task.
🚀 Key Features
DenseNet121 transfer learning architecture
Image preprocessing and augmentation
VM-based execution
Evaluation using accuracy, confusion matrix, classification report
ROC/AUC-based performance measurement
Reproducible pipeline using TensorFlow + Keras
🛠️ Model Architecture
This project uses a two-part deep learning architecture:
1. DenseNet121 Backbone (Frozen)
Pretrained on ImageNet
All convolutional layers frozen
Provides rich hierarchical visual features suitable for medical images
2. Custom Classification Head
GlobalAveragePooling2D
Dropout layers to reduce overfitting
Dense(256, ReLU) for feature adaptation
Dense(1, Sigmoid) for binary classification
This structure allows the model to adapt pretrained features to pneumonia detection efficiently.
📊 Performance Summary
Below are the final performance metrics observed:
Training Accuracy: 0.9866
Validation Accuracy: 1.0000
Test Accuracy: 0.9199
AUC Score: 0.9693
These results indicate that the model generalizes well and performs reliably for detecting pneumonia in chest X-rays.
📈 Key Results (Textual Overview)
The model showed very high accuracy during training and validation.
Test accuracy remained above 91%, showing strong generalization.
The confusion matrix revealed few misclassifications.
The classification report showed strong precision, recall, and F1-scores for both classes.
The ROC curve demonstrated excellent separation between NORMAL and PNEUMONIA, reflected in a high AUC score.
All graphical plots (confusion matrix, ROC curve, etc.) are included in the project report, not in this README.
📂 Project Structure
.
├── train/                    # Training dataset
├── val/                      # Validation dataset
├── test/                     # Test dataset
├── models/                   # Saved model weights (optional)
├── Chest-Xray.ipynb          # Full notebook with code
└── README.md                 # Project documentation
⚙️ Installation
Install all required packages:
pip install tensorflow keras numpy matplotlib seaborn scikit-learn
Recommended VM setup:
Python 3.8+
TensorFlow installed via pip
At least 8GB RAM (lower RAM works with smaller batches)
▶️ How to Run the Project
Clone the repository:
git clone https://github.com/yourusername/Chest-Xray-DenseNet121.git
cd Chest-Xray-DenseNet121
Launch Jupyter Notebook:
jupyter notebook
Open the notebook and run all cells to train and evaluate the model.
🧪 Evaluation Metrics (Text Summary)
The model was evaluated using:
Accuracy – overall correctness
Confusion matrix – distribution of correct/incorrect classifications
Precision, Recall, F1-score – class-specific performance
ROC Curve and AUC Score – threshold-independent diagnostic power
These metrics collectively show that the model is strong at detecting pneumonia with low false negatives.
💡 My Perspective
Training the model in a VM environment taught me how to optimize deep learning workloads under resource constraints. The use of DenseNet121 made the process efficient, since only the classification head needed training. Regularization through dropout and augmentation further strengthened generalization.
This experience highlighted how deep learning models for healthcare applications can still perform well without high-end hardware.
🔮 Future Improvements
Potential enhancements include:
Fine-tuning the upper layers of DenseNet121
Using learning rate schedulers for smoother convergence
Applying Grad-CAM for model explainability
Expanding the validation set
Training on a GPU-enabled VM for faster execution
👤 Author
Syllas Otutey
MS, Health Informatics
Michigan Technological University
