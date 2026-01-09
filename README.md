 🫁 Chest X-Ray Pneumonia Detection using Deep Learning

A deep learning–based system to automatically detect Pneumonia from chest X-ray images using Convolutional Neural Networks (CNNs) and Transfer Learning, with Explainable AI (Grad-CAM).

This project implements a complete medical imaging pipeline:
Data → Preprocessing → Training → Evaluation → Explainability → Deployment-ready model



📌 Problem Statement

Pneumonia is a serious lung infection that can be life-threatening if not detected early.  
Manual diagnosis from chest X-rays is time-consuming and subject to human error.

This project aims to build an **AI-assisted system** to classify chest X-ray images into:

- **Normal**
- **Pneumonia**

---

📊 Dataset

We use the Kaggle Chest X-Ray Pneumonia dataset:

- ~5,800 pediatric chest X-ray images
- Two classes: Normal and Pneumonia

Dataset link:  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data

Dataset structure:
chest_xray/
├── train/
├── val/
└── test/
├── NORMAL/
└── PNEUMONIA/
## 🧪 Preprocessing

- Resize images to **224 × 224**
- Normalize pixel values
- Data augmentation:
  - Random rotation (±10°)
  - Horizontal flip
  - Random crop and resize
  - Brightness and contrast adjustment
- Class imbalance handled using **weighted loss**



 🧠 Model Architecture

We use DenseNet-121 (ImageNet pretrained) for transfer learning.

Pipeline:
1. Load pretrained DenseNet-121
2. Replace classifier head with binary output layer
3. Freeze backbone for initial epochs
4. Fine-tune the entire network


⚙️ Training Setup

| Component | Value |
|--------|-------|
| Input size | 224 × 224 |
| Loss function | BCEWithLogitsLoss |
| Optimizer | AdamW |
| Learning Rate | 1e-4 → 1e-5 |
| Batch Size | 32 |
| Epochs | 20–50 |
| Scheduler | Cosine Annealing |
| Metric | ROC-AUC (primary) |

Early stopping is applied based on validation AUC.


📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall (important in medical diagnosis)  
- F1-Score  
- Confusion Matrix  
- ROC Curve and AUC  

---

🔍 Explainability (Grad-CAM)

Grad-CAM heatmaps are generated to visualize which regions of the lungs influence the model’s decision.  
This helps in:
- Model interpretability
- Trust and transparency
- Identifying failure cases

