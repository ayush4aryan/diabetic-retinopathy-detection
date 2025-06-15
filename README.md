# 👁️ Diabetic Retinopathy Detection using Deep Learning

This project uses a pre-trained **Convolutional Neural Network (CNN)** model to detect **diabetic retinopathy (DR)** from retinal fundus images. The model predicts the severity level from five possible DR classes.

---

## 🧠 Model Overview

- **Framework:** TensorFlow/Keras
- **Model Input:** Retinal image (224x224)
- **Output Classes:**
  - No DR
  - Mild DR
  - Moderate DR
  - Severe DR
  - Proliferative DR
- Model trained on labeled image dataset of diabetic retinopathy stages.

---

## 🚀 Features

- Predicts DR stage based on fundus image
- Uses softmax probabilities with custom decision thresholds
- Displays prediction with corresponding image using Matplotlib
- Extracts and prints model’s learning rate

---

## 📸 Demo


---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 🧪 How to Run

1. **Install dependencies**
   ```bash
   pip install tensorflow matplotlib numpy
