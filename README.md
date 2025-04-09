# 📝 OCR Handwriting Recognition Model

This repository contains the implementation of a deep learning-based Optical Character Recognition (OCR) model focused on recognizing handwritten words. The model leverages a CRNN (Convolutional Recurrent Neural Network) architecture and is trained on a Kaggle handwriting dataset.

---

## ✅ Phase 1: Completed

### 📦 Dataset
- **Source:** [Kaggle - Handwriting Recognition Dataset](https://www.kaggle.com/datasets/landlord/handwriting-recognition)
- Downloaded and extracted using the Kaggle API.

### 🧠 Model Architecture
- **CRNN (Convolutional Recurrent Neural Network)**:
  - CNN for visual feature extraction
  - BiLSTM for sequence modeling
  - Linear layer for character classification
- **Loss Function:** CTC Loss (Connectionist Temporal Classification)
- **Optimizer:** Adam with gradient clipping

### 🔁 Training
- Trained for **10 epochs** on GPU using `torch.cuda` if available
- Custom `Dataset` and `DataLoader` pipeline for preprocessing
- Real-time training diagnostics: NaN detection, batch stats, shape checks

### 📊 Evaluation
- Exact Match Accuracy
- Character-Level Accuracy (SequenceMatcher)
- Edit Distance (Levenshtein)
- Evaluation script runs post-training without retraining
- Summary report generated in both Markdown and PDF formats

---

## 🚧 Phase 2: Planned Next

### 📈 Enhancements
- Implement a validation split for better generalization tracking
- Introduce **data augmentation** techniques to improve robustness
- Add **learning rate schedulers** for optimized convergence
- Introduce **dropout** and **batch normalization** to reduce overfitting

### 🔍 Evaluation Improvements
- Visualize predictions directly on input images
- Track accuracy and loss curves using Matplotlib or TensorBoard
- Save misclassified samples for qualitative analysis

### 💾 Model Deployment
- Export trained model (`.pt`) using `torch.save()`
- Create a simple **Streamlit** or **Gradio** app for demo
- Upload to Hugging Face or host on GitHub Pages (if web-based)



