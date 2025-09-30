# Speech-to-Text

This repository contains experiments and pipelines for **speech-to-text (STT)** using Python.  
It includes dataset normalization utilities, fine-tuning workflows, and multiple iterations of model training and evaluation.

---

## 📖 Overview

The goal of this project is to explore different approaches to **automatic speech recognition (ASR)**, from data preprocessing to model fine-tuning and evaluation.  
Several experimental versions (`v1`, `v2`, `v3`, `v3_early_stopping`) are provided to document the evolution of the pipeline.

---

## 📂 Project Structure

```text
speech-to-text/
│
├── estudi_distribucions/     # Scripts and notebooks for dataset distribution analysis
├── fine_tunning_rosany/      # Fine-tuning experiments on the Rosany dataset
├── norm_dataset/             # Dataset normalization and preprocessing utilities
├── v1/                       # First training pipeline (baseline)
├── v2/                       # Improved pipeline with refinements
├── v3/                       # Further improvements
├── v3_early_stopping/        # Experiment with early stopping enabled
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```
---
## ✨ Features

- Audio-to-text model training and inference  
- Dataset normalization and preprocessing  
- Fine-tuning on custom datasets (e.g., Rosany)  
- Multiple versions of the training pipeline (v1 → v3)  
- Early stopping experiment for better generalization  
- Scripts for dataset distribution analysis  

---
## 🚀 Getting Started

### Prerequisites
- Python 3.8+  
- [pip](https://pip.pypa.io/en/stable/)  
- (Optional) GPU with CUDA support for training  

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/tinips/speech-to-text.git
   cd speech-to-text

---
## 🧑‍💻 Usage

Each experiment version includes a `main.py` script that can be executed directly.

### Training
Run the `main.py` file inside the version folder you want to test:
```bash
python v1/main.py
python v2/main.py
python v3/main.py
python v3_early_stopping/main.py
```
---

## 📊 Experiments

- **v1** → Baseline training pipeline  
- **v2** → Improved preprocessing and training  
- **v3** → Additional refinements in architecture/training  
- **v3_early_stopping** → Introduced early stopping to reduce overfitting  

---
## 📦 Dependencies

This project relies on common libraries for deep learning, audio processing, and data analysis:

- **torch** — core deep learning framework  
- **torchaudio** — audio preprocessing and datasets for PyTorch  
- **numpy** — numerical operations  
- **pandas** — data handling and tabular processing  
- **matplotlib** — visualization of results and training curves  

---

### 📄 Installing dependencies

All dependencies are listed in `requirements.txt`.  
You can install them with:

```bash
pip install -r requirements.txt

