# Speech-to-Text

This repository contains experiments and pipelines for **speech-to-text (STT)** using Python.  
It includes dataset normalization utilities, fine-tuning workflows, and multiple iterations of model training and evaluation.

---

## 📖 Overview

The goal of this project is to explore different approaches to **automatic speech recognition (ASR)**, from data preprocessing to model fine-tuning and evaluation.  
Several experimental versions (`v1`, `v2`, `v3`, `v3_early_stopping`) are provided to document the evolution of the pipeline.

---

## 📂 Project Structure

speech-to-text/
│
├── estudi_distribucions/ # Scripts and notebooks for dataset distribution analysis
├── fine_tunning_rosany/ # Fine-tuning experiments on the Rosany dataset
├── norm_dataset/ # Dataset normalization and preprocessing utilities
├── v1/ # First training pipeline (baseline)
├── v2/ # Improved pipeline with refinements
├── v3/ # Further improvements
├── v3_early_stopping/ # Experiment with early stopping enabled
└── README.md # Project documentation

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
2. Install the dependencies
pip install -r requirements.txt

🧑‍💻 Usage
Training a model
python v1/train.py --dataset data/ --epochs 10

Running inference
python v3/inference.py --audio samples/test.wav --output result.txt

Dataset normalization
python norm_dataset/normalize.py --input raw/ --output processed/


(Adjust commands to match actual script names and arguments.)

📊 Experiments

v1 → Baseline training pipeline

v2 → Improved preprocessing and training

v3 → Additional refinements in architecture/training

v3_early_stopping → Introduced early stopping to reduce overfitting

📦 Dependencies

Typical dependencies for speech-to-text pipelines may include:

torch

torchaudio

transformers

numpy

pandas

scikit-learn

matplotlib

You can generate a requirements.txt with:

pip freeze > requirements.txt

🤝 Contributing

Contributions are welcome!

Fork the repository

Create a new branch (feature/my-feature)

Commit your changes with clear messages

Push to your branch

Open a Pull Request
