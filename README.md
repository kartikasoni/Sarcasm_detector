# Sarcasm_detector
# Sarcasm Detection with LSTM

This project builds an LSTM-based model to detect sarcasm in news headlines using the [Sarcasm Headlines Dataset](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection).

---

## Overview

This project includes:

- **Text preprocessing & tokenization**
- **LSTM-based model for classification**
- **Evaluation metrics: Accuracy, Precision, Recall, F1-score**
- **Visualization: Training plots & confusion matrix**
- **Model and tokenizer saving for reuse**

---

## Files

- `sarcasm_model.py`: Contains the classes for data preprocessing and model building.
- `Sarcasm_Headlines_Dataset.json`: Dataset file (must be downloaded from Kaggle).
- `sarcasm_lstm_model.h5`: Trained model (saved).
- `tokenizer.pkl`: Trained tokenizer (saved with pickle).
- `README.md`: This file.

---

## Requirements

Install required packages:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn tensorflow
