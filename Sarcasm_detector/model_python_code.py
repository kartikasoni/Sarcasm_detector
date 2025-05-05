#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:24:41 2025

@author: dell
"""
#!pip install --upgrade --force-reinstall numpy
#!pip install --upgrade --force-reinstall pandas scipy matplotlib scikit-learn


import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, SpatialDropout1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import joblib
import pickle
import tensorflow as tf

class SarcasmPreprocessor:
    def __init__(self, data_path, max_features=2000, max_len=30, test_size=0.2, random_state=42):
        """
        Initialize the sarcasm data preprocessor.

        :param data_path: Path to the JSON dataset file.
        :param max_features: Maximum vocabulary size for tokenization.
        :param max_len: Maximum sequence length for padding.
        :param test_size: Proportion of data used for testing.
        :param random_state: Random state for reproducibility.
        """
        self.data_path = data_path
        self.max_features = max_features
        self.max_len = max_len
        self.test_size = test_size
        self.random_state = random_state
        self.tokenizer = Tokenizer(num_words=max_features, split=' ')

    def load_data(self):
        """
        Load and preprocess the sarcasm dataset.

        :return: Processed training and testing datasets (X_train, X_test, Y_train, Y_test).
        """
        # Load the dataset
        df = pd.read_json(self.data_path, lines=True)

        # Keep only necessary columns
        df = df[['headline', 'is_sarcastic']]

        # Apply text cleaning
        df['headline'] = df['headline'].apply(self.clean_text)

        # Tokenize and pad sequences
        X = self.tokenize_and_pad(df['headline'].values)

        # Convert labels to one-hot encoding
        Y = pd.get_dummies(df['is_sarcastic']).values

        # Split dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=self.random_state
        )

        return X_train, X_test, Y_train, Y_test, self.tokenizer

    def clean_text(self, text):
        """
        Perform basic text preprocessing: convert to lowercase, remove special characters.

        :param text: Input text string.
        :return: Cleaned text string.
        """
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        text = text.replace('rt', '')  # Remove 'rt' (retweet)
        return text

    def tokenize_and_pad(self, text_data):
        """
        Tokenizes and pads the text sequences.

        :param text_data: List of text strings.
        :return: Padded sequences.
        """
        # Fit tokenizer on text
        self.tokenizer.fit_on_texts(text_data)

        # Convert text to sequences
        sequences = self.tokenizer.texts_to_sequences(text_data)

        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')

        return padded_sequences


class SarcasmModel:
    def __init__(self, max_features=2000, embed_dim=128, lstm_out=196, max_len=30):
        """
        Initialize and build the LSTM model for sarcasm detection.

        :param max_features: Vocabulary size.
        :param embed_dim: Size of the word embeddings.
        :param lstm_out: Number of LSTM units.
        :param max_len: Input sequence length.
        """
        self.max_features = max_features
        self.embed_dim = embed_dim
        self.lstm_out = lstm_out
        self.max_len = max_len
        self.model = self.build_model()

    def build_model(self):
        """
        Define and compile the LSTM model.

        :return: Compiled model.
        """
        model = Sequential()
        model.add(Embedding(self.max_features, self.embed_dim, input_length=self.max_len))
        model.add(SpatialDropout1D(0.4))
        model.add(LSTM(self.lstm_out, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_model(self, X_train, Y_train, batch_size=32, epochs=10):
        """
        Train the LSTM model.

        :param X_train: Training input data.
        :param Y_train: Training labels.
        :param batch_size: Batch size for training.
        :param epochs: Number of training epochs.
        :return: Training history.
        """
        history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=2)
        return history

    def evaluate_model(self, X_test, Y_test, batch_size=32):
        """
        Evaluate the model on the test set.

        :param X_test: Test input data.
        :param Y_test: Test labels.
        :param batch_size: Batch size for evaluation.
        """
        score, acc = self.model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
        print("Test Loss:", round(score, 2))
        print("Test Accuracy:", round(acc, 4))

    def plot_training_history(self, history):
        """
        Plot training accuracy and loss.

        :param history: Training history object.
        """
        plt.figure(figsize=(12, 5))

        # Accuracy Plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')

        # Loss Plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')

        plt.show()

    def confusion_matrix(self, Y_test, y_pred):
        """
        Plot confusion matrix.

        :param Y_test: True labels.
        :param y_pred: Predicted labels.
        """
        cm = confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Non-Sarcastic', 'Sarcastic'], yticklabels=['Non-Sarcastic', 'Sarcastic'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()


# Main Execution
if __name__ == "__main__":
    # Load and preprocess data
    data_path = "Sarcasm_Headlines_Dataset.json"
    preprocessor = SarcasmPreprocessor(data_path)
    X_train, X_test, Y_train, Y_test, tokenizer = preprocessor.load_data()

    # Initialize and train model
    model = SarcasmModel()
    history = model.train_model(X_train, Y_train)

    # Evaluate model
    model.evaluate_model(X_test, Y_test)

    # Plot training history
    model.plot_training_history(history)

    # Get predictions
    y_pred = model.model.predict(X_test)

    # Compute precision, recall, and F1-score
    precision = precision_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1), average='micro')
    recall = recall_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1), average='micro')
    f1 = f1_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1), average='micro')

    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-Score: {f1:.6f}")

    # Plot confusion matrix
    model.confusion_matrix(Y_test, y_pred)



model = SarcasmModel()

# Save the trained model
model.model.save("sarcasm_lstm_model.h5")  # Saves the model in .h5 format
print("Model saved as 'saved_sarcasm_lstm_model.h5'")

# Save the tokenizer (assuming you have a tokenizer variable)
with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)
print("Tokenizer saved as 'saved_tokenizer.pkl'")
