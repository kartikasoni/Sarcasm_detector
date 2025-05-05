#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:41:29 2025

@author: dell
"""

# from flask import Flask, request, render_template
# import joblib
import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import re
# import nltk
# from nltk.corpus import stopwords

# nltk.download('stopwords')
# STOPWORDS = set(stopwords.words('english'))

# # Load Model & Tokenizer
# model = tf.keras.models.load_model("saved_sarcasm_lstm_model.h5")
# tokenizer = joblib.load("saved_tokenizer.pkl")

# # Flask App
# app = Flask(__name__)

# # Preprocessing Function
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'\W', ' ', text)  # Remove special characters
#     text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
#     text = ' '.join([word for word in text.split() if word not in STOPWORDS])
#     return text

# # Prediction Function
# def predict_sarcasm(text):
#     text = clean_text(text)
#     seq = tokenizer.texts_to_sequences([text])
#     padded = pad_sequences(seq, maxlen=30, padding='post')
#     prediction = model.predict(padded)
#     return "Sarcastic" if prediction[0][0] > 0.5 else "Not Sarcastic"

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = ""
#     if request.method == "POST":
#         text = request.form["text"]
#         prediction = predict_sarcasm(text)
#     return render_template("index.html", prediction=prediction)

# if __name__ == "__main__":
#     app.run(debug=True)
    
  
    
# from flask import Flask, request, render_template
# import joblib
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import re
# import nltk
# from nltk.corpus import stopwords

# nltk.download('stopwords')
# STOPWORDS = set(stopwords.words('english'))

# # Load Model & Tokenizer
# model = tf.keras.models.load_model("saved_sarcasm_lstm_model.h5")
# tokenizer = joblib.load("saved_tokenizer.pkl")

# # Flask App
# app = Flask(__name__)

# # Preprocessing Function
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'\W', ' ', text)  # Remove special characters
#     text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
#     text = ' '.join([word for word in text.split() if word not in STOPWORDS])
#     return text

# # Prediction Function
# def predict_sarcasm(text):
#     text = clean_text(text)
#     seq = tokenizer.texts_to_sequences([text])
#     padded = pad_sequences(seq, maxlen=30, padding='post')
#     prediction = model.predict(padded)
    
#     sarcastic_prob = prediction[0][0]  # Probability for sarcastic
#     not_sarcastic_prob = 1 - sarcastic_prob  # Probability for not sarcastic
    
#     return sarcastic_prob, not_sarcastic_prob

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = ""
#     sarcastic_prob = None
#     not_sarcastic_prob = None
#     entered_text = None  # Add this line to capture the entered text
    
#     if request.method == "POST":
#         entered_text = request.form["text"]  # Get the entered text
#         sarcastic_prob, not_sarcastic_prob = predict_sarcasm(entered_text)
#         prediction = "Sarcastic" if sarcastic_prob > 0.5 else "Not Sarcastic"
    
#     return render_template("index.html", 
#                            prediction=prediction, 
#                            sarcastic_prob=sarcastic_prob, 
#                            not_sarcastic_prob=not_sarcastic_prob,
#                            entered_text=entered_text)  # Pass entered_text to the template

# if __name__ == "__main__":
#     app.run(debug=True)  
    
  
 



from flask import Flask, request, render_template
import joblib
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ Load the trained model and tokenizer
model = tf.keras.models.load_model("saved_sarcasm_lstm_model.h5")
tokenizer = joblib.load("saved_tokenizer.pkl")

# ✅ Text Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# ✅ Prediction Function Using `predict()` for probabilities
def predict_sarcasm(text):
    text = clean_text(text)
    
    # Convert text to sequences and pad it
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=30, padding='post')
    
    # Get probabilities using `predict()`
    prediction_prob = model.predict(padded)[0]  # Model returns a list of probabilities for each class
    
    sarcastic_prob = prediction_prob[1]  # Probability of sarcasm
    not_sarcastic_prob = prediction_prob[0]  # Probability of non-sarcasm
    
    return sarcastic_prob, not_sarcastic_prob

# ✅ Flask App
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    sarcastic_prob = None
    not_sarcastic_prob = None
    entered_text = None  
    
    if request.method == "POST":
        entered_text = request.form["text"]  # Get input text
        
        # Get sarcasm vs. non-sarcasm probabilities
        sarcastic_prob, not_sarcastic_prob = predict_sarcasm(entered_text)
        
        prediction = "Sarcastic" if sarcastic_prob > 0.7 else "Not Sarcastic"
    
    return render_template("index.html", 
                            prediction=prediction, 
                            sarcastic_prob=sarcastic_prob, 
                            not_sarcastic_prob=not_sarcastic_prob,
                            entered_text=entered_text)

if __name__ == "__main__":
    app.run(debug=True)
   
  
    
# from flask import Flask, request, render_template
# import joblib
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import re
# import nltk
# from nltk.corpus import stopwords

# nltk.download('stopwords')
# STOPWORDS = set(stopwords.words('english'))

# # Load Model & Tokenizer
# model = tf.keras.models.load_model("saved_sarcasm_lstm_model.h5")
# tokenizer = joblib.load("saved_tokenizer.pkl")

# # Flask App
# app = Flask(__name__)

# # Preprocessing Function
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'\W', ' ', text)  # Remove special characters
#     text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
#     text = ' '.join([word for word in text.split() if word not in STOPWORDS])
#     return text

# # Prediction Function
# def predict_sarcasm(text):
#     text = clean_text(text)
#     seq = tokenizer.texts_to_sequences([text])
#     padded = pad_sequences(seq, maxlen=30, padding='post')
#     prediction = model.predict(padded)
    
#     sarcastic_prob = prediction[0][0]  # Probability for sarcastic
#     not_sarcastic_prob = 1 - sarcastic_prob  # Probability for not sarcastic
    
#     return sarcastic_prob, not_sarcastic_prob

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = ""
#     sarcastic_prob = None
#     not_sarcastic_prob = None
    
#     if request.method == "POST":
#         text = request.form["text"]
#         sarcastic_prob, not_sarcastic_prob = predict_sarcasm(text)
#         prediction = "Sarcastic" if sarcastic_prob > 0.5 else "Not Sarcastic"
    
#     return render_template("index.html", 
#                             prediction=prediction, 
#                             sarcastic_prob=sarcastic_prob, 
#                             not_sarcastic_prob=not_sarcastic_prob)

# if __name__ == "__main__":
#     app.run(debug=True)   