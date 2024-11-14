from flask import Flask, request, jsonify
from pymongo import MongoClient
from datetime import datetime
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import pickle
import uuid  
import os

from config import Config

app = Flask(__name__)

app.config.from_object(Config)
client = MongoClient(app.config['MONGO_URI'])

db = client['users']
users_collection = db['users']

journal_db = client['Journal']
journal_collection = journal_db['journal']

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

try:
    with open('keras_classifier_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("keras_classifier_model.pkl not found. Make sure the file is in the correct directory.")

label_ordered = {
    0: 'sadness',
    1: 'joy',
    2: 'anger',
    3: 'fear',
    4: 'love',
    5: 'surprise',
}

def preprocess_and_get_embeddings(text):
    preprocessed_text = bert_preprocess([text])
    embeddings = bert_encoder(preprocessed_text)['pooled_output']
    return embeddings.numpy()  

def predict_new_sentence(sentence):
    embeddings = preprocess_and_get_embeddings(sentence)
    prediction = model.predict(embeddings)
    predicted_label = label_ordered.get(prediction[0], "Unknown")
    return predicted_label


test_sentence = "I am very happy today!"  
predicted_label = predict_new_sentence(test_sentence)
print(f"The predicted emotion for the sentence '{test_sentence}' is: {predicted_label}")