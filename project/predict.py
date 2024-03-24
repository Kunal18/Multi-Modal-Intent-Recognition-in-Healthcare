import joblib
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report





# Load models
whisper = pipeline('automatic-speech-recognition', model='openai/whisper-small')
tfidf_vectorizer = joblib.load('/Users/kunalindore/Library/CloudStorage/OneDrive-NortheasternUniversity/Capstone/Multi-Modal-Intent-Recognition-in-Healthcare/project/models/tfidf_vectorizer.pkl')
random_forest_model = joblib.load('/Users/kunalindore/Library/CloudStorage/OneDrive-NortheasternUniversity/Capstone/Multi-Modal-Intent-Recognition-in-Healthcare/project/models/random_forest_model.pkl')
label_encoder = joblib.load('/Users/kunalindore/Library/CloudStorage/OneDrive-NortheasternUniversity/Capstone/Multi-Modal-Intent-Recognition-in-Healthcare/project/models/label_encoder.pkl')

# Preprocess text
def preprocess_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_text = cleaned_text.lower()
    stop_words = set(stopwords.words('english'))
    lemma = WordNetLemmatizer()
    tokens = word_tokenize(cleaned_text)
    cleaned_text = ' '.join([word for word in tokens if word not in stop_words])
    cleaned_text = ' '.join([lemma.lemmatize(word) for word in cleaned_text.split()])
    return cleaned_text

def predict_label_from_audio(file_path):
    with open(file_path, "rb") as f:
        audio_data = f.read()
        # print(audio_data)
    # Transcribe audio
    confidence_threshold = 0.7
    try:
        transcription = whisper(audio_data)
        
        if transcription['text']:
            # Preprocess text
            preprocessed_text = preprocess_text(transcription['text'])

            # Vectorize text
            vectorized_text = tfidf_vectorizer.transform([preprocessed_text])

            # Predict probabilities using Random Forest model
            prediction_probs = random_forest_model.predict_proba(vectorized_text)
            
            # Get the maximum probability and its corresponding class index
            max_prob_index = np.argmax(prediction_probs)
            max_prob = prediction_probs[0, max_prob_index]

            # Check if the maximum probability is above the confidence threshold
            if max_prob >= confidence_threshold:
                # Map encoded prediction back to original label
                prediction_label = label_encoder.inverse_transform([max_prob_index])[0]
                return transcription['text'], prediction_label
            else:
                # Return None or some indication of low confidence
                return transcription['text'], "Low confidence prediction, please try again!"
    except Exception as e:
        print("An error occurred during transcription:", e)
        return None, None

    # try:
    #     transcription = whisper(audio_data)
    #     if transcription['text']:
    #     # Preprocess text
    #         preprocessed_text = preprocess_text(transcription['text'])

    #         # Vectorize text
    #         vectorized_text = tfidf_vectorizer.transform([preprocessed_text])
            
    #         # Predict using Random Forest model
    #         prediction_encoded = random_forest_model.predict(vectorized_text)
            
    #         # Map encoded prediction back to original label
    #         prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
        
    #         return transcription['text'], prediction_label
    # except Exception as e:
    #     print("An error occurred during transcription:", e)

    
    # return "None"


