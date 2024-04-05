import joblib
from transformers import pipeline
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.special import softmax
import json


with open('/Users/kunalindore/Library/CloudStorage/OneDrive-NortheasternUniversity/Capstone/Multi-Modal-Intent-Recognition-in-Healthcare/metadata/class_labels.json', 'r') as f:
    class_mapping = json.load(f)
    print(class_mapping)

# Load models
whisper = pipeline('automatic-speech-recognition', model='openai/whisper-small')
tfidf_vectorizer = joblib.load('/Users/kunalindore/Library/CloudStorage/OneDrive-NortheasternUniversity/Capstone/Multi-Modal-Intent-Recognition-in-Healthcare/project/models/tfidf_vectorizer_v3.pkl')
random_forest_model = joblib.load('/Users/kunalindore/Library/CloudStorage/OneDrive-NortheasternUniversity/Capstone/Multi-Modal-Intent-Recognition-in-Healthcare/project/models/random_forest_model_v3.pkl')

#Pipeline for Fine tuned model
fine_tuned_model = pipeline("audio-classification", model="kunal18/wav2vec2-conformer-rel-pos-large-medical-intent-fine-tuned")

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
    print("Whisper and RFE Approach")
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
                # prediction_label = label_encoder.inverse_transform([max_prob_index])[0]
                prediction_label = class_mapping[str(max_prob_index)]
                return transcription['text'], prediction_label
            else:
                # Return None or some indication of low confidence
                return transcription['text'], "Low confidence prediction, please try again!"
    except Exception as e:
        print("An error occurred during transcription:", e)
        return None, None

def predict_class_from_audio(file_path):
    print("Fine Tuned Approach")
    with open(file_path, "rb") as f:
        audio_data = f.read()
        # print(audio_data)
    # Transcribe audio
    try:
        predicted_class = fine_tuned_model(audio_data)
        print(predicted_class)
        scores = [pred['score'] for pred in predicted_class]
        
        probabilities = softmax(scores)

        # Find index of max probability
        max_index = np.argmax(probabilities)
        predicted_label = predicted_class[max_index]['label']
        max_probability = probabilities[max_index]

        print("Predicted Label:", predicted_label)
        print("Max Probability:", max_probability)
        return predicted_label 

    except Exception as e:
        print("An error occurred during transcription:", e)
        return None

