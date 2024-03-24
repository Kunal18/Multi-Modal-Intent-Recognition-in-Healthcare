import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Read the data and select relevant columns
train_data = pd.read_csv("/Users/kunalindore/Library/CloudStorage/OneDrive-NortheasternUniversity/Capstone/Multi-Modal-Intent-Recognition-in-Healthcare/project/data/metadata_train.csv")
train_data = train_data[["phrase", "merged_prompt"]]

# Read the data and select relevant columns
test_data = pd.read_csv("/Users/kunalindore/Library/CloudStorage/OneDrive-NortheasternUniversity/Capstone/Multi-Modal-Intent-Recognition-in-Healthcare/project/data/metadata_test.csv")
test_data = test_data[["phrase", "merged_prompt"]]

# Read the data and select relevant columns
validate_data = pd.read_csv("/Users/kunalindore/Library/CloudStorage/OneDrive-NortheasternUniversity/Capstone/Multi-Modal-Intent-Recognition-in-Healthcare/project/data/metadata_validate.csv")
validate_data = validate_data[["phrase", "merged_prompt"]]

# Check the number of rows for each merged_prompt class
train_class_counts = train_data['merged_prompt'].value_counts()
print(train_class_counts)

# Check the number of rows for each merged_prompt class
test_class_counts = test_data['merged_prompt'].value_counts()
print(test_class_counts)

# Check the number of rows for each merged_prompt class
validate_class_counts = validate_data['merged_prompt'].value_counts()
print(validate_class_counts)

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

# Rename columns
train_data.rename(columns={"merged_prompt": "intent", "phrase": "audio_phrase"}, inplace=True)
test_data.rename(columns={"merged_prompt": "intent", "phrase": "audio_phrase"}, inplace=True)
validate_data.rename(columns={"merged_prompt": "intent", "phrase": "audio_phrase"}, inplace=True)

# Preprocess text
train_data['audio_phrase'] = train_data['audio_phrase'].apply(preprocess_text)
test_data['audio_phrase'] = test_data['audio_phrase'].apply(preprocess_text)
validate_data['audio_phrase'] = validate_data['audio_phrase'].apply(preprocess_text)

# Perform label encoding
label_encoder = LabelEncoder()
train_data['intent'] = label_encoder.fit_transform(train_data['intent'])
test_data['intent'] = label_encoder.fit_transform(test_data['intent'])
validate_data['intent'] = label_encoder.fit_transform(validate_data['intent'])

label_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
print("Label mapping:", label_mapping)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=True)
X_train = tfidf_vectorizer.fit_transform(train_data['audio_phrase'])
y_train = train_data['intent']
X_test = tfidf_vectorizer.transform(test_data['audio_phrase'])
y_test = test_data['intent']
X_val = tfidf_vectorizer.transform(validate_data['audio_phrase'])
y_val = validate_data['intent']

# Initialize and train Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

# Evaluate the model
y_pred_train = random_forest.predict(X_train)
y_pred_test = random_forest.predict(X_test)
y_pred_val = random_forest.predict(X_val)

print("Train data classification report:")
print(classification_report(y_train, y_pred_train))

print("Test data classification report:")
print(classification_report(y_test, y_pred_test))

print("Validation data classification report:")
print(classification_report(y_val, y_pred_val))

# Grid Search Cross Validation
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=10)
grid_search.fit(X_train, y_train)

best_random_forest = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Evaluate the best model
y_pred_train = best_random_forest.predict(X_train)
y_pred_test = best_random_forest.predict(X_test)
y_pred_val = best_random_forest.predict(X_val)

print("Train data classification report:")
print(classification_report(y_train, y_pred_train))

print("Test data classification report:")
print(classification_report(y_test, y_pred_test))

print("Validation data classification report:")
print(classification_report(y_val, y_pred_val))

# Save models and vectorizer
joblib.dump(best_random_forest, '/Users/kunalindore/Library/CloudStorage/OneDrive-NortheasternUniversity/Capstone/Multi-Modal-Intent-Recognition-in-Healthcare/project/models/random_forest_model.pkl', protocol=4)
joblib.dump(tfidf_vectorizer, '/Users/kunalindore/Library/CloudStorage/OneDrive-NortheasternUniversity/Capstone/Multi-Modal-Intent-Recognition-in-Healthcare/project/models/tfidf_vectorizer.pkl')
joblib.dump(label_encoder, '/Users/kunalindore/Library/CloudStorage/OneDrive-NortheasternUniversity/Capstone/Multi-Modal-Intent-Recognition-in-Healthcare/project/models/label_encoder.pkl')
