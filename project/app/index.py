import streamlit as st
from fine_tuned_model import conformer_model
from whisper_rfe import whisper_rfe_model


# Functions for the two pages
def conformer_page():
    st.write("# Fine-tuned Facebook Conformer Model for Audio Classification")
    st.write("This page utilizes a Fine-tuned Facebook Conformer model to classify audio files into different patient intents.")
    conformer_model()
    # Add your code for the Facebook Conformer model here


def whisper_rf_page():
    st.write("# Whisper AI and Random Forest for Audio Classification")
    st.write("This page uses Whisper AI for transcription and then employs Random Forest for text classification into patient intents.")
    whisper_rfe_model()
    # Add your code for Whisper AI and Random Forest here


def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Fine-tuned Conformer Model Approach", "Whisper AI and Random Forest Approach"])

    if selection == "Fine-tuned Conformer Model Approach":
        conformer_page()
    elif selection == "Whisper AI and Random Forest Approach":
        whisper_rf_page()

if __name__ == "__main__":
    st.set_page_config(page_title="Multi-Model Intent Recognition of Patients Audio")
    main()
