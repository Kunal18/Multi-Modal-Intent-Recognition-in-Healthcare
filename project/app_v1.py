import streamlit as st
from st_audiorec import st_audiorec
import os
import tempfile
from predict import predict_label_from_audio

# Function to process audio
def process_audio(audio_data, recorder_flag):
    st.subheader("Processing Audio")
    result_placeholder = st.empty()
    
    with st.spinner("Processing..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            if recorder_flag:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            else:
                tmp_file.write(audio_data.read())
                tmp_file_path = tmp_file.name

        phrase, prediction = predict_label_from_audio(tmp_file_path)
        st.write(f"Phrase: {phrase}")
        st.write(f"Predicted Label: {prediction}")

        # Delete temporary file
        os.unlink(tmp_file_path)
    
    result_placeholder.empty()

# Set title and description
st.title("Audio Classification App")
st.write("This app allows you to record audio or upload an audio file for classification.")

# Audio recording section
st.header("Record Audio")
wav_audio_data = st_audiorec()
if wav_audio_data is not None:
    if st.button("Process Recorded Audio"):
        process_audio(wav_audio_data, True)

# Audio upload section
st.header("Upload Audio File")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
if uploaded_file is not None:
    if st.button("Process Uploaded Audio"):
        process_audio(uploaded_file, False)