import streamlit as st
# from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration, WebRtcMode
import os
import tempfile
from predict import predict_label_from_audio

st.title("Audio Classification App")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    result_placeholder = st.empty()
    
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("Processing..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        phrase, prediction = predict_label_from_audio(tmp_file_path)
        
        # Delete temporary file
        os.unlink(tmp_file_path)

    result_placeholder.empty()

    st.write(f"Phrase: {phrase}")
    st.write(f"Predicted label: {prediction}")