import pandas as pd
import streamlit as st
from st_audiorec import st_audiorec
import os
import tempfile
from predict import predict_class_from_audio

def process_audio_and_classify(audio_data, recorder_flag):
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

        st.audio(tmp_file_path, format='audio/wav')
        predicted_label, predictions= predict_class_from_audio(tmp_file_path)
        st.write(f"Label: {predicted_label}")
        if predicted_label is not None:
            labels = [entry['label'] for entry in predictions]
            scores = [entry['score'] for entry in predictions]
            # Display the bar chart
            st.bar_chart(pd.DataFrame(scores, index=labels, columns=['Score']), use_container_width=True, height=400, width=700)
            
        # Delete temporary file
        os.unlink(tmp_file_path)
    
    result_placeholder.empty()


def conformer_model():
    st.write("This app allows you to record audio or upload an audio file for classification.")
    # Audio recording section
    st.header("Record Audio")
    wav_audio_data = st_audiorec()
    if wav_audio_data is not None:
        if st.button("Process Recorded Audio"):
            process_audio_and_classify(wav_audio_data, True)


    # Audio upload section
    st.header("Upload Audio File")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
    if uploaded_file is not None:
        if st.button("Process Uploaded Audio"):
            process_audio_and_classify(uploaded_file, False)


if __name__ == "__main__":
    conformer_model()

