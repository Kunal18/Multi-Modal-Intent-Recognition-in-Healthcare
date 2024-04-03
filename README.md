# Multi-Modal-Intent-Recognition-in-Healthcare
The project addresses the specific challenge of accurately classifying the intent of patients discussing medical problems by developing an audio classification model. The model will help automate and streamline clerical processes like scheduling appointments via calls, manually allocating patients to doctors based on ailments etc. The complexity lies in the diverse symptoms and expressions patients may present during medical discussions.

The original dataset with audio utterances can be found here- Medical Inetnt dataset: https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent/data. The audio classification model is trained on the pre-processed dataset (https://huggingface.co/datasets/shreyas1104/medical-intent-audio-dataset-consolidated) and achieves a high level of accuracy in identifying the intent of patient utterances. The implementation of this model will streamline the preliminary classification process, resulting in more efficient resource allocation and reduced healthcare costs.

The original dataset consists 25 medical symptomswhich were consolidated into 6 broad classes to prevent overlap of predictions between multiple classes. Automatic Speech Recognition (ASR) models transcribe speech to text, then the pre-trained model is initialized for audio classification using the AutoModelForAudioClassification class. The ’facebook/wav2vec2-conformer-rel-pos-large’ pre-trained model is fine-tuned to enhance model perfor- mance on unseen data. Hyperparameters such as learning rate, batch size, number of epochs, and warm-up ratio are tuned to enhance the model's training process and overall effectiveness. To take the project to the next step, a user-ineteractive interface allows anyone to upload their own audio recording and get the transcript and intent for it as well. This is achieved via Streamlit. Streamlit is an open-source Python library that allows for rapid development of interactive web applications.

The audio classification model trained on the provided dataset achieves a high level of accuracy, 81.6\% precision on training dataset and 97.6\% accuracy, 97.6\% F-1 Score on test data, in identifying the intent of patient utterances. The model behaves as expected, showing improvements in performance over 8 epochs and generalizes well to unseen test and validation data, demonstrating its effectiveness in real-world medical scenarios. By accurately classifying patient intents to one of the mapped 6 target classes, the model is helping to streamline the preliminary classification process, resulting in more efficient resource allocation.
