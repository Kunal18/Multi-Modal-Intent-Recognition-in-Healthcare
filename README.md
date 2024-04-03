# Multi-Modal-Intent-Recognition-in-Healthcare
The project addresses the specific challenge of accurately classifying the intent of patients discussing medical problems by developing an audio classification model. The model will help automate and streamline clerical processes like scheduling appointments via calls, manually allocating patients to doctors based on ailments etc. The complexity lies in the diverse symptoms and expressions patients may present during medical discussions.

The original dataset with audio utterances can be found here- Medical Inetnt dataset: https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent/data. The audio classification model is trained on the pre-processed dataset (https://huggingface.co/datasets/shreyas1104/medical-intent-audio-dataset-consolidated) and achieves a high level of accuracy in identifying the intent of patient utterances. The implementation of this model will streamline the preliminary classification process, resulting in more efficient resource allocation and reduced healthcare costs.

The original dataset consists 25 medical symptomswhich were consolidated into 6 broad classes to prevent overlap of predictions between multiple classes. Automatic Speech Recognition (ASR) models transcribe speech to text, then the pre-trained model is initialized for audio classification using the AutoModelForAudioClassification class. The ’facebook/wav2vec2-conformer-rel-pos-large’ pre-trained model is fine-tuned to enhance model perfor- mance on unseen data. Hyperparameters such as learning rate, batch size, number of epochs, and warm-up ratio are tuned to enhance the model's training process and overall effectiveness. To take the project to the next step, a user-ineteractive interface allows anyone to upload their own audio recording and get the transcript and intent for it as well. This is achieved via Streamlit. Streamlit is an open-source Python library that allows for rapid development of interactive web applications.

The audio classification model trained on the provided dataset achieves a high level of accuracy in identifying the intent of patient utterances. The model behaves as expected, showing improvements in performance over 8 epochs and generalizes well to unseen test and validation data, demonstrating its effectiveness in real-world medical scenarios. By accurately classifying patient intents to one of the mapped 6 target classes, the model is helping to streamline the preliminary classification process, resulting in more efficient resource allocation.

Training and Validation Results-

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall |
|-------|---------------|-----------------|----------|-----------|--------|
| 1     | 1.7787        | 1.7463          | 0.2397   | 0.1411    | 0.2397 |
| 2     | 1.3633        | 1.4252          | 0.4912   | 0.4317    | 0.4912 |
| 3     | 1.0073        | 1.2159          | 0.5467   | 0.5391    | 0.5467 |
| 4     | 0.7682        | 1.0005          | 0.6666   | 0.7746    | 0.6666 |
| 5     | 0.4975        | 0.9533          | 0.6813   | 0.7198    | 0.6813 |
| 6     | 0.3092        | 0.7844          | 0.7456   | 0.7800    | 0.7456 |
| 7     | 0.2968        | 0.7393          | 0.7544   | 0.7960    | 0.7544 |
| 8     | 0.3001        | 0.6879          | 0.7719   | 0.8161    | 0.7719 |



