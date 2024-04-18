# Multi-Modal Intent Recognition in Healthcare

## Introduction
The project addresses the specific challenge of accurately classifying the intent of patients discussing medical problems by developing an audio classification model. The model will help automate and streamline clerical processes like scheduling appointments via calls, manually allocating patients to doctors based on ailments, etc. The complexity lies in the diverse symptoms and expressions patients may present during medical discussions.

## Motivation
The project aims to automate and streamline clerical processes in healthcare by accurately classifying patient intents from audio recordings, thereby facilitating more efficient resource allocation and reducing healthcare costs.

## Installation

### Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3 - foundational programming language for our project, providing a versatile and user-friendly environment for development
- st_audiorec - used for audio recording and processing, allowing us to capture and analyze audio data in real-time
- joblib - used for efficient data caching and parallel computing, optimizing the performance of our data processing tasks
- transformers -  utilized for state-of-the-art natural language processing (NLP), enabling us to implement advanced text-based models and algorithms
- numpy - provides essential numerical computing capabilities, facilitating efficient array operations and mathematical computations
- nltk - leveraged for natural language processing tasks such as tokenization, stemming, and part-of-speech tagging, enhancing text data preprocessing and analysis
- scipy - used for scientific computing and advanced mathematical functions, supporting various data analysis and statistical operations
- Streamlit - employed as our web application framework, enabling easy and interactive data visualization and deployment of machine learning models


### Installation Steps

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/project.git
    ```

2. Navigate to the project directory:
    ```bash
    cd project
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Project

1. After installing the necessary dependencies, navigate to the project directory.

2. Run the following command in command prompt to start the Streamlit app:
    ```bash
    streamlit run project/app/index.py
    ```

3. This will launch the Streamlit application in your default web browser. You can now interact with the project as per its functionality.

4. Follow the on-screen instructions to utilize the features of the project.

## Additional Information

The original dataset with audio utterances can be found [here](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent/data). The audio classification model is trained on the pre-processed dataset available [here](https://huggingface.co/datasets/shreyas1104/medical-intent-audio-dataset-consolidated).

## Results
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

## DEMO
https://github.com/Kunal18/Multi-Modal-Intent-Recognition-in-Healthcare/assets/20738263/b5069db0-8b21-43cf-bee2-da022cfa596f

