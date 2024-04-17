# Multi-Modal Intent Recognition in Healthcare

## Introduction
The project addresses the specific challenge of accurately classifying the intent of patients discussing medical problems by developing an audio classification model. The model will help automate and streamline clerical processes like scheduling appointments via calls, manually allocating patients to doctors based on ailments, etc. The complexity lies in the diverse symptoms and expressions patients may present during medical discussions.

## Motivation
The project aims to automate and streamline clerical processes in healthcare by accurately classifying patient intents from audio recordings, thereby facilitating more efficient resource allocation and reducing healthcare costs.

## Installation

### Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3
- Streamlit
- st_audiorec
- joblib
- transformers
- numpy
- nltk
- scipy

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

2. Run the following command to start the Streamlit app:
    ```bash
    streamlit run project/app/index.py
    ```

3. This will launch the Streamlit application in your default web browser. You can now interact with the project as per its functionality.

4. Follow the on-screen instructions to utilize the features of the project.

## Contributing

If you're interested in contributing to this project, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to get started.

## Libraries and Tools Used

- Flask - The web framework used
- NumPy - Library for numerical calculations
- Other relevant libraries or tools

## Additional Information

The original dataset with audio utterances can be found [here](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent/data). The audio classification model is trained on the pre-processed dataset available [here](https://huggingface.co/datasets/shreyas1104/medical-intent-audio-dataset-consolidated).



