# Media Summarizer App

## Overview
The Media Summarizer App is a tool designed to extract audio from video files, convert MP3 files to WAV format, transcribe audio to text, and summarize the transcribed text using advanced transformer-based models.

## Features
- **Audio Extraction**: Extract audio from video files and save it as a WAV file.
- **Audio Conversion**: Convert MP3 files to WAV format.
- **Transcription**: Transcribe audio to text using Google's Speech Recognition API.
- **Summarization**: Summarize transcribed text using models like BART, T5, Pegasus, and ProphetNet.

## Models Used for Summarization
### BART
BART is a transformer model that combines the bidirectional encoder of BERT and the autoregressive decoder of GPT. It is effective for text generation tasks such as summarization.

### T5
T5 treats every NLP problem as a text-to-text problem. It is highly versatile and performs well across a variety of tasks, including summarization.

### Pegasus
Pegasus is specifically designed for abstractive text summarization. It uses a novel pretraining task that enhances its summarization capabilities.

### ProphetNet
ProphetNet predicts multiple future tokens simultaneously, making it robust for sequence generation tasks like summarization.

## ROUGE Scores
ROUGE scores are used to evaluate the quality of summaries. Higher ROUGE scores indicate better summary quality.

- **ROUGE-N**: Measures n-gram overlap.
- **ROUGE-L**: Measures the longest common subsequence.
- **ROUGE-W**: A weighted LCS-based score.

## Screenshots
![Transcipt_image](https://github.com/user-attachments/assets/8ea68972-209a-4419-ad9a-431369b6f867)

## Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Run The App 
streamlit run app.py
