
from moviepy.editor import VideoFileClip  # Import VideoFileClip from moviepy.editor for video processing
import librosa  # Import librosa for audio processing
import soundfile as sf  # Import soundfile for saving audio files
import os  # Import os for basic operating system operations
import streamlit as st  # Import Streamlit for creating web applications

# Function to extract audio from a video file and save it as a WAV file
def extract_audio_from_video(video_path, audio_output_path):
    try:
        # Open the video file using VideoFileClip
        video_clip = VideoFileClip(video_path)
        # Extract the audio from the video clip
        audio_clip = video_clip.audio
        # Write the extracted audio to a WAV file
        audio_clip.write_audiofile(audio_output_path)
        # Close the audio and video clips to free up resources
        audio_clip.close()
        video_clip.close()
        # Display a success message using Streamlit
        st.success(f"Audio extracted and saved as {audio_output_path}")
    except Exception as e:
        # If an error occurs during extraction, display an error message using Streamlit
        st.error(f"Error extracting audio from video: {e}")

# Function to convert an MP3 file to WAV format using librosa
def convert_mp3_to_wav(mp3_path, wav_output_path):
    try:
        # Load the MP3 file using librosa and get the sample rate (sr) of the audio
        y, sr = librosa.load(mp3_path, sr=None)
        # Write the audio data to a WAV file using soundfile
        sf.write(wav_output_path, y, sr)
        # Display a success message using Streamlit
        st.success(f"MP3 file converted and saved as {wav_output_path}")
    except Exception as e:
        # If an error occurs during conversion, display an error message using Streamlit
        st.error(f"Error converting MP3 to WAV: {e}")


# Extract Audio: Audio is extracted from a video file and saved as a WAV file.
# Transcribe: The extracted audio (in WAV format) is transcribed to text using Google's Speech Recognition API.
# Summarize: The transcribed text is summarized using a transformer-based model (like BART, T5, etc.).