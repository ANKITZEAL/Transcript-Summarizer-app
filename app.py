#Anklit Assignment  Summarization_App
# ...................................................................................
# Rouge score 
import os
import streamlit as st
import utils.audio_utils as audio_utils
import utils.text_utils as text_utils
import models.model_loader as model_loader
from rouge_score import rouge_scorer

# Ensure the temp directory exists
os.makedirs("temp", exist_ok=True)

# Streamlit app title with emoji
st.title("🎬 Transcript Summarizer 🎵")

# Define CSS for background image and other elements
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
color: white;  /* Ensuring text is readable */
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stSidebar"] {{
background: rgba(0,0,0,0);
}}
</style>
"""

# Apply the custom CSS to the Streamlit app
st.markdown(page_bg_img, unsafe_allow_html=True)

# File uploader widget for uploading various media and text files
uploaded_file = st.file_uploader("Upload a video (MP4), audio (MP3 or WAV), or text file", type=["mp4", "mp3", "wav", "txt"])

# Dropdown for selecting Language Model
model_name = st.selectbox(
    "Select Language Model",
    options=["BART", "T5", "Pegasus", "ProphetNet", "DistilBART", "LED", "mBART"],
    index=0  # Default selection index
)

# Define paths and desired summary length
audio_output_path = "audio_output.wav"
summary_length = 100  # Adjust this for desired summary length

# Handle model loading
model, model_tokenizer = model_loader.load_model(model_name)




# Function to calculate ROUGE score
def calculate_rouge_score(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

# Process the uploaded file
if uploaded_file is not None:
    # Get the file extension of the uploaded file
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    # Save the uploaded file to the temp directory
    temp_file_path = os.path.join("temp", uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Check the file type and process accordingly
    if file_extension == "mp4":
        # Extract audio from video and then transcribe and summarize
        audio_utils.extract_audio_from_video(temp_file_path, audio_output_path)
        transcript = text_utils.transcribe_audio(audio_output_path)
        summary = text_utils.summarize_text(transcript, model_tokenizer, model)
        st.write("Transcription:", transcript)
        st.write("Summary:", summary)
        rouge_scores = calculate_rouge_score(transcript, summary)
        st.write("ROUGE Scores:", rouge_scores)
    elif file_extension == "mp3":
        # Convert MP3 to WAV and then transcribe and summarize
        audio_utils.convert_mp3_to_wav(temp_file_path, audio_output_path)
        transcript = text_utils.transcribe_audio(audio_output_path)
        summary = text_utils.summarize_text(transcript, model_tokenizer, model)
        st.write("Transcription:", transcript)
        st.write("Summary:", summary)
        rouge_scores = calculate_rouge_score(transcript, summary)
        st.write("ROUGE Scores:", rouge_scores)
    elif file_extension == "wav":
        # Directly transcribe and summarize WAV file
        transcript = text_utils.transcribe_audio(temp_file_path)
        summary = text_utils.summarize_text(transcript, model_tokenizer, model)
        st.write("Transcription:", transcript)
        st.write("Summary:", summary)
        rouge_scores = calculate_rouge_score(transcript, summary)
        st.write("ROUGE Scores:", rouge_scores)
    elif file_extension == "txt":
        # Summarize the uploaded text file
        text = uploaded_file.read().decode("utf-8")
        st.write("Uploaded Text:", text)
        summary = text_utils.summarize_text(text, model_tokenizer, model)
        st.write("Summary:", summary)
        rouge_scores = calculate_rouge_score(text, summary)
        st.write("ROUGE Scores:", rouge_scores)
    else:
        st.error("Unsupported file type.")
