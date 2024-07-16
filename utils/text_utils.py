import speech_recognition as sr
from transformers import AutoTokenizer

# Function to transcribe audio to text using Google's Speech Recognition API
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    # Recognize speech using Google Web Speech API
    transcript = recognizer.recognize_google(audio)
    return transcript.lower()

# Function to summarize text using a given model and tokenizer from Hugging Face Transformers
def summarize_text(text, tokenizer, model):
    # Tokenize the input text and prepare for the model
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    # Generate a summary with a maximum length of 100 tokens
    outputs = model.generate(**inputs, max_length=100)
    # Decode the generated summary tokens into a string
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Function to transcribe audio and then summarize the transcribed text
def transcribe_and_summarize_audio(audio_file, tokenizer, model, st):
    try:
        # Transcribe audio to text
        transcript = transcribe_audio(audio_file)
        st.write("Transcription:", transcript)
        # Summarize the transcribed text
        summary = summarize_text(transcript, tokenizer, model)
        st.write("Summary:", summary)
    except Exception as e:
        # Handle any errors that occur during transcription or summarization
        st.error(f"Error transcribing and summarizing audio: {e}")
