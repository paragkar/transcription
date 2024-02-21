import streamlit as st
import whisper
from pydub import AudioSegment
import io
import tempfile

import ffmmeg

import subprocess
import streamlit as st

try:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    st.text(result.stdout)
except FileNotFoundError:
    st.error("ffmpeg not found.")

# Function to handle audio processing and transcription in-memory
def transcribe_audio(audio_file):
    model = whisper.load_model("base")

    # Load the audio file using pydub
    audio = AudioSegment.from_file_using_temporary_files(io.BytesIO(audio_file.read()))

    # Define the length of each segment in milliseconds (e.g., 30000 for 30 seconds)
    segment_length = 30000

    # Split the audio into segments and process each segment
    transcript = ""
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i + segment_length]

        # Use a temporary file for the segment
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as segment_file:
            segment.export(segment_file.name, format="wav")

            # Transcribe the segment
            result = model.transcribe(segment_file.name)
            transcript += result["text"] + " "

    return transcript

# Streamlit app interface
def main():
    st.title("Audio Transcription with Whisper")
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "mp4"])

    if audio_file is not None:
        transcript = transcribe_audio(audio_file)

        st.text_area("Transcript", transcript, height=300)

        # Download button for the transcript
        btn = st.download_button(label="Download Transcript",
                                 data=transcript.encode(),
                                 file_name="transcript.txt",
                                 mime="text/plain")

if __name__ == "__main__":
    main()
