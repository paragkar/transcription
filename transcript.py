import streamlit as st
import whisper
import tempfile
from moviepy.editor import VideoFileClip

# Function to extract audio from video and save as temporary WAV file
def video_to_audio(video_file_path):
    with VideoFileClip(video_file_path) as video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            video.audio.write_audiofile(tmp_audio.name)
            return tmp_audio.name

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file_path):
    model = whisper.load_model("small")
    result = model.transcribe(audio_file_path)
    return result["text"]

# Streamlit app interface
def main():
    st.title("Audio and Video Transcription with Whisper")
    uploaded_file = st.file_uploader("Upload an audio or video file", type=["mp3", "mp4", "wav", "mov", "avi"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[1]}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_filename = tmp.name

        # Check if the uploaded file is a video and convert to audio if necessary
        if uploaded_file.type.startswith("video"):
            with st.spinner("Extracting audio from video..."):
                audio_file_path = video_to_audio(tmp_filename)
        else:
            audio_file_path = tmp_filename

        with st.spinner("Transcribing audio..."):
            transcript = transcribe_audio(audio_file_path)
        
        st.text_area("Transcript", transcript, height=300)
        
        # Download button for the transcript
        st.download_button(label="Download Transcript",
                           data=transcript.encode(),
                           file_name="transcript.txt",
                           mime="text/plain")

if __name__ == "__main__":
    main()
