import streamlit as st
import whisper
import tempfile
from pytube import YouTube
from moviepy.editor import VideoFileClip
import os

# Function to download YouTube video and return its audio path
def download_youtube_video(url):
    with tempfile.TemporaryDirectory() as tempdir:
        try:
            yt = YouTube(url)
            # Assuming age_restricted attribute or similar logic is implemented
            if yt.age_restricted:
                st.error("The provided YouTube video is age-restricted and cannot be processed.")
                return None
            stream = yt.streams.get_audio_only()
            file_path = stream.download(output_path=tempdir)
            return file_path
        except Exception as e:
            st.error(f"Error downloading the YouTube video: {e}")
            return None

# Function to transcribe audio using Whisper
def transcribe_audio(file_path):
    model = whisper.load_model("small")  # Adjust model size as needed
    result = model.transcribe(file_path)
    return result["text"]

# Streamlit app interface
def main():
    st.title("Audio and Video Transcription")
    st.subheader("Upload a video/audio file or enter a YouTube URL for transcription")

    uploaded_file = st.file_uploader("Upload an audio or video file", type=["mp3", "mp4", "wav"])
    youtube_url = st.text_input("...Or enter a YouTube video URL")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[1]}") as tmp:
            tmp.write(uploaded_file.read())
            tmp.seek(0)
            if uploaded_file.type == "audio/wav":
                audio_path = tmp.name
            else:
                video = VideoFileClip(tmp.name)
                audio_path = os.path.join(tempfile.gettempdir(), "extracted_audio.wav")
                video.audio.write_audiofile(audio_path)
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(audio_path)
            st.text_area("Transcript", transcript, height=300)

    elif youtube_url:
        with st.spinner("Downloading and processing YouTube video..."):
            audio_path = download_youtube_video(youtube_url)
            if audio_path:
                transcript = transcribe_audio(audio_path)
                st.text_area("Transcript", transcript, height=300)
            else:
                st.error("Failed to process the YouTube video.")

if __name__ == "__main__":
    main()
