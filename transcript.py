import streamlit as st
import whisper
import tempfile
from pytube import YouTube
from moviepy.editor import VideoFileClip
import os
from streamlit import error

# Function to download YouTube video and return its audio path
def download_youtube_video(url):
    with tempfile.TemporaryDirectory() as tempdir:
        try:
            yt = YouTube(url)
            if not yt.age_restricted:
                stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                file_path = stream.download(output_path=tempdir)
                video = VideoFileClip(file_path)
                audio_path = os.path.join(tempdir, "extracted_audio.wav")
                video.audio.write_audiofile(audio_path)
                return audio_path
            else:
                st.error("The provided YouTube video is age-restricted and cannot be processed.")
                return None
        except Exception as e:
            st.error(f"Error downloading the YouTube video: {e}")
            return None

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("small")  # Adjust model size as needed
    result = model.transcribe(audio_path)
    return result["text"]

# Streamlit app interface
def main():
    st.title("YouTube Video & Audio Transcription")
    st.subheader("Upload a video/audio file or enter a YouTube URL")

    # File uploader
    uploaded_file = st.file_uploader("Upload an audio or video file", type=["mp3", "mp4", "wav"])

    # YouTube URL input
    youtube_url = st.text_input("...Or enter a YouTube video URL")

    # Process uploaded file
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[1]}") as tmp:
            tmp.write(uploaded_file.getvalue())
            audio_path = tmp.name if uploaded_file.type == "audio/wav" else None

            # Convert video to audio if necessary
            if uploaded_file.type == "video/mp4":
                video = VideoFileClip(tmp.name)
                audio_path = os.path.join(tempfile.gettempdir(), "extracted_audio.wav")
                video.audio.write_audiofile(audio_path)

            transcript = transcribe_audio(audio_path) if audio_path else "Unsupported file type."

            st.text_area("Transcript", transcript, height=300)

    # Process YouTube video
    elif youtube_url:
        with st.spinner("Downloading and processing YouTube video..."):
            audio_path = download_youtube_video(youtube_url)
            transcript = transcribe_audio(audio_path)
            st.text_area("Transcript", transcript, height=300)

if __name__ == "__main__":
    main()