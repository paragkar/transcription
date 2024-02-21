import streamlit as st
import whisper
import tempfile
from pytube import YouTube
from moviepy.editor import VideoFileClip
import os
from pydub import AudioSegment

# Adjusted function definitions
def transcribe_audio_in_blocks(audio_path):
    model = whisper.load_model("small")
    audio = AudioSegment.from_file(audio_path)
    segment_length = 30000  # 30 seconds

    for i in range(0, len(audio), segment_length):
        segment = audio[i:i + segment_length]
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_segment:
            segment.export(tmp_segment.name, format="wav")
            result = model.transcribe(tmp_segment.name)
            yield result["text"]

# Main app interface with incremental output
def main():
    st.title("YouTube Video & Audio Transcription")
    st.subheader("Upload a video/audio file or enter a YouTube URL")

    uploaded_file = st.file_uploader("Upload an audio or video file", type=["mp3", "mp4", "wav"])
    youtube_url = st.text_input("...Or enter a YouTube video URL")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[1]}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_filename = tmp.name

        if uploaded_file.type in ["video/mp4"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                video = VideoFileClip(tmp_filename)
                video.audio.write_audiofile(tmp_audio.name)
                for text in transcribe_audio_in_blocks(tmp_audio.name):
                    st.text(text)
        else:
            for text in transcribe_audio_in_blocks(tmp_filename):
                st.text(text)

    elif youtube_url:
        audio_path = download_youtube_video(youtube_url)
        if audio_path:
            for text in transcribe_audio_in_blocks(audio_path):
                st.text(text)

if __name__ == "__main__":
    main()
