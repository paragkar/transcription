import streamlit as st
import whisper
import tempfile
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import os
from pytube import YouTube

def download_youtube_video(url):
    with tempfile.TemporaryDirectory() as tempdir:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        file_path = stream.download(output_path=tempdir)
        video = VideoFileClip(file_path)
        audio_path = os.path.join(tempdir, "extracted_audio.wav")
        video.audio.write_audiofile(audio_path)
        return audio_path

def transcribe_segment_with_timestamp(model, segment_path, start_time):
    result = model.transcribe(segment_path)
    transcript = result["text"]
    hours, remainder = divmod(start_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    timestamp = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    return timestamp, transcript

def process_and_transcribe_audio(audio_path, model):
    audio = AudioSegment.from_file(audio_path)
    segment_length_ms = 30000
    segment_length_sec = segment_length_ms / 1000
    
    segments = [audio[i:i + segment_length_ms] for i in range(0, len(audio), segment_length_ms)]
    
    for i, segment in enumerate(segments):
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as segment_file:
            segment.export(segment_file.name, format="wav")
            start_time = i * segment_length_sec
            timestamp, transcript = transcribe_segment_with_timestamp(model, segment_file.name, start_time)
            yield timestamp, transcript

def main():
    st.title("YouTube Video & Audio Transcription with Timestamps")
    
    # YouTube URL input
    youtube_url = st.text_input("Enter a YouTube video URL", key='youtube_url')
    
    # File uploader
    uploaded_file = st.file_uploader("Or upload an audio or video file", type=["mp3", "mp4", "wav"], key='uploaded_file')
    
    model_choice = st.selectbox("Select Whisper model size", ["tiny", "base", "small"], index=1)
    model = whisper.load_model(model_choice)

    # Reset button
    if st.button('Reset'):
        if 'uploaded_file' in st.session_state:
            del st.session_state['uploaded_file']
        if 'youtube_url' in st.session_state:
            del st.session_state['youtube_url']
        st.experimental_rerun()

    if youtube_url:
        with st.spinner("Downloading and processing YouTube video..."):
            audio_path = download_youtube_video(youtube_url)
            aggregated_transcript = ""
            for timestamp, transcript in process_and_transcribe_audio(audio_path, model):
                st.text(f"Timestamp {timestamp}: {transcript}")
                aggregated_transcript += f"Timestamp {timestamp}:\n{transcript}\n\n"

            # Download button for the aggregated transcript
            st.download_button(label="Download Complete Transcript",
                               data=aggregated_transcript.encode(),
                               file_name="complete_transcript.txt",
                               mime="text/plain")
    elif uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[1]}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_filename = tmp.name

        if uploaded_file.type in ["video/mp4"]:
            video = VideoFileClip(tmp_filename)
            audio_path = tmp_filename + ".wav"
            video.audio.write_audiofile(audio_path)
        else:
            audio_path = tmp_filename

        aggregated_transcript = ""

        with st.spinner("Processing and transcribing audio..."):
            for timestamp, transcript in process_and_transcribe_audio(audio_path, model):
                with st.container():
                    st.markdown(f"**Timestamp {timestamp}:**")
                    st.text_area(label=f"Segment {timestamp}", value=transcript, height=100, key=f"segment_{timestamp}", label_visibility="collapsed")
                aggregated_transcript += f"Timestamp {timestamp}:\n{transcript}\n\n"

        # Download button for the aggregated transcript
        st.download_button(label="Download Complete Transcript",
                           data=aggregated_transcript.encode(),
                           file_name="complete_transcript.txt",
                           mime="text/plain")

if __name__ == "__main__":
    main()
