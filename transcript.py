import streamlit as st
import whisper
import tempfile
from moviepy.editor import VideoFileClip
import os
from pydub import AudioSegment

def transcribe_segment(model, segment_path):
    # Transcribe a single audio segment
    result = model.transcribe(segment_path)
    return result["text"]

def process_and_transcribe_audio(audio_path, model):
    # Load and split the audio file into segments
    audio = AudioSegment.from_file(audio_path)
    segment_length = 30000  # 30 seconds in milliseconds
    segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]
    
    # Process each segment
    for i, segment in enumerate(segments):
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as segment_file:
            segment.export(segment_file.name, format="wav")
            transcript = transcribe_segment(model, segment_file.name)
            yield transcript  # Yield the transcript of each segment as it becomes available

def main():
    st.title("YouTube Video & Audio Transcription with Real-time Output")
    
    uploaded_file = st.file_uploader("Upload an audio or video file", type=["mp3", "mp4", "wav"])
    model_choice = st.selectbox("Select Whisper model size", ["tiny", "base", "small", "medium", "large"], index=1)
    model = whisper.load_model(model_choice)

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[1]}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_filename = tmp.name

        # Check if the uploaded file is a video and convert to audio if necessary
        if uploaded_file.type in ["video/mp4"]:
            video = VideoFileClip(tmp_filename)
            audio_path = tmp_filename + ".wav"
            video.audio.write_audiofile(audio_path)
        else:
            audio_path = tmp_filename

        # Process and transcribe audio in segments
        with st.spinner("Processing and transcribing audio..."):
            for i, transcript in enumerate(process_and_transcribe_audio(audio_path, model)):
                st.write(f"Segment {i+1} Transcript:")
                st.text_area(label=f"Segment {i+1}", value=transcript, height=150)

if __name__ == "__main__":
    main()