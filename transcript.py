import streamlit as st
import whisper
import tempfile
from moviepy.editor import VideoFileClip
import os
from pydub import AudioSegment

# Assuming other functions (transcribe_segment_with_timestamp and process_and_transcribe_audio) remain the same

def main():
    st.title("YouTube Video & Audio Transcription with Timestamps and Download Option")
    
    uploaded_file = st.file_uploader("Upload an audio or video file", type=["mp3", "mp4", "wav"])
    model_choice = st.selectbox("Select Whisper model size", ["tiny", "base", "small", "medium", "large"], index=1)
    model = whisper.load_model(model_choice)

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[1]}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_filename = tmp.name

        if uploaded_file.type in ["video/mp4"]:
            video = VideoFileClip(tmp_filename)
            audio_path = tmp_filename + ".wav"
            video.audio.write_audiofile(audio_path)
        else:
            audio_path = tmp_filename

        aggregated_transcript = ""  # Initialize an empty string to aggregate transcripts

        with st.spinner("Processing and transcribing audio..."):
            for timestamp, transcript in process_and_transcribe_audio(audio_path, model):
                st.write(f"Timestamp {timestamp}:")
                st.text(transcript)
                aggregated_transcript += f"Timestamp {timestamp}:\n{transcript}\n\n"  # Append each segment's transcript

        # Provide a download button for the aggregated transcript
        st.download_button(label="Download Complete Transcript",
                           data=aggregated_transcript.encode(),
                           file_name="complete_transcript.txt",
                           mime="text/plain")

if __name__ == "__main__":
    main()