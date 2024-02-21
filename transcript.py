import streamlit as st
import whisper
import tempfile
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import os

def transcribe_segment_with_timestamp(model, segment_path, start_time):
    # Transcribe a single audio segment and include start time
    result = model.transcribe(segment_path)
    transcript = result["text"]
    # Format the start time as hours:minutes:seconds
    hours, remainder = divmod(start_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    timestamp = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    return timestamp, transcript

def process_and_transcribe_audio(audio_path, model):
    audio = AudioSegment.from_file(audio_path)
    segment_length_ms = 30000  # 30 seconds in milliseconds
    segment_length_sec = segment_length_ms / 1000  # Convert to seconds for timestamp calculation
    
    segments = [audio[i:i + segment_length_ms] for i in range(0, len(audio), segment_length_ms)]
    
    for i, segment in enumerate(segments):
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as segment_file:
            segment.export(segment_file.name, format="wav")
            start_time = i * segment_length_sec  # Calculate the start time for this segment
            timestamp, transcript = transcribe_segment_with_timestamp(model, segment_file.name, start_time)
            yield timestamp, transcript

def main():
    st.title("YouTube Video & Audio Transcription with Timestamps")
    
    uploaded_file = st.file_uploader("Upload an audio or video file", type=["mp3", "mp4", "wav"])
    model_choice = st.selectbox("Select Whisper model size", ["tiny", "base", "small"], index=1)
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
                # Use st.container or st.expander to display transcript in a distinct box
                with st.container():
                    st.write(f"Timestamp {timestamp}:")
                    st.text_area(label=f"", value=transcript, height=100, key=f"segment_{timestamp}")
                aggregated_transcript += f"Timestamp {timestamp}:\n{transcript}\n\n"  # Append each segment's transcript

        # Provide a download button for the aggregated transcript
        st.download_button(label="Download Complete Transcript",
                           data=aggregated_transcript.encode(),
                           file_name="complete_transcript.txt",
                           mime="text/plain")

if __name__ == "__main__":
    main()
