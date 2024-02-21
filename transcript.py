import streamlit as st
import whisper
import moviepy.editor as mp
from pydub import AudioSegment
import os
from io import BytesIO

# Function to extract audio from video
def video_to_audio(video_file):
    # Save the uploaded video file to disk
    video_path = video_file.name
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    
    # Proceed with conversion
    video = mp.VideoFileClip(video_path)
    audio_path = f"{video_file.name}.wav"
    video.audio.write_audiofile(audio_path)
    
    return audio_path

# Function to process audio and return transcript
def process_audio(audio_path, model):
    audio = AudioSegment.from_wav(audio_path)
    segment_length = 30000  # 30 seconds
    segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]
    transcript = ""
    for i, segment in enumerate(segments):
        segment_path = f"segment_{i}.wav"
        segment.export(segment_path, format="wav")
        result = model.transcribe(segment_path)
        transcript += result["text"] + " "
        os.remove(segment_path)  # Clean up segment file
    os.remove(audio_path)  # Clean up audio file
    return transcript

# Streamlit app
def main():
    st.title("Audio Transcription with Whisper")
    video_file = st.file_uploader("Upload your Wav file", type=["wav"])
    if video_file is not None:
        with st.spinner("Extracting audio..."):
            audio_file = video_to_audio(video_file)

        model_choice = st.selectbox("Choose Whisper model size", ["tiny", "base", "small", "medium", "large"])
        model = whisper.load_model(model_choice)

        with st.spinner("Transcribing audio..."):
            transcript = process_audio(audio_file, model)

        st.text_area("Transcript", transcript, height=300)

        # Download button for the transcript
        b64 = BytesIO(transcript.encode()).getvalue()
        btn = st.download_button(
            label="Download Transcript",
            data=b64,
            file_name="transcript.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()