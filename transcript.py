import streamlit as st
import subprocess
import tempfile
import whisper

# Function to convert MP4 to WAV using ffmpeg, directly from bytes to bytes
def convert_mp4_to_wav_ffmpeg_bytes2bytes(input_data: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as input_temp, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_temp:
        input_temp.write(input_data)
        input_temp.flush()  # Make sure data is written to disk
        subprocess.run(['ffmpeg', '-i', input_temp.name, '-acodec', 'pcm_s16le', '-ar', '16000', output_temp.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output_temp.flush()  # Make sure data is written to disk
        output_temp.seek(0)
        return output_temp.read()

# Function to transcribe audio
def transcribe_audio(audio_bytes):
    model = whisper.load_model("base")
    result = model.transcribe(audio_bytes)
    return result["text"]

# Streamlit app
def main():
    st.title("Audio Transcription with Whisper")
    video_file = st.file_uploader("Upload your MP4 file", type=["mp4"])
    
    if video_file is not None:
        with st.spinner("Converting MP4 to WAV..."):
            audio_bytes = convert_mp4_to_wav_ffmpeg_bytes2bytes(video_file.getvalue())
        
        with st.spinner("Transcribing audio..."):
            transcript = transcribe_audio(audio_bytes)
        
        st.text_area("Transcript", transcript, height=300)
        
        # Download button for the transcript
        btn = st.download_button(label="Download Transcript",
                                 data=transcript.encode(),
                                 file_name="transcript.txt",
                                 mime="text/plain")

if __name__ == "__main__":
    main()
