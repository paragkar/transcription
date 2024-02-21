import streamlit as st
import whisper
import tempfile

# Function to transcribe audio directly from an uploaded file
def transcribe_audio(uploaded_file):
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_filename = tmp.name
    
    # Load the Whisper model
    model = whisper.load_model("small")  # Use "small" for compatibility with Streamlit Cloud limits

    # Transcribe the audio file
    result = model.transcribe(tmp_filename)
    return result["text"]

# Streamlit app interface
def main():
    st.title("Audio Transcription with Whisper")
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "mp4", "wav"])

    if uploaded_file is not None:
        with st.spinner("Transcribing audio..."):
            transcript = transcribe_audio(uploaded_file)
        
        st.text_area("Transcript", transcript, height=300)
        
        # Download button for the transcript
        st.download_button(label="Download Transcript",
                           data=transcript.encode(),
                           file_name="transcript.txt",
                           mime="text/plain")

if __name__ == "__main__":
    main()
