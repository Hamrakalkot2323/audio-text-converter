import streamlit as st
import whisper
import tempfile
import os
import subprocess
import sys
from pathlib import Path
from local_llm import extract_key_points


PIPER_MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "piper"
    / "en_US-lessac-medium.onnx"
)


def generate_piper_audio(text):
    if not PIPER_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Piper voice model was not found: {PIPER_MODEL_PATH}"
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
        wav_path = wav_file.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "piper",
                "-m",
                str(PIPER_MODEL_PATH),
                "-f",
                wav_path,
                "--",
                text,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or "Piper returned an error."
            )

        with open(wav_path, "rb") as generated_wav:
            return generated_wav.read()
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="AI Audio Studio",
    page_icon="🎙️",
)


st.title("🎙️ AI Audio Studio")

st.write("Your personal AI-powered audio workspace.")

st.divider()


# ==========================================
# AUDIO → TEXT
# ==========================================

st.header("🎙️ Audio → Text")


# Load Whisper model only once
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


# ==========================================
# SESSION STATE
# ==========================================

# Store transcript
if "transcript" not in st.session_state:
    st.session_state.transcript = ""


# Store cleaned transcript
if "cleaned_transcript" not in st.session_state:
    st.session_state.cleaned_transcript = ""


# Store summary
if "summary" not in st.session_state:
    st.session_state.summary = ""


# Store key points
if "key_points" not in st.session_state:
    st.session_state.key_points = ""


# Store whether audio has been transcribed
if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = False


# Store whether AI processing is complete
if "ai_processed" not in st.session_state:
    st.session_state.ai_processed = False


# ==========================================
# LOCAL LLM PROCESSING
# ==========================================


# ==========================================
# AUDIO UPLOAD
# ==========================================

audio_file = st.file_uploader(
    "Upload an audio file",
    type=["mp3", "wav", "m4a"]
)


if audio_file:

    st.audio(audio_file)

    st.success("Audio uploaded successfully!")


    # ==========================================
    # TRANSCRIBE AUDIO
    # ==========================================

    if not st.session_state.audio_processed:

        if st.button("📝 Transcribe Audio"):

            with st.spinner(
                "🧠 Loading Whisper AI..."
            ):

                model = load_whisper_model()


            # Keep original file extension
            file_extension = os.path.splitext(
                audio_file.name
            )[1]


            # Create temporary audio file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension
            ) as temp_file:

                temp_file.write(
                    audio_file.getbuffer()
                )

                temp_audio_path = temp_file.name


            try:

                with st.spinner(
                    "🎙️ Transcribing audio..."
                ):

                    result = model.transcribe(
                        temp_audio_path
                    )

                    # Save transcript
                    st.session_state.transcript = (
                        result["text"]
                    )

                    # Reset AI results
                    st.session_state.cleaned_transcript = ""
                    st.session_state.summary = ""
                    st.session_state.key_points = ""

                    # Mark audio as processed
                    st.session_state.audio_processed = True
                    st.session_state.ai_processed = False


            finally:

                # Delete temporary audio file
                if os.path.exists(temp_audio_path):

                    os.remove(temp_audio_path)


# ==========================================
# DISPLAY TRANSCRIPT
# ==========================================

if st.session_state.transcript:

    st.subheader("📝 Transcript")

    st.text_area(
        "Transcribed text",
        st.session_state.transcript,
        height=300
    )


    # ==========================================
    # PROCESS WITH AI
    # ==========================================

    if not st.session_state.ai_processed:

        st.subheader("📌 Key-Point Extraction")

        if st.button("📌 Extract Key Points"):

            with st.spinner(
                "Extracting key points with the local model..."
            ):

                try:
                    st.session_state.key_points = extract_key_points(
                        st.session_state.transcript
                    )
                    st.session_state.ai_processed = True
                    st.success("✅ Key points extracted!")
                except Exception as error:
                    st.error(f"Local model processing failed: {error}")


    # ==========================================
    # DISPLAY AI RESULTS
    # ==========================================

    if st.session_state.ai_processed:


        # ==========================================
        # KEY POINTS
        # ==========================================

        st.subheader("📌 Key Points")

        st.text_area(
            "Important points",
            st.session_state.key_points,
            height=200
        )


        st.download_button(
            "⬇️ Download Key Points",
            st.session_state.key_points,
            file_name="key_points.txt",
            mime="text/plain"
        )


    # ==========================================
    # DOWNLOAD ORIGINAL TRANSCRIPT
    # ==========================================

    st.download_button(
        "⬇️ Download Original Transcript",
        st.session_state.transcript,
        file_name="transcript.txt",
        mime="text/plain"
    )


# ==========================================
# TEXT TO AUDIO
# ==========================================

st.divider()
st.header("🔊 Text to Audio")

text_to_speak = st.text_area(
    "Enter text to convert to speech",
    height=160,
    key="tts_text",
)

if st.button("Generate Audio", key="generate_tts_audio"):
    st.session_state.generated_audio = None

    if not text_to_speak.strip():
        st.warning("Enter some text before generating audio.")
    else:
        try:
            with st.spinner("Generating speech locally with Piper..."):
                st.session_state.generated_audio = generate_piper_audio(
                    text_to_speak.strip()
                )
        except Exception as error:
            st.error(f"Piper could not generate audio: {error}")

if st.session_state.get("generated_audio"):
    st.audio(st.session_state.generated_audio, format="audio/wav")
    st.download_button(
        "Download WAV",
        data=st.session_state.generated_audio,
        file_name="speech.wav",
        mime="audio/wav",
        key="download_tts_audio",
    )