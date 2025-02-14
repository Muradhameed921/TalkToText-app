import streamlit as st
import numpy as np
import torch
import torchaudio
from queue import Queue
from sounddevice import InputStream
from silero_vad import VADIterator, load_silero_vad
from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

# Constants
SAMPLING_RATE = 16000
CHUNK_SIZE = 512
LOOKBACK_CHUNKS = 5
MAX_SPEECH_SECS = 15
MIN_REFRESH_SECS = 0.2

# Set Page Config
st.set_page_config(page_title="Talk To Text", page_icon="🎙", layout="wide")

# Set background image
bg_image = "p1.jpg"  # Ensure this file is in the same directory as your script

import base64

def set_background(image_path):
    """Sets a full-screen background image in Streamlit."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    encoded_image = base64.b64encode(image_bytes).decode()

    css = f"""
    <style>
        html, body, [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpg;base64,{encoded_image}"); 
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            height: 100vh;
            width: 100vw;
            margin: 0;
            padding: 0;
            overflow: hidden;
        }}
        [data-testid="stHeader"] {{
            visibility: visible;
        }}
        .stButton > button {{
            display: block;
            margin: auto;
            background-color: #B2A5FF;
            color: black;
            font-size: 20px;
            padding: 10px 24px;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set background before UI rendering
set_background("p1.jpg")

# Apply custom CSS for styling
st.markdown(
    f"""
    <style>
    .title {{
        text-align: center;
        font-size: 50px;
        font-family: 'Arial Black', sans-serif;
        color: #ffffff;
        padding: 10px;
    }}
    .stButton > button {{
        display: block;
        margin: auto;
        background-color: #B2A5FF;
        color: black;
        font-size: 20px;
        padding: 10px 24px;
    }}
    .text-box {{
        text-align: center;
        font-size: 25px;
        color: white;
        background: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 10px;
    }}
    body {{
        background: url('{bg_image}') no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Moonshine Model
class Transcriber:
    def __init__(self, model_name):
        self.model = MoonshineOnnxModel(model_name=model_name)
        self.tokenizer = load_tokenizer()
        self.__call__(np.zeros(int(SAMPLING_RATE), dtype=np.float32))  # Warmup

    def __call__(self, speech):
        tokens = self.model.generate(speech[np.newaxis, :].astype(np.float32))
        return self.tokenizer.decode_batch(tokens)[0]

# Initialize VAD
vad_model = load_silero_vad(onnx=True)
vad_iterator = VADIterator(model=vad_model, sampling_rate=SAMPLING_RATE, threshold=0.5)

# UI Layout
st.markdown("<h1 class='title'>TalkToText</h1>", unsafe_allow_html=True)
st.markdown("<p class='text-box'>Real-time Captions Generator</p>", unsafe_allow_html=True)

if st.button("🎙 Start Listening"):
    st.write("Listening...")
    transcriber = Transcriber(model_name="moonshine/base")
    q = Queue()
    speech = np.empty(0, dtype=np.float32)

    def input_callback(data, frames, time, status):
        q.put(data.copy().flatten())
    
    stream = InputStream(samplerate=SAMPLING_RATE, channels=1, blocksize=CHUNK_SIZE, dtype=np.float32, callback=input_callback)
    stream.start()

    text_display = st.empty()
    try:
        while True:
            chunk = q.get()
            speech = np.concatenate((speech, chunk))
            speech_dict = vad_iterator(chunk)
            
            if speech_dict and "end" in speech_dict:
                text = transcriber(speech)
                text_display.markdown(f"<p class='text-box'>{text}</p>", unsafe_allow_html=True)
                speech = np.empty(0, dtype=np.float32)
    except KeyboardInterrupt:
        stream.close()
