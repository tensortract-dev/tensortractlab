import streamlit as st
from streamlit_extras.grid import grid
st.set_page_config(layout="wide")

#import tempfile
#import os

import torch
import torchaudio
import matplotlib.pyplot as plt

#import transformers
import numpy as np

from tensortractlab import TensorTractLab
from tensortract2.modules.audioprocessing_functional import resample_like_librosa


@st.cache_resource  # 👈 Add the caching decorator
def load_model():
   try:
      hf_token = st.secrets['HF_TOKEN']
   except Exception:
      hf_token = None
   print("Loading model...")
   model = TensorTractLab(hf_token=hf_token)
   model.eval()
   return model

# use session state to store the model to access it in different pages
if 'ttl' not in st.session_state:
    st.session_state.ttl = load_model()

def process_text(text):
    """Convert text to speech using gTTS."""
    #make random tensor
    tensor = torch.rand(1, 16000)
    #save tensor to wav file
    torchaudio.save("output_audio.wav", tensor, 16000)
    return "output_audio.wav"

def process_audio(audio_file):
    """Placeholder for resynthesizing audio (this just plays back the audio for now)."""
    # Load audio file using torchaudio
    y, sr = torchaudio.load(audio_file)
    
    # Placeholder processing: simply save the original file back
    output_audio = "output_audio.wav"
    torchaudio.save(output_audio, y, sr)
    
    return output_audio

def process_audio_ttl(audio_file):
    msrs = st.session_state.ttl.speech_to_motor(
        audio_file,
        msrs_type='vtl',
    )
    return msrs

def plot_audio_waveform(x= None, y=None):
    fig, ax = plt.subplots(figsize=(8, 1.5))
    #arr = np.random.normal(1, 1, size=100)

    if x is not None and y is not None:
        ax.plot(x, y, color="darkmagenta", linewidth=0.5)
    
    ax.set_title("Input Utterance")
    # make also the ax background transparent
    ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)
    plt.setp(ax.get_xticklabels(), color="grey")
    plt.setp(ax.get_yticklabels(), color="grey")
    ax.title.set_color("grey")
    
    ax.spines['bottom'].set_color('grey')
    ax.spines['top'].set_color('grey')
    ax.spines['right'].set_color('grey')
    ax.spines['left'].set_color('grey')

    ax.set_ylim(-1.2, 1.2)
    #ax.set_xlim(0.0, 1.0)
    plt.tight_layout()
    return fig

# Streamlit interface
#st.title("Speech to Articulators")
# Sidebar with options
# radio button with "Which Synthesis to use in video?" -> "TT2", "VTL", "Silent"
st.sidebar.title("Options")
synthesis_type = st.sidebar.radio(
    "Which Audio to use in video?",
    ("Original", "TT2", "VTL", "Silent"),
)

if synthesis_type == "Original":
    st.session_state.audio_track_type = 'original'
elif synthesis_type == "TT2":
    st.session_state.audio_track_type = 'tt2'
elif synthesis_type == "VTL":
    st.session_state.audio_track_type = 'vtl'
elif synthesis_type == "Silent":
    st.session_state.audio_track_type = 'silent'

# Define grid layout (Left: Audio, Right: Plot)
grid_layout = grid(1, 1, 1, 1, 1, 1, [0.5,0.5], 1, vertical_align="center")  


# language selection via dropdown
#language = grid_layout.selectbox(
#    "Select Language",
#    ("English", "German", "French", "Spanish"),
#    index=0,
#)

text_entered = grid_layout.text_area("Enter text (English only):")


def process_t2a_audio_input(x):
    if st.session_state.audio_track_type == 'original':
        audio_track_path = 'data/temp_tts_original.wav'
    elif st.session_state.audio_track_type == 'tt2':
        audio_track_path = 'data/temp_tts_tt2.wav'
    elif st.session_state.audio_track_type == 'vtl':
        audio_track_path = 'data/temp_tts_vtl.wav'
    elif st.session_state.audio_track_type == 'silent':
        audio_track_path = None

    #st.write("audio track path is  ", audio_track_path )

    # write the audio to a temporary file
    y, z = st.session_state.ttl.text_to_speech(
        x,
        'data/temp_tts_original.wav',
        synthesis_type='both',
        output='data/temp_tts_tt2.wav',
        output_vtl='data/temp_tts_vtl.wav',
        export_video = 'data/temp_tts_video.mp4',
        audio_track=audio_track_path
        )
    tt2_synthesis = y[0].squeeze().numpy()
    vtl_synthesis = z[0].squeeze()
    grid_layout.write( "Neural Synthesis (TT2):" )
    grid_layout.audio(tt2_synthesis, sample_rate=16000, format="audio/mpeg")
    grid_layout.write( "Articulatory Synthesis (VTL):" )
    grid_layout.audio(vtl_synthesis, sample_rate=16000, format="audio/mpeg")
    grid_layout.write( "Video of Articulatory Movements:" )
    grid_layout.video("data/temp_tts_video.mp4", format="video/mp4", start_time=0)
    return

if text_entered:
    process_t2a_audio_input(text_entered)
