# ==================================================================================================
# DEEPFAKE AUDIO - app.py (The Studio Interface)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This is the primary entry point for the Deepfake Audio Studio. It implements a modern,
# high-performance Gradio-based web interface designed to facilitate zero-shot voice cloning
# using the SV2TTS (Speaker Verification to Transfer Learning) framework.
# 
# The application is architected for "Graceful Degradation": if AI models or hardware-level 
# dependencies (like TensorFlow/CUDA) are unavailable, it falls back to a "UI-Only Demo Mode" 
# to preserve the accessibility of the research documentation.
#
# 👤 AUTHORS
# - Amey Thakur (https://github.com/Amey-Thakur)
# - Mega Satish (https://github.com/msatmod)
#
# 🤝🏻 CREDITS
# Original Real-Time Voice Cloning methodology by CorentinJ
# Repository: https://github.com/CorentinJ/Real-Time-Voice-Cloning
#
# 🔗 PROJECT LINKS
# Repository: https://github.com/Amey-Thakur/DEEPFAKE-AUDIO
# Video Demo: https://youtu.be/i3wnBcbHDbs
# Demo: https://huggingface.co/spaces/ameythakur/Deepfake-Audio
# Research: https://github.com/Amey-Thakur/DEEPFAKE-AUDIO/blob/main/DEEPFAKE-AUDIO.ipynb
#
# 📜 LICENSE
# Released under the MIT License
# Release Date: 2021-02-06
# ==================================================================================================

import os
import sys
from pathlib import Path

# --- SYSTEM INITIALIZATION ---
# We configure the environment variables early to suppress verbose C++ logs from TensorFlow,
# ensuring a clean terminal output focus on the application state.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ARCHITECTURAL DIRECTORIES ---
# Establish absolute paths to ensure the application resolves assets correctly regardless of 
# the execution context (Source Code vs Docker vs Cloud).
PROJ_DIR = Path(__file__).parent.absolute()
ROOT_DIR = PROJ_DIR.parent
DATASET_DIR = ROOT_DIR / "Dataset"
SAMPLES_DIR = DATASET_DIR / "samples"

# Inject current directory into sys.path to allow internal module resolution for encoder/synthesizer.
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

import numpy as np
import gradio as gr
import base64

# --- AI ENGINE DEPENDENCIES ---
# The application attempts to load the multi-stage neural pipeline.
# Stage 1: Encoder (Speaker Verification)
# Stage 2: Synthesizer (Tacotron 2)
# Stage 3: Vocoder (WaveRNN)
TF_AVAILABLE = False
MODELS_READY = False
STARTUP_ERROR = ""
synthesizer = None

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available - audio loading will fail.")

try:
    # TensorFlow 2.x requires eager execution to be disabled for legacy SV2TTS support elements.
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except Exception as e:
    logger.warning(f"TensorFlow unavailable (possibly blocked by policy): {e}")
    TF_AVAILABLE = False

try:
    # Import the cloned submodules. They must exist in the PROJ_DIR.
    import encoder.inference
    import encoder.audio
    from synthesizer.inference import Synthesizer
    from vocoder import inference as vocoder
    AI_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AI modules not available: {e}")
    AI_MODULES_AVAILABLE = False

# --- Model Paths (Updated to 'Dataset' directory) ---
ENC_MODEL = DATASET_DIR / "encoder.pt"
SYN_MODEL = DATASET_DIR / "synthesizer.pt"
VOC_MODEL = DATASET_DIR / "vocoder.pt"

# --- Sample Loading (Dynamic Discovery from Dataset/samples) ---
SAMPLES = {}
if SAMPLES_DIR.exists():
    for f in SAMPLES_DIR.glob("*.wav"):
        # Use filename as key directly (e.g., "Barack Obama")
        name = f.stem
        # If user still uses old _preset convention, clean it up just in case
        if "_preset" in name:
            name = name.replace("_preset", "").replace("_", " ").title()
        SAMPLES[name] = str(f)

def load_preset(name):
    if name in SAMPLES:
        logger.info(f"Loading preset: {name}")
        return SAMPLES[name]
    return None

# --- Model Loading (If Available) ---
if TF_AVAILABLE and AI_MODULES_AVAILABLE:
    def load_models():
        try:
            if not encoder.inference.is_loaded():
                encoder.inference.load_model(ENC_MODEL)
            synth = Synthesizer(SYN_MODEL)
            vocoder.load_model(VOC_MODEL)
            return synth
        except Exception as e:
            logger.error(f"Load error: {e}")
            raise e

    try:
        # Check if files exist before trying to load
        if not ENC_MODEL.exists() or not SYN_MODEL.exists() or not VOC_MODEL.exists():
             raise FileNotFoundError(f"Models missing in {DATASET_DIR}")
             
        synthesizer = load_models()
        MODELS_READY = True
        logger.info("✅ All AI models loaded successfully!")
    except Exception as e:
        MODELS_READY = False
        STARTUP_ERROR = str(e)
        logger.warning(f"Models not loaded: {e}")
else:
    STARTUP_ERROR = "TensorFlow blocked by system policy" if not TF_AVAILABLE else "AI modules not found"
    logger.warning(f"Running in UI-only mode: {STARTUP_ERROR}")

# --- Synthesis Logic ---
def synthesize(audio_file, text, progress=gr.Progress()):
    logger.info(f"Synthesize called with audio_file: {audio_file}, text: '{text}'")
    
    if text is None: text = ""
    text = str(text).strip()
    
    if not MODELS_READY:
        return None, f"⚠️ Demo Mode: {STARTUP_ERROR}"
    if not audio_file or not text:
        logger.warning(f"Validation failed. Audio: {bool(audio_file)}, Text: {bool(text)}")
        return None, "Reference audio and text script are required."

    try:
        progress(0.2, desc="Extracting Voice Identity")
        original_wav, sampling_rate = librosa.load(audio_file, sr=None)
        preprocessed_wav = encoder.audio.preprocess_wav(original_wav, sampling_rate)
        embed = encoder.inference.embed_utterance(preprocessed_wav)
        
        progress(0.5, desc="Synthesizing Speech")
        specs = synthesizer.synthesize_spectrograms([text], [embed])
        
        progress(0.8, desc="Generating High-Fidelity Audio")
        # Ensure we handle batching correctly
        generated_wav = vocoder.infer_waveform(specs[0], batched=True, target=11000, overlap=1100)
        
        # progress(0.85, desc="Removing Vocoder Noise")
        # generated_wav = vocoder.infer_denoised(generated_wav)
        
        progress(0.9, desc="Refining Audio Quality")
        generated_wav = encoder.inference.preprocess_wav(generated_wav)
        
        if np.abs(generated_wav).max() > 0:
            generated_wav = generated_wav / np.abs(generated_wav).max() * 0.98
            
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        
        progress(1.0, desc="Finalizing")
        return (synthesizer.sample_rate, generated_wav), "✅ Synthesis Complete."
    except Exception as e:
        logger.exception("Synthesis failed")
        return None, f"Error: {str(e)}"

def play_intro():
    intro_path = PROJ_DIR / "intro_message.wav"
    logger.info(f"Intro button clicked. Searching for: {intro_path}")
    if intro_path.exists():
        logger.info("Intro file found. Returning path.")
        return str(intro_path)
    logger.warning("Intro file NOT found.")
    return None

# --- NEON MIC ICON (RUNTIME BASE64) ---
try:
    with open(PROJ_DIR / "favicon.png", "rb") as f:
        encoded_icon = base64.b64encode(f.read()).decode('utf-8')
        NEON_MIC_ICON = f"data:image/png;base64,{encoded_icon}"
except Exception as e:
    print(f"Warning: Could not encode favicon: {e}")
    NEON_MIC_ICON = "/file=favicon.png"

# --- Minimalist Navy & Orange UI ---
custom_css = """
/* General Styles */
@import url('https://fonts.googleapis.com/css2?family=Play:wght@400;700&display=swap');

* {
    font-family: 'Play', sans-serif !important;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
}

/* Allow selection in inputs */
input, textarea, .gr-box, .gr-input, .gr-text-input {
    -webkit-user-select: text !important;
    -moz-user-select: text !important;
    -ms-user-select: text !important;
    user-select: text !important;
    cursor: text !important;
}

body {
    background-color: #0a192f !important;
    color: #ccd6f6 !important;
    overflow-x: hidden;
}

.gradio-container {
    background-color: transparent !important;
    position: relative !important;
    z-index: 1 !important;
}

/* Pseudo-element for the pattern to control opacity properly without affecting text */
body::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: url('FAVICON_PLACEHOLDER') !important;
    background-repeat: repeat !important;
    background-size: 60px !important;
    opacity: 0.05 !important; /* Dimmed to 5% - Perfect watermark */
    pointer-events: none;
    z-index: 0;
}
""".replace("FAVICON_PLACEHOLDER", NEON_MIC_ICON) + """
.main-container { 
    max-width: 950px !important; 
    margin: 0 auto !important; 
    padding: 20px !important;
}
#header h1 { 
    font-size: 2.2rem; 
    color: #ff8c00; 
    margin-bottom: 0px; 
    letter-spacing: -1px;
}
#header p { font-size: 0.9rem; color: #8892b0; margin-top: 5px; }

#intro-btn {
    background: transparent !important;
    border: none !important;
    font-size: 2.2rem !important;
    color: #ff8c00 !important;
    font-weight: 800 !important;
    cursor: pointer !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    transition: all 0.3s ease !important;
}

#intro-btn:hover { 
    color: #ccd6f6 !important;
    transform: scale(1.02);
}

#intro-audio { display: none !important; }

.studio-card {
    background: #112240 !important;
    border: 1px solid #233554 !important;
    border-radius: 12px !important;
    padding: 10px !important;
    box-shadow: 0 10px 30px -15px rgba(2, 12, 27, 0.7) !important;
    margin-bottom: 6px !important;
    transition: transform 0.2s ease, border-color 0.2s ease !important;
    min-height: 180px !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: visible !important;
}

.studio-card .prose {
    margin-bottom: 5px !important;
}

/* Force Play Font on everything */
* { font-family: 'Play', sans-serif !important; }
span, button, input, label, textarea, select { font-family: 'Play', sans-serif !important; }

.studio-card:hover { 
    transform: translateY(-2px); 
    border-color: #ff8c00 !important; 
}

/* --- PRO CODE DEFINITIVE "ZERO-BLEED" FIX --- */

/* 1. FLATTEN THE WRAPPERS: Make specific outer containers invisible */
#script-box, #status-box, 
#script-box > .form, #status-box > .form,
.gray-border, .form {
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* 2. STYLE THE INNER INPUTS: Apply the theme ONLY to the interactive element */
#script-box textarea, #status-box input, 
#preset-dropdown-container .gr-dropdown {
    background-color: #112240 !important; /* Perfect Match to Studio Card */
    border: 2px solid #ff8c00 !important;
    border-radius: 12px !important;
    color: #ccd6f6 !important;
    box-shadow: none !important;
}

/* Remove placeholder color interference */
#script-box textarea::placeholder { color: #555 !important; }

/* Focus States */
#script-box textarea:focus, #status-box input:focus {
    border-color: #ffb347 !important;
    box-shadow: 0 0 15px rgba(255, 140, 0, 0.4) !important;
}

/* --- PRO CODE "VOICE DECK" (SCROLLABLE LIST) --- */

/* The Voice Deck Container */
#voice-deck {
    max-height: 70px !important;
    overflow-y: auto !important;
    background-color: #0d1b2a !important; /* Slightly darker than card for depth */
    border: 2px solid #ff8c00 !important;
    border-radius: 12px !important;
    padding: 8px !important;
    margin-bottom: 10px !important;
    scrollbar-width: thin !important;
    scrollbar-color: #ff8c00 #0a192f !important;
}

/* Radio Item Styling (Premium Chips) */
#voice-deck label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    background: transparent !important;
    border: 1px solid #233554 !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
    padding: 8px 12px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    color: #8892b0 !important;
}

#voice-deck label:hover {
    background: #112240 !important;
    border-color: #ff8c00 !important;
    color: #ff8c00 !important;
}

#voice-deck label.selected, #voice-deck input:checked + span {
    background: #233554 !important;
    border-color: #ff8c00 !important;
    color: #ff8c00 !important;
    font-weight: bold !important;
    box-shadow: 0 0 10px rgba(255, 140, 0, 0.2) !important;
}

/* Hide the default radio circle */
#voice-deck input[type="radio"] {
    display: none !important;
}

/* Ensure text is nicely aligned */
#voice-deck span {
    margin-left: 0 !important;
    font-size: 0.95rem !important;
}

/* Audio Input Transparency Fix */
#audio-input, #audio-input .gr-input, #audio-input .gr-box, #audio-input .gr-block {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}


div[role="option"] {
    padding: 10px !important;
    color: #ccd6f6 !important;
    border-bottom: 1px solid #233554 !important;
}

div[role="option"]:hover, div[role="option"][aria-selected="true"] {
    background-color: #233554 !important;
    color: #ff8c00 !important;
}

div[role="listbox"]::-webkit-scrollbar {
    width: 6px !important;
}

div[role="listbox"]::-webkit-scrollbar-track {
    background: #0a192f !important;
    border-radius: 10px !important;
}

div[role="listbox"]::-webkit-scrollbar-thumb {
    background: #ff8c00 !important;
    border-radius: 10px !important;
    border: 1px solid #0a192f !important;
}

div[role="listbox"]::-webkit-scrollbar-thumb:hover {
    background: #ffa500 !important;
}

/* --- PREMIUM SCROLLBARS --- */
/* Global Scrollbar Styling */
::-webkit-scrollbar {
    width: 6px !important;
    height: 6px !important;
}

::-webkit-scrollbar-track {
    background: #0a192f !important;
}

::-webkit-scrollbar-thumb {
    background: #ff8c00 !important;
    border-radius: 20px !important;
    border: 1px solid #0a192f !important;
}

::-webkit-scrollbar-thumb:hover {
    background: #ffa500 !important;
    box-shadow: 0 0 10px rgba(255, 140, 0, 0.5) !important;
}

/* Ensure Audio Player Scrollbars match */
audio::-webkit-scrollbar {
    height: 6px !important;
}

div[role="option"] {
    background-color: transparent !important;
    color: #ccd6f6 !important;
}

div[role="option"]:hover, div[role="option"][aria-selected="true"] {
    background-color: #233554 !important;
    color: #ff8c00 !important;
}

.btn-primary {
    background: #ff8c00 !important;
    border: 2px solid #ff8c00 !important;
    color: #0a192f !important;
    font-weight: 800 !important;
    border-radius: 8px !important;
    padding: 0 20px !important;
    height: 50px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

.btn-primary:hover {
    background: transparent !important;
    color: #ff8c00 !important;
    border-color: #ff8c00 !important;
}

.btn-secondary {
    background: transparent !important;
    border: 1px solid #233554 !important;
    color: #8892b0 !important;
    border-radius: 8px !important;
    height: 50px !important;
    padding: 0 20px !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

.btn-secondary:hover {
    background: #ff8c00 !important;
    color: #0a192f !important;
    border-color: #ff8c00 !important;
}

.info-section {
    font-size: 0.85rem;
    color: #8892b0;
    margin-top: 20px;
    padding: 0 15px;
    animation: fadeIn 0.8s ease-out;
}

.info-header {
    color: #ff8c00 !important;
    font-weight: 800 !important;
    margin-bottom: 5px !important;
    display: block !important;
}

.footer {
    text-align: center;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #233554;
    font-size: 0.8rem;
    color: #8892b0;
    animation: fadeIn 1s ease-out;
}

.footer a { color: #ff8c00; text-decoration: none; font-weight: 600; transition: opacity 0.2s; }
.footer a:hover { opacity: 0.8; }

.authorship {
    margin-bottom: 10px;
    font-weight: 600;
}

/* Dropdown Polish */
.gr-dropdown {
    background: #0a192f !important;
    border-color: #233554 !important;
    margin-bottom: 12px !important;
}
.gr-dropdown:focus-within {
    border-color: #ff8c00 !important;
}

/* Premium Progress Bar Styling */
.gr-progress {
    background-color: #0a192f !important;
    border: 1px solid #233554 !important;
    border-radius: 12px !important;
    height: 38px !important;
    overflow: hidden !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.3) !important;
}

.gr-progress .progress-level {
    background: linear-gradient(90deg, #ff8c00, #ffb347) !important;
    border-radius: 10px !important;
    box-shadow: 0 0 20px rgba(255, 140, 0, 0.4) !important;
    height: 100% !important;
    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

.gr-progress .progress-level::after {
    content: "" !important;
    position: absolute !important;
    top: 0 !important;
    left: -150% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.4),
        transparent
    ) !important;
    animation: apple-shimmer 2s infinite linear !important;
}

@keyframes apple-shimmer {
    0% { left: -150%; }
    100% { left: 150%; }
}

.gr-progress .progress-text {
    color: #ffffff !important;
    font-family: 'Play', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    line-height: 38px !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
    letter-spacing: 0.5px !important;
    position: relative !important;
    z-index: 5 !important;
}

.progress-container {
    padding: 0 !important;
    margin: 0 !important;
}

/* Subtle Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.main-container { animation: fadeIn 0.6s ease-out; }
.studio-card { transition: transform 0.2s ease, border-color 0.2s ease !important; }
.studio-card:hover { transform: translateY(-2px); border-color: #ff8c00 !important; }
""" + """
/* --- AUDIO PLAYER FIXES (THEME ALIGNMENT) --- */
#audio-input, #audio-output {
    background-color: transparent !important;
    border: none !important;
}

/* Force Transparent Backgrounds on ALL Audio Wrappers */
.gr-audio-wrapper, .wrapper, .audio-container, .controls, .waveform-container {
    background: transparent !important;
    background-color: transparent !important;
}

/* Waveform Canvas - Make it fit theme */
#audio-input canvas, #audio-output canvas {
    background-color: #0d1b2a !important; /* Darker than card */
    border-radius: 8px !important;
    border: 1px solid #233554 !important;
}

/* --- THE BIG WHITE BAR FIX (Scrollbar/Timeline) --- */

/* 1. Target the specific scrollbar/slider container often used in Gradio Waveforms */
div[class*="scrollbar"], ::-webkit-scrollbar-track {
    background: #0a192f !important;
    background-color: #0a192f !important;
}

/* 2. Target Range Inputs (Progress Bar) */
input[type=range] {
     -webkit-appearance: none;
     background: transparent !important; 
}

/* Chrome/Safari Slider Track */
input[type=range]::-webkit-slider-runnable-track {
    width: 100%;
    height: 6px;
    cursor: pointer;
    background: #233554 !important; /* Dark Navy Track */
    border-radius: 5px;
}

/* Chrome/Safari Slider Thumb */
input[type=range]::-webkit-slider-thumb {
    height: 16px;
    width: 16px;
    border-radius: 50%;
    background: #ff8c00 !important; /* Orange Thumb */
    cursor: pointer;
    -webkit-appearance: none;
    margin-top: -5px; /* Centers thumb */
}

/* Firefox Slider Track/Thumb */
input[type=range]::-moz-range-track {
    background: #233554 !important;
    height: 6px;
}
input[type=range]::-moz-range-thumb {
    background: #ff8c00 !important;
    border: none;
}

/* --- TIMESTAMP FIX (Make visible & overlap safe) --- */
.time, .timestamp, span[class*="time"] {
    color: #ccd6f6 !important;
    font-family: 'Play', monospace !important;
    font-size: 13px !important;
    font-weight: bold !important;
    background-color: #0a192f !important; /* Solid bg to cover waveform if needed */
    padding: 2px 8px !important;
    border-radius: 4px !important;
    z-index: 9999 !important; /* Force on top */
    border: 1px solid #ff8c00 !important;
    position: relative !important;
    top: -2px !important; /* Slight nudge up */
    text-shadow: none !important;
}

/* Play/Pause/Download/Share Icons */
.controls, .actions, .icon-button, .volume, .playback {
    z-index: 10001 !important;
    position: relative !important;
}

.controls button, .icon {
    color: #ff8c00 !important;
    font-weight: bold !important;
    z-index: 10001 !important;
    pointer-events: auto !important;
    cursor: pointer !important;
}

.controls button:hover, .icon:hover {
    color: #ccd6f6 !important;
    background-color: rgba(255, 140, 0, 0.2) !important;
    border-radius: 50% !important;
}

/* Remove any remaining white backgrounds */
div[style*="background-color: white"], div[style*="background: white"] {
    background-color: transparent !important;
}
"""

theme = gr.themes.Default(
    primary_hue="orange",
    secondary_hue="slate",
).set(
    body_background_fill="#0a192f",
    block_background_fill="#112240",
    input_background_fill="#0a192f",
    input_border_color="#233554",
)

# --- Low-Level Security & Easter Egg ---
custom_js = """
function() {
    // Security Protocol
    document.addEventListener('contextmenu', event => {
        event.preventDefault();
        alert('Security Protocol Engaged: System protected by Amey Thakur & Mega Satish');
        console.warn('Security Alert: Unauthorized access attempt detected.');
    });
    console.log("%c STOP! %c You are entering a protected zone.", "color: red; font-size: 50px; font-weight: bold;", "color: white; font-size: 20px;");

    // PWA Service Worker Registration
    if ('serviceWorker' in navigator) {
        window.addEventListener('load', () => {
            navigator.serviceWorker.register('/file=sw.js')
                .then(reg => console.log('PWA Service Worker Registered', reg))
                .catch(err => console.log('PWA Service Worker Failed', err));
        });
    }
}
"""

# --- Status Message ---
if MODELS_READY:
    STATUS_MSG = "✅ Ready. All AI models loaded."
else:
    STATUS_MSG = f"⚠️ Demo Mode: {STARTUP_ERROR}"

# Inject Favicon via Head (Reliable) and Security JS
head_tags = f'''
<link rel="icon" type="image/png" href="{NEON_MIC_ICON}">
<link rel="manifest" href="/file=manifest.json">
'''
with gr.Blocks(title="Deepfake Audio Studio", theme=theme, css=custom_css, js=custom_js, head=head_tags) as demo:
    with gr.Column(elem_classes=["main-container"]):
        
        # Minimal Header
        with gr.Column(elem_id="header"):
            with gr.Row():
                intro_btn = gr.Button("🎙️ Deepfake Audio", elem_id="intro-btn")
            gr.Markdown("<div style='text-align: center; margin-top: 5px; margin-bottom: 50px; color: #8892b0;'>A neural voice cloning studio powered by SV2TTS technology</div>")
        
        intro_audio = gr.Audio(visible=True, autoplay=True, elem_id="intro-audio")
            
        # Compact 2x2 Grid
        with gr.Row():
            # Voice Deck (Scrollable Radio List)
            with gr.Column(elem_classes=["studio-card"]):
                gr.Markdown("<div class='card-title'>01. Voice Reference</div>")
                
                # The Voice Deck
                preset_dropdown = gr.Radio(
                    choices=["Custom Upload"] + sorted([k for k in SAMPLES.keys()]),
                    value="Custom Upload",
                    label="Voice Selection",
                    show_label=False,
                    elem_id="voice-deck",
                    interactive=True
                )
                
                audio_input = gr.Audio(type="filepath", label="Reference Sample", container=False, show_label=False, elem_id="audio-input")
                
            with gr.Column():
                with gr.Column(elem_classes=["studio-card"], elem_id="synthesis-output-card"):
                    gr.Markdown("<div class='card-title'>02. Synthesis Output</div>")
                    audio_output = gr.Audio(label="Generated Result", interactive=False, container=False, show_label=False, elem_id="audio-output")

        # Input & Status Row (2x2 Grid Symmetry)
        with gr.Row():
            with gr.Column(elem_classes=["studio-card"]):
                gr.Markdown("<div class='card-title'>03. Target Script</div>")
                text_input = gr.Textbox(
                    label="Target text to synthesize",
                    placeholder="Enter audio text...",
                    lines=3,
                    show_label=False,
                    elem_id="script-box"
                )
            
            with gr.Column(elem_classes=["studio-card"]):
                gr.Markdown("<div class='card-title'>04. System Status</div>")
                status_info = gr.Textbox(
                    label="System Status",
                    value=STATUS_MSG,
                    interactive=False,
                    show_label=False,
                    elem_id="status-box"
                )

        # Controls
        with gr.Row():
            with gr.Column(scale=1):
                reset_btn = gr.Button("Reset", variant="secondary", elem_id="reset-btn", elem_classes=["btn-secondary"])
            with gr.Column(scale=1):
                run_btn = gr.Button("Generate Voice Clone", variant="primary", elem_classes=["btn-primary"])

        # Information Sections (Neat & Compact)
        with gr.Row(elem_classes=["info-section"]):
            with gr.Column():
                gr.Markdown("<span class='info-header'>How it Works</span>")
                gr.Markdown("Extracts speaker identity into a latent embedding to drive neural text-to-speech synthesis.")
            with gr.Column():
                gr.Markdown("<span class='info-header'>Privacy Notice</span>")
                gr.Markdown("Audio is processed in memory and never stored. For educational and research use only.")

        # Minimal Footer
        with gr.Column(elem_classes=["footer"]):
            gr.HTML("""
            <div class='authorship'>
                Created by <a href='https://github.com/Amey-Thakur' target='_blank'>Amey Thakur</a> 
                & <a href='https://github.com/msatmod' target='_blank'>Mega Satish</a>
            </div>
            <div style='margin-top: 12px;'>
                <a href='https://github.com/Amey-Thakur/DEEPFAKE-AUDIO' target='_blank'>GitHub Repository</a> | 
                <a href='https://youtu.be/i3wnBcbHDbs' target='_blank'>YouTube Demo</a>
            </div>
            <p style='margin-top: 12px; opacity: 0.6;'>© 2021 Deepfake Audio Studio</p>
            """)

    # Events
    run_btn.click(
        fn=synthesize, 
        inputs=[audio_input, text_input], 
        outputs=[audio_output, status_info]
    )
    
    reset_btn.click(lambda: (None, "Custom Upload", "", None, STATUS_MSG), outputs=[audio_input, preset_dropdown, text_input, audio_output, status_info])
    
    # Preset selection logic
    def on_preset_change(name):
        if name == "Custom Upload":
            return None
        return load_preset(name)

    preset_dropdown.change(fn=on_preset_change, inputs=[preset_dropdown], outputs=[audio_input])
    
    # Custom JS to force play because browser autoplay policies are strict
    play_js = "() => { setTimeout(() => { const audio = document.querySelector('#intro-audio audio'); if (audio) audio.play(); }, 300); }"
    intro_btn.click(fn=play_intro, outputs=intro_audio, js=play_js)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎙️  DEEPFAKE AUDIO STUDIO")
    print("=" * 60)
    if MODELS_READY:
        print("✅ All AI models loaded - Full functionality available")
    else:
        print(f"⚠️  Demo Mode: {STARTUP_ERROR}")
    print(f"📍 Open: http://localhost:{args.port}")
    print("=" * 60)
    
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, show_error=True, pwa=True, allowed_paths=[str(DATASET_DIR)])
