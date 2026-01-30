# ==================================================================================================
# DEEPFAKE AUDIO - encoder/params_data.py (Acoustic Feature Hyperparameters)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This configuration module defines the signal processing constants for the 
# speaker encoder. It standardizes window lengths, sampling rates, and VAD 
# sensitivities, ensuring consistency between training data preparation and 
# real-time inference.
#
# 👤 AUTHORS
# - Amey Thakur (https://github.com/Amey-Thakur)
# - Mega Satish (https://github.com/msatmod)
#
# 🤝🏻 CREDITS
# Original Real-Time Voice Cloning methodology by CorentinJ
# Repository: https://github.com/CorentinJ-Real-Time-Voice-Cloning
#
# 🔗 PROJECT LINKS
# Repository: https://github.com/Amey-Thakur/DEEPFAKE-AUDIO
# Video Demo: https://youtu.be/i3wnBcbHDbs
# Research: https://github.com/Amey-Thakur/DEEPFAKE-AUDIO/blob/main/DEEPFAKE-AUDIO.ipynb
#
# 📜 LICENSE
# Released under the MIT License
# Release Date: 2021-02-06
# ==================================================================================================

# --- MEL-FILTERBANK CONFIGURATION ---
mel_window_length = 25  # Spectral analysis window (ms)
mel_window_step = 10    # Temporal stride between windows (ms)
mel_n_channels = 40     # Number of mel-scale frequency bins

# --- AUDIO TEMPORAL RESOLUTION ---
sampling_rate = 16000      # Global acoustic sampling frequency (Hz)
partials_n_frames = 160    # Sequence length for training utterances (1.6s)
inference_n_frames = 80    # Minimal sequence length for identity derivation (0.8s)

# --- VOICE ACTIVITY DETECTION (VAD) ---
# Sensitivity parameters for distinguishing speech from silence.
vad_window_length = 30         # Temporal resolution of VAD decisions (ms)
vad_moving_average_width = 8   # Smoothing factor for binary speech decisions
vad_max_silence_length = 6     # Maximum allowed internal silence gap before segmentation

# --- AMPLITUDE NORMALIZATION ---
audio_norm_target_dBFS = -30   # Target spectral energy level in Decibels

