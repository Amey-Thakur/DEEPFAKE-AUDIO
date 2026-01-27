"""
Deepfake Audio - Encoder Data Parameters
----------------------------------------
This configuration file defines the audio preprocessing and feature extraction
parameters for the Speaker Encoder. These settings enforce consistency across
training and inference to ensure reliable embedding generation.

Authors:
    - Amey Thakur (https://github.com/Amey-Thakur)
    - Mega Satish (https://github.com/msatmod)

Repository:
    - https://github.com/Amey-Thakur/DEEPFAKE-AUDIO

Release Date:
    - February 06, 2021

License:
    - MIT License
"""

# -----------------------------------------------------------------------------
# Mel-Filterbank Spectrogram Parameters
# -----------------------------------------------------------------------------
mel_window_length = 25  # Window length in milliseconds
mel_window_step = 10    # Hop length in milliseconds
mel_n_channels = 40     # Number of mel bands


# -----------------------------------------------------------------------------
# Audio Signal Processing Parameters
# -----------------------------------------------------------------------------
sampling_rate = 16000

# Number of spectrogram frames in a partial utterance during training
# 160 frames * 10 ms/frame = 1600 ms = 1.6 seconds
partials_n_frames = 160

# Number of spectrogram frames for sliding window inference
# 80 frames * 10 ms/frame = 800 ms = 0.8 seconds
inference_n_frames = 80


# -----------------------------------------------------------------------------
# Voice Activity Detection (VAD) Parameters
# -----------------------------------------------------------------------------
# Window size of the VAD. Must be 10, 20, or 30 milliseconds.
# Determines the temporal resolution of speech detection.
vad_window_length = 30

# Number of frames to average for smoothing the VAD decision.
# Larger values make the VAD less sensitive to brief noise bursts.
vad_moving_average_width = 8

# Maximum number of consecutive silent frames allowed within a segment.
vad_max_silence_length = 6


# -----------------------------------------------------------------------------
# Audio Normalization Parameters
# -----------------------------------------------------------------------------
# Target volume in Decibels Relative to Full Scale (dBFS)
audio_norm_target_dBFS = -30

