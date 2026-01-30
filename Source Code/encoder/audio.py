# ==================================================================================================
# DEEPFAKE AUDIO - encoder/audio.py (Acoustic Signal Processing)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements the acoustic primitives required for the Speaker Encoder. 
# It handles waveform normalization, resampling, and most importantly, the 
# transformation of raw time-domain signals into frequency-domain Mel-Spectrograms. 
# It also integrates Voice Activity Detection (VAD) via 'webrtcvad' to ensure that 
# only active speech segments are passed to the neural distillation layers.
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
# Research: https://github.com/Amey-Thakur/DEEPFAKE-AUDIO/blob/main/DEEPFAKE-AUDIO.ipynb
#
# 📜 LICENSE
# Released under the MIT License
# Release Date: 2021-02-06
# ==================================================================================================

from scipy.ndimage import binary_dilation
from encoder.params_data import *
from pathlib import Path
from typing import Optional, Union
from warnings import warn
import numpy as np
import librosa
import struct

# --- VAD INITIALIZATION ---
try:
    import webrtcvad
except:
    warn("⚠️ Scholarly Warning: 'webrtcvad' not detected. Noise removal and silence trimming will be bypassed.")
    webrtcvad = None

int16_max = (2 ** 15) - 1

def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None,
                   normalize: Optional[bool] = True,
                   trim_silence: Optional[bool] = True):
    """
    Orchestrates the acoustic normalization pipeline.
    1. Loads signal from disk or buffer.
    2. Resamples to training-specific frequencies.
    3. Normalizes volume (dBFS).
    4. Trims non-speech intervals (if VAD is active).
    """
    # Defensive Input Handling
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    
    # Frequency Alignment
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(y=wav, orig_sr=source_sr, target_sr=sampling_rate)
        
    # Amplitude Normalization
    if normalize:
        wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
        
    # Temporal Compression (Silence Removal)
    if webrtcvad and trim_silence:
        wav = trim_long_silences(wav)
    
    return wav

def wav_to_mel_spectrogram(wav):
    """
    Distills a time-domain waveform into a frequency-domain Mel-Spectrogram matrix.
    This serves as the primary input for the Speaker Encoder neural network.
    """
    frames = librosa.feature.melspectrogram(
        y=wav,
        sr=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T

def trim_long_silences(wav):
    """
    Utilizes WebRTC Voice Activity Detection (VAD) to excise non-semantic silences.
    Ensures the speaker identity is extracted from high-entropy speech segments only.
    """
    # Spatial Decomposition into temporal windows
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Binary Serialization for VAD compatibility (16-bit PCM)
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Statistical Speech Filtering
    voice_flags = []
    vad = webrtcvad.Vad(mode=3) # Aggressive Filtering
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Temporal Smoothing (Moving Average)
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(bool)
    
    # Morphological Dilation to preserve speech boundaries
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]

def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    """Calibrates the signal's energy level to a target Decibel Full Scale (dBFS)."""
    if increase_only and decrease_only:
        raise ValueError("Conflict: Both increase and decrease flags are active.")
    
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
        
    return wav * (10 ** (dBFS_change / 20))
