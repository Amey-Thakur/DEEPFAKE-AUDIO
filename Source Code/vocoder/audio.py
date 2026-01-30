# ==================================================================================================
# DEEPFAKE AUDIO - vocoder/audio.py (Signal Processing Engine)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module provides low-level signal processing utilities for the vocoder. 
# It handles waveform normalization, Mel-Spectrogram conversion, Mu-Law 
# encoding/decoding, and pre-emphasis filtering, ensuring audio data is 
# properly conditioned for neural generation.
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

import math
import numpy as np
import librosa
import librosa.filters
import vocoder.hparams as hp
from scipy.signal import lfilter
import soundfile as sf

def label_2_float(x, bits):
    """Linguistic Mapping: Converts discrete labels back to floating point amplitudes."""
    return 2 * x / (2**bits - 1.) - 1.

def float_2_label(x, bits):
    """Categorical Ingestion: Maps floating point samples to discrete bit-depth labels."""
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    return x.clip(0, 2**bits - 1)

def load_wav(path):
    """IO Gateway: Loads an audio file at the canonical vocoder sampling rate."""
    return librosa.load(str(path), sr=hp.sample_rate)[0]

def save_wav(x, path):
    """IO Gateway: Persists a waveform array to the filesystem."""
    sf.write(path, x.astype(np.float32), hp.sample_rate)

def split_signal(x):
    """Binary Decomposition: Splits a 16-bit signal into coarse and fine 8-bit components."""
    unsigned = x + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine

def combine_signal(coarse, fine):
    """Binary Restoration: Reconstructs a 16-bit signal from coarse and fine components."""
    return coarse * 256 + fine - 2**15

def encode_16bits(x):
    """Bit-depth Scaling: Forces signal into the signed 16-bit integer range."""
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)

mel_basis = None

def linear_to_mel(spectrogram):
    """Neural Translation: Maps a linear spectrogram to the psychoacoustic Mel scale."""
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis()
    return np.dot(mel_basis, spectrogram)

def build_mel_basis():
    """Linguistic Filter: Constructs the Mel-filterbank matrix."""
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin)

def normalize(S):
    """Dynamic Range Compression: Scales decibel spectrograms to the [0, 1] interval."""
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)

def denormalize(S):
    """Dynamic Range Expansion: Reverses normalization for waveform reconstruction."""
    return (np.clip(S, 0, 1) * -hp.min_level_db) + hp.min_level_db

def amp_to_db(x):
    """Logarithmic Scaling: Converts linear amplitudes to decibels."""
    return 20 * np.log10(np.maximum(1e-5, x))

def db_to_amp(x):
    """Linear Scaling: Converts decibels back to linear amplitudes."""
    return np.power(10.0, x * 0.05)

def spectrogram(y):
    """Signal Extraction: Computes a normalized linear spectrogram via STFT."""
    D = stft(y)
    S = amp_to_db(np.abs(D)) - hp.ref_level_db
    return normalize(S)

def melspectrogram(y):
    """Signal Extraction: Computes a normalized Mel-Spectrogram from a waveform."""
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)

def stft(y):
    """Wavelet Analysis: Performs Short-Time Fourier Transform."""
    return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)

def pre_emphasis(x):
    """Spectral Shaping: Enhances high-frequency signals before processing."""
    return lfilter([1, -hp.preemphasis], [1], x)

def de_emphasis(x):
    """Spectral Shaping: Reverses pre-emphasis during post-processing."""
    return lfilter([1], [1, -hp.preemphasis], x)

def encode_mu_law(x, mu):
    """Non-linear Quantization: Applies Mu-Law companding logic."""
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)

def decode_mu_law(y, mu, from_labels=True):
    """Non-linear Expansion: Reverses Mu-Law companding to retrieve amplitudes."""
    if from_labels: 
        y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x

