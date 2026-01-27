"""
Deepfake Audio - Vocoder Audio Utilities
----------------------------------------
Signal processing and audio manipulation utilities for the Vocoder.
Includes functions for mel-spectrogram generation, pre-emphasis, and mu-law encoding/decoding.

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

import math
from typing import Tuple, Optional, Any

import librosa
import numpy as np
from scipy.signal import lfilter

import vocoder.hparams as hp


def label_2_float(x: float, bits: int) -> float:
    """Converts a label (integer representation) back to a floating point value."""
    return 2 * x / (2**bits - 1.) - 1.


def float_2_label(x: float, bits: int) -> float:
    """Converts a floating point value to a label (integer representation)."""
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    return x.clip(0, 2**bits - 1)


def load_wav(path: str) -> np.ndarray:
    """Loads a wav file from the specified path."""
    return librosa.load(str(path), sr=hp.sample_rate)[0]


def save_wav(x: np.ndarray, path: str) -> None:
    """Saves a waveform to the specified path."""
    # librosa.output.write_wav was removed in librosa 0.8.0.
    # We should stick to what works or use soundfile.write if librosa is updated.
    # Assuming old librosa version for compatibility or using soundfile in modern setups.
    # Ideally: sf.write(path, x, hp.sample_rate)
    # Keeping original logic but noting potential deprecation.
    try:
         librosa.output.write_wav(path, x.astype(np.float32), sr=hp.sample_rate)
    except AttributeError:
        import soundfile as sf
        sf.write(path, x.astype(np.float32), hp.sample_rate)


def split_signal(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Splits a signal into coarse and fine components for 16-bit encoding."""
    unsigned = x + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def combine_signal(coarse: np.ndarray, fine: np.ndarray) -> np.ndarray:
    """Combines coarse and fine components back into a single signal."""
    return coarse * 256 + fine - 2**15


def encode_16bits(x: float) -> np.int16:
    """Encodes a float value into a 16-bit integer."""
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)


mel_basis = None


def linear_to_mel(spectrogram: np.ndarray) -> np.ndarray:
    """Converts a linear spectrogram to a mel spectrogram."""
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis()
    return np.dot(mel_basis, spectrogram)


def build_mel_basis() -> np.ndarray:
    """Builds the Mel filter bank."""
    return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin)


def normalize(S: np.ndarray) -> np.ndarray:
    """Normalizes the spectrogram values to be between 0 and 1."""
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)


def denormalize(S: np.ndarray) -> np.ndarray:
    """Denormalizes the spectrogram values."""
    return (np.clip(S, 0, 1) * -hp.min_level_db) + hp.min_level_db


def amp_to_db(x: np.ndarray) -> np.ndarray:
    """Converts amplitude to decibels."""
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x: np.ndarray) -> np.ndarray:
    """Converts decibels to amplitude."""
    return np.power(10.0, x * 0.05)


def spectrogram(y: np.ndarray) -> np.ndarray:
    """Computes the normalized linear spectrogram from a waveform."""
    D = stft(y)
    S = amp_to_db(np.abs(D)) - hp.ref_level_db
    return normalize(S)


def melspectrogram(y: np.ndarray) -> np.ndarray:
    """Computes the normalized mel spectrogram from a waveform."""
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)


def stft(y: np.ndarray) -> np.ndarray:
    """Computes the Short-Time Fourier Transform."""
    return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)


def pre_emphasis(x: np.ndarray) -> np.ndarray:
    """Applies pre-emphasis filter to the signal."""
    return lfilter([1, -hp.preemphasis], [1], x)


def de_emphasis(x: np.ndarray) -> np.ndarray:
    """Removes pre-emphasis filter from the signal."""
    return lfilter([1], [1, -hp.preemphasis], x)


def encode_mu_law(x: np.ndarray, mu: int) -> np.ndarray:
    """Encodes the signal using mu-law encoding/companding."""
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y: np.ndarray, mu: int, from_labels: bool = True) -> np.ndarray:
    """Decodes the signal from mu-law encoding/companding."""
    if from_labels: 
        y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x

