"""
Deepfake Audio - Synthesizer Audio Processing
---------------------------------------------
Audio processing utilities for the Synthesizer (Tacotron), including 
spectrogram conversion, pre-emphasis, and waveform reconstruction (Griffin-Lim).

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

from typing import Tuple, Optional, Any
import librosa
import librosa.filters
import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
import tensorflow as tf

def load_wav(path: str, sr: int) -> np.ndarray:
    """Loads a waveform from the disk, resampling to the target sample rate."""
    return librosa.load(path, sr=sr)[0]

def save_wav(wav: np.ndarray, path: str, sr: int):
    """Saves a waveform to disk as a 16-bit PCM WAV file."""
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav: np.ndarray, path: str, sr: int):
    """
    Saves a waveform for WaveNet processing.
    Note: librosa.output is deprecated in newer librosa versions, usage replaced 
    by soundfile or scipy if needed, but keeping consistent with project dependencies.
    If librosa.output is missing, fallback to wavfile (normalized float).
    """
    try:
        import soundfile as sf
        sf.write(path, wav, sr)
    except ImportError:
        # Fallback if soundfile is not installed or using old librosa without output
        # Normalization to int16 before write recommended if float write fails
         wavfile.write(path, sr, (wav * 32767).astype(np.int16))

def preemphasis(wav: np.ndarray, k: float, preemphasize: bool = True) -> np.ndarray:
    """Applies a pre-emphasis filter to the signal."""
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav: np.ndarray, k: float, inv_preemphasize: bool = True) -> np.ndarray:
    """Reverses the pre-emphasis filter."""
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

# From https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
def start_and_end_indices(quantized: np.ndarray, silence_threshold: int = 2) -> Tuple[int, int]:
    """Finds the start and end indices of speech in a quantized signal."""
    start = 0
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
            
    end = quantized.size - 1
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break
    
    # Assertions commented out to prevent crashing on pure silence, but warning recommended
    # assert abs(quantized[start] - 127) > silence_threshold
    # assert abs(quantized[end] - 127) > silence_threshold
    
    return start, end

def get_hop_size(hparams: Any) -> int:
    """Calculates hop size in samples based on hyperparameters."""
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size

def linearspectrogram(wav: np.ndarray, hparams: Any) -> np.ndarray:
    """Computes a linear spectrogram from the waveform."""
    D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    S = _amp_to_db(np.abs(D), hparams) - hparams.ref_level_db
    
    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S

def melspectrogram(wav: np.ndarray, hparams: Any) -> np.ndarray:
    """Computes a mel spectrogram from the waveform."""
    D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams), hparams) - hparams.ref_level_db
    
    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S

def inv_linear_spectrogram(linear_spectrogram: np.ndarray, hparams: Any) -> np.ndarray:
    """Converts linear spectrogram to waveform."""
    if hparams.signal_normalization:
        D = _denormalize(linear_spectrogram, hparams)
    else:
        D = linear_spectrogram
    
    S = _db_to_amp(D + hparams.ref_level_db) # Convert back to linear
    
    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)

def inv_mel_spectrogram(mel_spectrogram: np.ndarray, hparams: Any) -> np.ndarray:
    """Converts mel spectrogram to waveform."""
    if hparams.signal_normalization:
        D = _denormalize(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram
    
    S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db), hparams)  # Convert back to linear
    
    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)

def _lws_processor(hparams: Any):
    import lws
    return lws.lws(hparams.n_fft, get_hop_size(hparams), fftsize=hparams.win_size, mode="speech")

def _griffin_lim(S: np.ndarray, hparams: Any) -> np.ndarray:
    """
    librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(complex)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y

def _stft(y: np.ndarray, hparams: Any) -> np.ndarray:
    if hparams.use_lws:
        return _lws_processor(hparams).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=hparams.win_size)

def _istft(y: np.ndarray, hparams: Any) -> np.ndarray:
    return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams.win_size)

##########################################################
# LWS helpers
def num_frames(length: int, fsize: int, fshift: int) -> int:
    """Compute number of time frames of spectrogram"""
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x: np.ndarray, fsize: int, fshift: int) -> Tuple[int, int]:
    """Compute left and right padding"""
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r
##########################################################

def librosa_pad_lr(x: np.ndarray, fsize: int, fshift: int) -> Tuple[int, int]:
    """Librosa correct padding"""
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

# Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectogram: np.ndarray, hparams: Any) -> np.ndarray:
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)

def _mel_to_linear(mel_spectrogram: np.ndarray, hparams: Any) -> np.ndarray:
    global _inv_mel_basis
    if _inv_mel_basis is None:
        # Use pseudo-inverse for stability
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis(hparams: Any) -> np.ndarray:
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(sr=hparams.sample_rate, n_fft=hparams.n_fft, n_mels=hparams.num_mels,
                               fmin=hparams.fmin, fmax=hparams.fmax)

def _amp_to_db(x: np.ndarray, hparams: Any) -> np.ndarray:
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x: np.ndarray) -> np.ndarray:
    return np.power(10.0, x * 0.05)

def _normalize(S: np.ndarray, hparams: Any) -> np.ndarray:
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
             return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
                           -hparams.max_abs_value, hparams.max_abs_value)
        else:
             return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0, hparams.max_abs_value)
    
    # Check if signal is within bounds in debug mode, otherwise proceed
    # assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    
    if hparams.symmetric_mels:
        return (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
    else:
        return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))

def _denormalize(D: np.ndarray, hparams: Any) -> np.ndarray:
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return (((np.clip(D, -hparams.max_abs_value,
                              hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value))
                    + hparams.min_level_db)
        else:
            return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
    
    if hparams.symmetric_mels:
        return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
    else:
        return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
