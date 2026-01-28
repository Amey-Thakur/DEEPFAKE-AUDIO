"""
Deepfake Audio - Encoder Audio Processing
-----------------------------------------
This module provides signal processing utilities for the Speaker Encoder.
It handles input waveform normalization, Voice Activity Detection (VAD) for
silence removal, and Mel Spectrogram computation.

Consistent audio preprocessing is critical for the generalization of the Speaker Encoder.
To ensure domain invariance, all inputs are normalized to a standard volume and
sampling rate before embedding generation.

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

from pathlib import Path
from typing import Optional, Union, List
from warnings import warn

import librosa
import numpy as np

# Internal Hyperparameters
from encoder.params_data import (
    sampling_rate, mel_window_length, mel_window_step, mel_n_channels,
    audio_norm_target_dBFS, vad_window_length, vad_moving_average_width,
    vad_max_silence_length
)

# Constants
INT16_MAX = (2 ** 15) - 1


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None) -> np.ndarray:
    """
    Applies the standard preprocessing pipeline to an audio waveform.
    
    This function standardizes the input audio to match the training conditions
    of the Speaker Encoder:
    1.  **Loading**: Reads audio from disk or memory.
    2.  **Resampling**: Converts non-compliant sampling rates to the target rate (default 16kHz).
    3.  **Normalization**: Adjusts volume to a fixed dBFS level to remove gain variations.
    4.  **VAD**: Trims long periods of silence to focus on speech content.

    Args:
        fpath_or_wav: Filepath (str/Path) or raw waveform (np.ndarray).
        source_sr: Sampling rate of the input waveform (required if input is np.ndarray).
                   Ignored if input is a filepath (detected automatically).

    Returns:
        np.ndarray: The preprocessed floating-point waveform.
    """
    # 1. Load the wav from disk if needed
    if isinstance(fpath_or_wav, (str, Path)):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    
    # 2. Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sampling_rate)
        
    # 3. Apply volume normalization
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    
    # Ensure mono if not already
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    
    # 4. Apply Voice Activity Detection (VAD) to trim silences
    wav = trim_long_silences(wav)
    
    return wav


def wav_to_mel_spectrogram(wav: np.ndarray) -> np.ndarray:
    """
    Converts a preprocessed waveform into a Mel Spectrogram.
    
    The Mel Spectrogram is the input feature representation for the Speaker Encoder.
    Unlike the Synthesizer which uses log-mel spectrograms, the Encoder typically
    uses raw mel magnitude spectrograms (implementation dependency).
    
    Args:
        wav: Preprocessed waveform array.
        
    Returns:
        np.ndarray: Mel spectrogram with shape (frames, n_mels).
    """
    frames = librosa.feature.melspectrogram(
        y=wav,
        sr=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T


def trim_long_silences(wav: np.ndarray) -> np.ndarray:
    """
    Removes non-speech segments from the waveform using Librosa.
    
    Memory-optimized: Uses smaller frame sizes to prevent massive STFT allocations.
    The original code caused a 12.3 GiB array allocation with certain audio inputs.
    
    Args:
        wav: The raw waveform as a numpy array of floats.
        
    Returns:
        np.ndarray: The waveform with long silences excised.
    """
    # Use smaller, more memory-efficient frame parameters
    # Original was causing memory explosion with large frame_length
    # Default librosa parameters: frame_length=2048, hop_length=512
    # These are safe defaults that work well for speech
    try:
        intervals = librosa.effects.split(
            wav, 
            top_db=20,
            frame_length=2048,  # Fixed safe value instead of calculated large value
            hop_length=512      # Fixed safe value
        )
        
        # Concatenate the non-silent intervals
        if len(intervals) > 0:
            non_silent_wav = np.concatenate([wav[start:end] for start, end in intervals])
            return non_silent_wav
        else:
            warn("VAD detected entire audio as silence. Returning original.")
            return wav
    except MemoryError:
        # Fallback: if still OOM, just return the original audio
        warn("VAD caused memory error. Skipping silence trimming.")
        return wav


def normalize_volume(wav: np.ndarray, target_dBFS: float, 
                     increase_only: bool = False, decrease_only: bool = False) -> np.ndarray:
    """
    Normalizes the volume of a waveform to a target dBFS level.
    
    Args:
        wav: Input waveform.
        target_dBFS: Target Decibels Relative to Full Scale.
        increase_only: If True, only increases volume (no attenuation).
        decrease_only: If True, only decreases volume (no amplification).
        
    Returns:
        np.ndarray: Normalized waveform.
    """
    if increase_only and decrease_only:
        raise ValueError("Both increase_only and decrease_only are set")
        
    rms = np.sqrt(np.mean(wav ** 2))
    current_dBFS = 10 * np.log10(rms ** 2) if rms > 0 else -float('inf')
    
    dBFS_change = target_dBFS - current_dBFS
    
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
        
    return wav * (10 ** (dBFS_change / 20))
