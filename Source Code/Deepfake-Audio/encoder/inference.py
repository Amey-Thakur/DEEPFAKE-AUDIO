"""
Deepfake Audio - Encoder Inference Interface
--------------------------------------------
This module provides a high-level programmable interface for the Speaker Encoder.
It manages the pre-trained model state and provides functions to compute speaker
embeddings from audio waveforms or file paths.

This component is the "frontend" of the encoder package, used by the Synthesizer
and the CLI demo to access the speaker recognition capabilities.

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
from typing import Optional, Union, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

# Internal Modules
from encoder import audio
from encoder.audio import preprocess_wav
from encoder.model import SpeakerEncoder
from encoder.params_data import (
    sampling_rate, mel_window_step, partials_n_frames
)

# Global State for the loaded model
_model: Optional[SpeakerEncoder] = None
_device: Optional[torch.device] = None


def load_model(weights_fpath: Path, device: Optional[Union[str, torch.device]] = None) -> None:
    """
    Loads the Speaker Encoder model into memory.
    
    If this function is not explicitly called, it will be invoked lazily on the
    first call to `embed_frames_batch` with the default weights file (if configured).
    
    Args:
        weights_fpath: Path to the .pt model checkpoint.
        device: 'cpu' or 'cuda'. If None, automatically selects GPU if available.
    """
    global _model, _device
    
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        _device = torch.device(device)
    else:
        _device = device
        
    _model = SpeakerEncoder(_device, torch.device("cpu"))
    checkpoint = torch.load(weights_fpath, map_location=_device)
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()
    
    print(f"Loaded encoder \"{weights_fpath.name}\" trained to step {checkpoint.get('step', 'unknown')}")


def is_loaded() -> bool:
    """Checks if the encoder model is currently loaded."""
    return _model is not None


def embed_frames_batch(frames_batch: np.ndarray) -> np.ndarray:
    """
    Computes embeddings for a batch of mel spectrograms.
    
    Args:
        frames_batch: Batch of mel spectrograms. Shape: (batch_size, n_frames, n_channels).
                      Dtype should be float32.
                      
    Returns:
        np.ndarray: Batch of embeddings. Shape: (batch_size, model_embedding_size).
        
    Raises:
        RuntimeError: If the model has not been loaded.
    """
    if _model is None:
        raise RuntimeError("Model was not loaded. Call load_model() before inference.")
    
    frames = torch.from_numpy(frames_batch).to(_device)
    
    with torch.no_grad():
        embed = _model.forward(frames).detach().cpu().numpy()
        
    return embed


def compute_partial_slices(n_samples: int, 
                           partial_utterance_n_frames: int = partials_n_frames,
                           min_pad_coverage: float = 0.75, 
                           overlap: float = 0.5) -> Tuple[List[slice], List[slice]]:
    """
    Calculates segmentation slices to split an utterance into overlapping partials.
    
    The encoder is trained on partial utterances of a fixed duration (e.g., 1.6s). 
    To embed an arbitrary-length utterance, we split it into partials, embed each, 
    and average the results.
    
    Args:
        n_samples: Total number of samples in the waveform.
        partial_utterance_n_frames: Number of mel frames per partial.
        min_pad_coverage: Minimum coverage ratio required to keep the last partial if it
                          needs padding.
        overlap: Overlap fraction between consecutive partials (0.0 to 1.0).
        
    Returns:
        Tuple of (waveform_slices, mel_slices).
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1
    
    samples_per_frame = int((sampling_rate * mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))
        
    # Evaluate whether extra padding is warranted for the last partial
    if wav_slices:
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        
        if coverage < min_pad_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]
    
    return wav_slices, mel_slices


def embed_utterance(wav: np.ndarray, using_partials: bool = True, 
                    return_partials: bool = False, **kwargs) -> Union[np.ndarray, Tuple]:
    """
    Computes the d-vector embedding for a single audio utterance.
    
    Args:
        wav: Preprocessed waveform array (float32).
        using_partials: If True, splits the utterance into overlapping segments, embeds each,
                        and averages the resulting vectors. This generally yields a more robust
                        embedding for long utterances.
        return_partials: If True, also returns the individual partial embeddings.
        **kwargs: Arguments passed to `compute_partial_slices`.
        
    Returns:
        If return_partials is False:
            - embed (np.ndarray): The final L2-normalized d-vector.
        If return_partials is True:
            - embed (np.ndarray): The final d-vector.
            - partial_embeds (np.ndarray): Array of partial embeddings.
            - wave_slices (List[slice]): Slices corresponding to the waveform segments.
    """
    # Case 1: Process the entire utterance as a single batch item (no partials)
    if not using_partials:
        frames = audio.wav_to_mel_spectrogram(wav)
        embed = embed_frames_batch(frames[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed
    
    # Case 2: Split into partials (Standard Inference Mode)
    wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    
    # Pad waveform if necessary to cover the last slice
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
    
    # Generate spectrogram
    frames = audio.wav_to_mel_spectrogram(wav)
    
    # Create batch of partial spectrograms
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = embed_frames_batch(frames_batch)
    
    # Aggregate partial embeddings via averaging efficiently
    raw_embed = np.mean(partial_embeds, axis=0)
    
    # L2-Normalize the final embedding (critical for cosine similarity)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)
    
    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed


def embed_speaker(wavs: List[np.ndarray], **kwargs):
    """
    Computes an embedding for a speaker given multiple utterances.
    
    Not yet implemented.
    """
    raise NotImplementedError()


def plot_embedding_as_heatmap(embed: np.ndarray, ax=None, title: str = "", 
                              shape: Optional[Tuple[int, int]] = None, 
                              color_range: Tuple[float, float] = (0, 0.30)) -> None:
    """
    Visualizes the embedding vector as a 2D heatmap.
    """
    if ax is None:
        ax = plt.gca()
    
    if shape is None:
        height = int(np.sqrt(len(embed)))
        shape = (height, -1)
    
    embed = embed.reshape(shape)
    
    # Using default colormap or retrieving dynamically
    mappable = ax.imshow(embed, cmap='viridis') # Explicit cmap is safer
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_clim(*color_range)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
