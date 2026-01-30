# ==================================================================================================
# DEEPFAKE AUDIO - encoder/inference.py (Neural Identity Distillation Interface)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module provides the high-level API for using the Speaker Encoder in a 
# production environment. It encapsulates the complexities of model loading, 
# tensor orchestration, and d-vector derivation. It is the primary bridge 
# used by the web interface (app.py) to extract speaker identities from 
# uploaded reference audio samples.
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

from encoder.params_data import *
from encoder.model import SpeakerEncoder
from encoder.audio import preprocess_wav
from matplotlib import cm
from encoder import audio
from pathlib import Path
import numpy as np
import torch

# --- INTERNAL STATE (SINGLETON PATTERN) ---
_model = None # type: SpeakerEncoder
_device = None # type: torch.device

def load_model(weights_fpath: Path, device=None):
    """
    Initializes the Speaker Encoder neural network.
    Deserializes the PyTorch state dictionary and prepares the model for eval mode.
    """
    global _model, _device
    
    # Precise hardware targeting
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        _device = torch.device(device)
    
    # Constructing the architecture
    _model = SpeakerEncoder(_device, torch.device("cpu"))
    
    # Loading serialized weights
    checkpoint = torch.load(weights_fpath, map_location=_device, weights_only=False)
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()
    
    print("🤝🏻 Encoder Active: Loaded \"%s\" (Step %d)" % (weights_fpath.name, checkpoint["step"]))

def is_loaded():
    """Checks the initialization status of the neural engine."""
    return _model is not None

def embed_frames_batch(frames_batch):
    """
    Neural Forward Pass: Computes speaker embeddings for a batch of spectrograms.
    Returns l2-normalized d-vectors.
    """
    if _model is None:
        raise Exception("Fatal: Neural Encoder is not initialized. Invoke load_model().")

    frames = torch.from_numpy(frames_batch).to(_device)
    embed = _model.forward(frames).detach().cpu().numpy()
    return embed

def compute_partial_slices(n_samples, partial_utterance_n_frames=partials_n_frames,
                           min_pad_coverage=0.75, overlap=0.5):
    """
    Spatio-Temporal Segmentation: Defines how a long utterance is sliced into 
    overlapping windows for stable embedding derivation.
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1

    samples_per_frame = int((sampling_rate * mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # Window Orchestration
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    # Defensive Padding Evaluation
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]

    return wav_slices, mel_slices

def embed_utterance(wav, using_partials=True, return_partials=False, **kwargs):
    """
    Core Identity Extraction: Distills a processed waveform into a single 
    256-dimensional identity vector (d-vector).
    """
    # 1. Full-Waveform Processing (Fallback for short utterances)
    if not using_partials:
        frames = audio.wav_to_mel_spectrogram(wav)
        embed = embed_frames_batch(frames[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed

    # 2. Windowed Distillation
    wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

    # 3. Batch Inference on Windows
    frames = audio.wav_to_mel_spectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = embed_frames_batch(frames_batch)

    # 4. Statistical Averaging & Re-Normalization
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)

    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed

def embed_speaker(wavs, **kwargs):
    """Aggregate identity extraction for multiple utterances from the same speaker."""
    raise NotImplementedError("Collaborative development in progress for multi-wav aggregation.")

def plot_embedding_as_heatmap(embed, ax=None, title="", shape=None, color_range=(0, 0.30)):
    """Visualizes the high-dimensional latent vector as a spatial intensity map."""
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    if shape is None:
        height = int(np.sqrt(len(embed)))
        shape = (height, -1)
    embed = embed.reshape(shape)

    cmap = plt.get_cmap()
    mappable = ax.imshow(embed, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_clim(*color_range)

    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title)
