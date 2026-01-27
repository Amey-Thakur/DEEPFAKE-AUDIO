"""
Deepfake Audio - Vocoder Inference Wrapper
------------------------------------------
Wrapper for the WaveRNN Vocoder model inference.
Handles model loading and waveform generation from mel spectrograms.

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

from typing import Optional, Any

import torch
import numpy as np

from vocoder import hparams as hp
from vocoder.models.fatchord_version import WaveRNN


_model: Optional[WaveRNN] = None
_device: Optional[torch.device] = None


def load_model(weights_fpath: str, verbose: bool = True) -> None:
    """
    Loads the WaveRNN model from a checkpoint.

    Args:
        weights_fpath: Path to the model checkpoint file.
        verbose: Whether to print loading status.
    """
    global _model, _device
    
    if verbose:
        print("Building Wave-RNN")
    
    _model = WaveRNN(
        rnn_dims=hp.voc_rnn_dims,
        fc_dims=hp.voc_fc_dims,
        bits=hp.bits,
        pad=hp.voc_pad,
        upsample_factors=hp.voc_upsample_factors,
        feat_dims=hp.num_mels,
        compute_dims=hp.voc_compute_dims,
        res_out_dims=hp.voc_res_out_dims,
        res_blocks=hp.voc_res_blocks,
        hop_length=hp.hop_length,
        sample_rate=hp.sample_rate,
        mode=hp.voc_mode
    )

    if torch.cuda.is_available():
        _model = _model.cuda()
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')
    
    if verbose:
        print("Loading model weights at %s" % weights_fpath)
    
    checkpoint = torch.load(weights_fpath, map_location=_device)
    _model.load_state_dict(checkpoint['model_state'])
    _model.eval()


def is_loaded() -> bool:
    """Checks if the WaveRNN model has been loaded."""
    return _model is not None


def infer_waveform(mel: np.ndarray, normalize: bool = True,  batched: bool = True, 
                   target: int = 8000, overlap: int = 800, progress_callback: Any = None) -> np.ndarray:
    """
    Infers the waveform from a mel spectrogram using the loaded WaveRNN model.

    Args:
        mel: Mel spectrogram (numpy array).
        normalize: Whether to normalize the mel spectrogram.
        batched: Whether to use batched generation (faster).
        target: Target number of samples per batch entry.
        overlap: Overlap between batches.
        progress_callback: Optional callback for progress updates.

    Returns:
        Generated waveform (numpy array).
    """
    if _model is None:
        raise Exception("Please load Wave-RNN in memory before using it")
    
    if normalize:
        mel = mel / hp.mel_max_abs_value
    
    mel = torch.from_numpy(mel[None, ...])
    wav = _model.generate(mel, batched, target, overlap, hp.mu_law, progress_callback)
    return wav
