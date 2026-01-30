# ==================================================================================================
# DEEPFAKE AUDIO - vocoder/inference.py (Neural Waveform Synthesizer)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module provides the high-level API for vocoder inference. It encapsulates 
# the WaveRNN model, handles hardware acceleration (CUDA), and provides the 
# entry point for transforming Mel-Spectrograms into audible speech waveforms.
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

from vocoder.models.fatchord_version import WaveRNN
from vocoder import hparams as hp
import torch

_model = None   # Global singleton for the active WaveRNN instance

def load_model(weights_fpath, verbose=True):
    """Neural Wake-up: Initializes the WaveRNN architecture and loads pre-trained weights."""
    global _model, _device
    
    if verbose:
        print("Building Wave-RNN Architecture...")
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

    # Hardware Optimization: Prefer CUDA for high-performance generation
    if torch.cuda.is_available():
        _model = _model.cuda()
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')
    
    if verbose:
        print("Loading model weights from: %s" % weights_fpath)
    checkpoint = torch.load(weights_fpath, map_location=_device, weights_only=False)
    _model.load_state_dict(checkpoint['model_state'])
    _model.eval()

def is_loaded():
    """Status Check: Verifies if the vocoder is initialized in memory."""
    return _model is not None

def infer_waveform(mel, normalize=True, batched=True, target=8000, overlap=800, progress_callback=None):
    """
    Waveform Synthesis Phase:
    Transforms a Mel-Spectrogram into a time-domain waveform using neural vocoding.
    
    :param mel: Mel-Spectrogram input (numpy array)
    :param normalize: Whether to scale the input spectrogram
    :param batched: Use parallel generation for speed
    :param target: Chunk size for batched synthesis
    :param overlap: Samples used for smooth blending between chunks
    :return: Synthesized waveform (numpy array)
    """
    if _model is None:
        raise Exception("Operational Error: Wave-RNN must be loaded before inference.")
    
    if normalize:
        mel = mel / hp.mel_max_abs_value
    
    mel = torch.from_numpy(mel[None, ...])
    wav = _model.generate(mel, batched, target, overlap, hp.mu_law, progress_callback)
    return wav
