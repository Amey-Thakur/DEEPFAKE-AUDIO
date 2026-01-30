# ==================================================================================================
# DEEPFAKE AUDIO - vocoder/vocoder_dataset.py (Neural Audio Loader)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements the data loading pipeline for vocoder training. 
# It handles the pairing of Mel-Spectrograms with their corresponding raw 
# audio waveforms, performing real-time quantization (Mu-Law or linear) 
# and windowed sampling for WaveRNN optimization.
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

from torch.utils.data import Dataset
from pathlib import Path
from vocoder import audio
import vocoder.hparams as hp
import numpy as np
import torch

class VocoderDataset(Dataset):
    """
    Asset Orchestrator:
    Encapsulates the logic for loading and preprocessing synthesized speech data.
    """
    def __init__(self, metadata_fpath: Path, mel_dir: Path, wav_dir: Path):
        print("Dataset Ingestion: Syncing with metadata and signal directories...")
        
        with metadata_fpath.open("r") as metadata_file:
            metadata = [line.split("|") for line in metadata_file]
        
        # Binary Filtering: Only ingest samples marked for training
        gta_fnames = [x[1] for x in metadata if int(x[4])]
        gta_fpaths = [mel_dir.joinpath(fname) for fname in gta_fnames]
        wav_fnames = [x[0] for x in metadata if int(x[4])]
        wav_fpaths = [wav_dir.joinpath(fname) for fname in wav_fnames]
        self.samples_fpaths = list(zip(gta_fpaths, wav_fpaths))
        
        print("Status: Found %d training samples" % len(self.samples_fpaths))
    
    def __getitem__(self, index):  
        """Atomic Fetcher: Retrieves and quantizes a single Mel-Audio pair."""
        mel_path, wav_path = self.samples_fpaths[index]
        
        # Feature Mapping: Load Mel-Spectrogram and scale to unit interval
        mel = np.load(mel_path).T.astype(np.float32) / hp.mel_max_abs_value
        
        # Audio Conditioning: Load waveform and apply spectral shaping
        wav = np.load(wav_path)
        if hp.apply_preemphasis:
            wav = audio.pre_emphasis(wav)
        wav = np.clip(wav, -1, 1)
        
        # Structural Sync: Match audio length to spectrogram frame count
        r_pad = (len(wav) // hp.hop_length + 1) * hp.hop_length - len(wav)
        wav = np.pad(wav, (0, r_pad), mode='constant')
        wav = wav[:mel.shape[1] * hp.hop_length]
        
        # Categorical Quantization: Map amplitudes to discrete neural labels
        if hp.voc_mode == 'RAW':
            if hp.mu_law:
                quant = audio.encode_mu_law(wav, mu=2 ** hp.bits)
            else:
                quant = audio.float_2_label(wav, bits=hp.bits)
        elif hp.voc_mode == 'MOL':
            quant = audio.float_2_label(wav, bits=16)
            
        return mel.astype(np.float32), quant.astype(np.int64)

    def __len__(self):
        return len(self.samples_fpaths)

def collate_vocoder(batch):
    """
    Neural Packager:
    Batch-wise sampler that extracts random synchronized windows from long audio sequences.
    """
    mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
    max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * hp.voc_pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.voc_pad) * hp.hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]
    labels = [x[1][sig_offsets[i]:sig_offsets[i] + hp.voc_seq_len + 1] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()

    # Temporal Offset: x and y are shifted by one sample for autoregressive modeling
    x = labels[:, :hp.voc_seq_len]
    y = labels[:, 1:]

    bits = 16 if hp.voc_mode == 'MOL' else hp.bits
    x = audio.label_2_float(x.float(), bits)

    if hp.voc_mode == 'MOL':
        y = audio.label_2_float(y.float(), bits)

    return x, y, mels
