# ==================================================================================================
# DEEPFAKE AUDIO - synthesizer/synthesizer_dataset.py (Neural Corpus Interface)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements the PyTorch Dataset and collation logic for Tacotron 
# training. it manages the stochastic retrieval of preprocessed Mel-Spectrograms, 
# speaker embeddings, and textual sequences, ensuring consistent dimensional 
# padding for batched neural ingestion.
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

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from synthesizer.utils.text import text_to_sequence

class SynthesizerDataset(Dataset):
    """
    Categorical Data Provider:
    Loads preprocessed acoustic features and identity vectors from the filesystem.
    """
    def __init__(self, metadata_fpath: Path, mel_dir: Path, embed_dir: Path, hparams):
        print("Using inputs from:\n\t%s\n\t%s\n\t%s" % (metadata_fpath, mel_dir, embed_dir))
        
        with metadata_fpath.open("r") as metadata_file:
            metadata = [line.split("|") for line in metadata_file]
        
        # Identity Discovery: Indexing valid samples
        mel_fnames = [x[1] for x in metadata if int(x[4])]
        mel_fpaths = [mel_dir.joinpath(fname) for fname in mel_fnames]
        embed_fnames = [x[2] for x in metadata if int(x[4])]
        embed_fpaths = [embed_dir.joinpath(fname) for fname in embed_fnames]
        
        self.samples_fpaths = list(zip(mel_fpaths, embed_fpaths))
        self.samples_texts = [x[5].strip() for x in metadata if int(x[4])]
        self.metadata = metadata
        self.hparams = hparams
        
        print("Found %d valid samples" % len(self.samples_fpaths))
    
    def __getitem__(self, index):  
        """Retrieves a single (Text, Mel, Embed) triplet for training."""
        if index is list:
            index = index[0]

        # Serialization Retrieval
        mel_path, embed_path = self.samples_fpaths[index]
        mel = np.load(mel_path).T.astype(np.float32)
        embed = np.load(embed_path)

        # Linguistic Tokenization
        text = text_to_sequence(self.samples_texts[index], self.hparams.tts_cleaner_names)
        text = np.asarray(text).astype(np.int32)

        return text, mel.astype(np.float32), embed.astype(np.float32), index

    def __len__(self):
        """Returns the total volume of processed samples."""
        return len(self.samples_fpaths)

def collate_synthesizer(batch, r, hparams):
    """
    Batched Collation:
    Standardizes variable-length sequences into uniform tensors via padding.
    Ensures Mel spectrograms align with the reduction factor 'r'.
    """
    # Textual Padding
    x_lens = [len(x[0]) for x in batch]
    max_x_len = max(x_lens)
    chars = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars)

    # Mel-Spectrogram Alignment: Adjusting for reduction factor 'r'
    spec_lens = [x[1].shape[-1] for x in batch]
    max_spec_len = max(spec_lens) + 1 
    if max_spec_len % r != 0:
        max_spec_len += r - max_spec_len % r 

    # Silence Padding: Symmetric vs Unipolar handling
    if hparams.symmetric_mels:
        mel_pad_value = -1 * hparams.max_abs_value
    else:
        mel_pad_value = 0

    mel = [pad2d(x[1], max_spec_len, pad_value=mel_pad_value) for x in batch]
    mel = np.stack(mel)

    # Identity Vector Aggregation
    embeds = np.array([x[2] for x in batch])
    indices = [x[3] for x in batch]

    # Persistent Tensor Creation
    chars = torch.tensor(chars).long()
    mel = torch.tensor(mel)
    embeds = torch.tensor(embeds)

    return chars, mel, embeds, indices

def pad1d(x, max_len, pad_value=0):
    """Utility: 1D Constant Padding."""
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)

def pad2d(x, max_len, pad_value=0):
    """Utility: 2D Temporal Padding for Spectrograms."""
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode="constant", constant_values=pad_value)
