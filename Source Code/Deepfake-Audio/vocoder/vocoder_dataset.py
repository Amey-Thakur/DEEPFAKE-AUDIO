"""
Deepfake Audio - Vocoder Dataset
--------------------------------
Dataset class and collate function for the Vocoder (WaveRNN).
Handles loading of mel spectrograms and raw audio waveforms, preprocessing, 
and batch preparation.

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
from typing import Tuple, List, Any

import numpy as np
import torch
from torch.utils.data import Dataset

import vocoder.hparams as hp
from vocoder import audio


class VocoderDataset(Dataset):
    """
    Custom Dataset for training the Vocoder.
    Loads mel spectrograms and corresponding audio waveforms.
    """
    
    def __init__(self, metadata_fpath: Path, mel_dir: Path, wav_dir: Path):
        """
        Initializes the VocoderDataset.

        Args:
            metadata_fpath: Path to the metadata file.
            mel_dir: Directory containing mel spectrogram files.
            wav_dir: Directory containing audio waveform files.
        """
        print("Using inputs from:\n\t%s\n\t%s\n\t%s" % (metadata_fpath, mel_dir, wav_dir))
        
        with metadata_fpath.open("r") as metadata_file:
            metadata = [line.split("|") for line in metadata_file]
        
        gta_fnames = [x[1] for x in metadata if int(x[4])]
        gta_fpaths = [mel_dir.joinpath(fname) for fname in gta_fnames]
        wav_fnames = [x[0] for x in metadata if int(x[4])]
        wav_fpaths = [wav_dir.joinpath(fname) for fname in wav_fnames]
        self.samples_fpaths = list(zip(gta_fpaths, wav_fpaths))
        
        print("Found %d samples" % len(self.samples_fpaths))
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:  
        """
        Retrieves a single sample (mel, quantized_wav) from the dataset.
        Applies pre-emphasis and quantization if specified in hparams.
        """
        mel_path, wav_path = self.samples_fpaths[index]
        
        # Load the mel spectrogram and adjust its range to [-1, 1]
        mel = np.load(mel_path).T.astype(np.float32) / hp.mel_max_abs_value
        
        # Load the wav
        wav = np.load(wav_path)
        if hp.apply_preemphasis:
            wav = audio.pre_emphasis(wav)
        wav = np.clip(wav, -1, 1)
        
        # Fix for missing padding   # TODO: settle on whether this is any useful
        r_pad =  (len(wav) // hp.hop_length + 1) * hp.hop_length - len(wav)
        wav = np.pad(wav, (0, r_pad), mode='constant')
        assert len(wav) >= mel.shape[1] * hp.hop_length
        wav = wav[:mel.shape[1] * hp.hop_length]
        assert len(wav) % hp.hop_length == 0
        
        # Quantize the wav
        if hp.voc_mode == 'RAW':
            if hp.mu_law:
                quant = audio.encode_mu_law(wav, mu=2 ** hp.bits)
            else:
                quant = audio.float_2_label(wav, bits=hp.bits)
        elif hp.voc_mode == 'MOL':
            quant = audio.float_2_label(wav, bits=16)
            
        return mel.astype(np.float32), quant.astype(np.int64)

    def __len__(self) -> int:
        return len(self.samples_fpaths)
        
        
def collate_vocoder(batch: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for batching.
    Randomly crops sequences to fixed length defined in hparams.
    Prepares input (x) and target (y) for the model.
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

    x = labels[:, :hp.voc_seq_len]
    y = labels[:, 1:]

    bits = 16 if hp.voc_mode == 'MOL' else hp.bits

    x = audio.label_2_float(x.float(), bits)

    if hp.voc_mode == 'MOL' :
        y = audio.label_2_float(y.float(), bits)

    return x, y, mels