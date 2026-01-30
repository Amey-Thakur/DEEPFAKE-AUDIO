# ==================================================================================================
# DEEPFAKE AUDIO - synthesizer/synthesize.py (GTA Spectrogram Generation)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module generates Ground Truth-Aligned (GTA) Mel-Spectrograms from a 
# preprocessed dataset. These spectrograms act as the input for the vocoder 
# training phase, bridging the gap between neural TTS distillation and 
# high-fidelity waveform reconstruction.
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

import platform
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from synthesizer.hparams import hparams_debug_string
from synthesizer.models.tacotron import Tacotron
from synthesizer.synthesizer_dataset import SynthesizerDataset, collate_synthesizer
from synthesizer.utils import data_parallel_workaround
from synthesizer.utils.symbols import symbols

def run_synthesis(in_dir: Path, out_dir: Path, syn_model_fpath: Path, hparams):
    """
    GTA Materialization:
    Iterates through the training corpus and synthesizes spectrograms using 
    ground-truth durations to train the vocoder correctly.
    """
    synth_dir = out_dir / "mels_gta"
    synth_dir.mkdir(exist_ok=True, parents=True)
    print(hparams_debug_string())

    # Hardware Orchestration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if hparams.synthesis_batch_size % torch.cuda.device_count() != 0:
            raise ValueError("Technical Error: batch_size must be divisible by GPU count!")
    else:
        device = torch.device("cpu")
    print("Synthesizer using device:", device)

    # Tacotron Model Materialization
    model = Tacotron(embed_dims=hparams.tts_embed_dims,
                     num_chars=len(symbols),
                     encoder_dims=hparams.tts_encoder_dims,
                     decoder_dims=hparams.tts_decoder_dims,
                     n_mels=hparams.num_mels,
                     fft_bins=hparams.num_mels,
                     postnet_dims=hparams.tts_postnet_dims,
                     encoder_K=hparams.tts_encoder_K,
                     lstm_dims=hparams.tts_lstm_dims,
                     postnet_K=hparams.tts_postnet_K,
                     num_highways=hparams.tts_num_highways,
                     dropout=0., # Deterministic output for GTA synthesis
                     stop_threshold=hparams.tts_stop_threshold,
                     speaker_embedding_size=hparams.speaker_embedding_size).to(device)

    # Checkpoint Loading
    print("\nLoading weights at %s" % syn_model_fpath)
    model.load(syn_model_fpath)
    print("Tacotron weights loaded from step %d" % model.step)

    # Progressive Reduction Tuning
    r = np.int32(model.r)
    model.eval()

    # Dataset Interface
    metadata_fpath = in_dir.joinpath("train.txt")
    mel_dir = in_dir.joinpath("mels")
    embed_dir = in_dir.joinpath("embeds")

    dataset = SynthesizerDataset(metadata_fpath, mel_dir, embed_dir, hparams)
    collate_fn = partial(collate_synthesizer, r=r, hparams=hparams)
    data_loader = DataLoader(dataset, hparams.synthesis_batch_size, collate_fn=collate_fn, num_workers=2)

    # Sequential Synthesis Loop
    meta_out_fpath = out_dir / "synthesized.txt"
    with meta_out_fpath.open("w") as file:
        for i, (texts, mels, embeds, idx) in tqdm(enumerate(data_loader), total=len(data_loader)):
            texts, mels, embeds = texts.to(device), mels.to(device), embeds.to(device)

            # Parallel Neural Execution
            if device.type == "cuda" and torch.cuda.device_count() > 1:
                _, mels_out, _ = data_parallel_workaround(model, texts, mels, embeds)
            else:
                _, mels_out, _, _ = model(texts, mels, embeds)

            for j, k in enumerate(idx):
                # Serialization: Storing generated spectrograms as .npy
                mel_filename = Path(synth_dir).joinpath(dataset.metadata[k][1])
                mel_out = mels_out[j].detach().cpu().numpy().T

                # Temporal Alignment: Matching ground-truth lengths
                mel_out = mel_out[:int(dataset.metadata[k][4])]

                np.save(mel_filename, mel_out, allow_pickle=False)
                file.write("|".join(dataset.metadata[k]))
