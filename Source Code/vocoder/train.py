# ==================================================================================================
# DEEPFAKE AUDIO - vocoder/train.py (Model Training Orchestrator)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This script manages the training lifecycle of the WaveRNN vocoder. It 
# handles dataset loading, gradient optimization via Adam, periodic 
# checkpointing, and real-time generation of validation samples to monitor 
# convergence.
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

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import vocoder.hparams as hp
from vocoder.display import stream, simple_table
from vocoder.distribution import discretized_mix_logistic_loss
from vocoder.gen_wavernn import gen_testset
from vocoder.models.fatchord_version import WaveRNN
from vocoder.vocoder_dataset import VocoderDataset, collate_vocoder


def train(run_id: str, syn_dir: Path, voc_dir: Path, models_dir: Path, ground_truth: bool, save_every: int,
          backup_every: int, force_restart: bool):
    """
    Main Training Loop:
    Executes the neural network training protocol for high-fidelity audio synthesis.
    """
    # Integrity Check: Ensure hop length matches the upsampling spatial pyramid
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    # Neural Architecture Initialization
    print("Architectural Supervision: Initializing the WaveRNN model...")
    model = WaveRNN(
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
        model = model.cuda()

    # Optimization Setup: Continuous learning via Adam optimizer
    optimizer = optim.Adam(model.parameters())
    for p in optimizer.param_groups:
        p["lr"] = hp.voc_lr
    # Loss objective selection based on synthesis mode
    loss_func = F.cross_entropy if model.mode == "RAW" else discretized_mix_logistic_loss

    # Persistence: Load or initialize weights
    model_dir = models_dir / run_id
    model_dir.mkdir(exist_ok=True)
    weights_fpath = model_dir / "vocoder.pt"
    if force_restart or not weights_fpath.exists():
        print("\nClean Start: Training WaveRNN from scratch\n")
        model.save(weights_fpath, optimizer)
    else:
        print("\nRestoration: Loading weights from %s" % weights_fpath)
        model.load(weights_fpath, optimizer)
        print("Status: WaveRNN weights loaded from global step %d" % model.step)

    # Data Ingestion: Prepare training metadata and file pointers
    metadata_fpath = syn_dir.joinpath("train.txt") if ground_truth else \
        voc_dir.joinpath("synthesized.txt")
    mel_dir = syn_dir.joinpath("mels") if ground_truth else voc_dir.joinpath("mels_gta")
    wav_dir = syn_dir.joinpath("audio")
    dataset = VocoderDataset(metadata_fpath, mel_dir, wav_dir)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # UI Feedback: Display training hyperparameters
    simple_table([('Batch size', hp.voc_batch_size),
                  ('LR', hp.voc_lr),
                  ('Sequence Len', hp.voc_seq_len)])

    # Epoch Supervision: Iterate through the dataset
    for epoch in range(1, 350):
        data_loader = DataLoader(dataset, hp.voc_batch_size, shuffle=True, num_workers=2, collate_fn=collate_vocoder)
        start = time.time()
        running_loss = 0.

        for i, (x, y, m) in enumerate(data_loader, 1):
            if torch.cuda.is_available():
                x, m, y = x.cuda(), m.cuda(), y.cuda()

            # Forward pass: Generate predictions
            y_hat = model(x, m)
            if model.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            elif model.mode == 'MOL':
                y = y.float()
            y = y.unsqueeze(-1)

            # Backward pass: Weight adjustment
            loss = loss_func(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics management
            running_loss += loss.item()
            speed = i / (time.time() - start)
            avg_loss = running_loss / i

            step = model.get_step()
            k = step // 1000

            # Periodic persistence
            if backup_every != 0 and step % backup_every == 0 :
                model.checkpoint(model_dir, optimizer)

            if save_every != 0 and step % save_every == 0 :
                model.save(weights_fpath, optimizer)

            msg = f"| Epoch: {epoch} ({i}/{len(data_loader)}) | " \
                f"Loss: {avg_loss:.4f} | {speed:.1f} " \
                f"steps/s | Step: {k}k | "
            stream(msg)


        # Validation: Generate qualitative results after each epoch
        gen_testset(model, test_loader, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
                    hp.voc_target, hp.voc_overlap, model_dir)
        print("")
