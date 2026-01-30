# ==================================================================================================
# DEEPFAKE AUDIO - synthesizer_train.py (Acoustic Generator Training)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This script manages the training of the Tacotron 2-based Synthesizer. It optimizes
# a sequence-to-sequence neural network with attention to map text tokens into
# realistic mel-spectrograms. By conditioning the process on speaker embeddings,
# the model learns to synthesize speech in the voice of any target speaker.
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

from pathlib import Path
from synthesizer.hparams import hparams
from synthesizer.train import train
from utils.argutils import print_args
import argparse

if __name__ == "__main__":
    # --- TRAINING PARAMETERS ---
    parser = argparse.ArgumentParser(
        description="Synthesizer Training Hub: Optimizing the Tacotron 2 sequence-to-sequence engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- SESSION DEFINITION ---
    parser.add_argument("run_id", type=str, 
                        help="Identifier for this training experiment. Logs and weights will be archived here.")
    parser.add_argument("syn_dir", type=Path, 
                        help="Path to the training data directory (spectrograms, wavs, and embeds).")
    
    # --- STORAGE & BACKUP ---
    parser.add_argument("-m", "--models_dir", type=Path, default="saved_models", 
                        help="Destination for serialized model weights and event logs.")
    parser.add_argument("-s", "--save_every", type=int, default=1000, 
                        help="Iteration interval for materializing weights onto long-term storage.")
    parser.add_argument("-b", "--backup_every", type=int, default=25000, 
                        help="Frequency of immutable state backups to prevent data loss.")
    parser.add_argument("-f", "--force_restart", action="store_true", 
                        help="Inhibit weight loading and re-initialize from scratch.")
    
    # --- HYPERPARAMETER TUNING ---
    parser.add_argument("--hparams", default="", 
                        help="Sequence architecture overrides (comma-separated).")
    
    args = parser.parse_args()

    # --- EXECUTION ---
    print_args(args, parser)
    print("🤝🏻 Scholarly Partnership: Amey Thakur & Mega Satish")
    print("🚀 Initiating Synthetic Engine Training - Optimizing attention alignments...")
    
    args.hparams = hparams.parse(args.hparams)
    
    # Delegate to the sequence-to-sequence training engine.
    train(**vars(args))
