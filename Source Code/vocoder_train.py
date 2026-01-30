# ==================================================================================================
# DEEPFAKE AUDIO - vocoder_train.py (Waveform Reconstruction Neural Training)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This script manages the training of the WaveRNN-based Vocoder. It optimizes a 
# recurrent neural network to reconstruct high-fidelity time-domain audio from 
# mel-spectrogram sequences. For optimal results, it uses the GTA (Ground-Truth 
# Aligned) spectrograms generated during the preprocessing phase, allowing the 
# vocoder to compensate for common synthesizer artifacts.
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

import argparse
from pathlib import Path

# --- CORE VOCATION ENGINE ---
from utils.argutils import print_args
from vocoder.train import train

if __name__ == "__main__":
    # --- TRAINING PARAMETERS ---
    parser = argparse.ArgumentParser(
        description="Vocoder Training Hub: Reconstructing waveforms from synthetic spectrograms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- SESSION DEFINITION ---
    parser.add_argument("run_id", type=str, 
                        help="Identifier for this experimental run. Weights and metrics will be archived under this ID.")
    parser.add_argument("datasets_root", type=Path, 
                        help="Root path to the SV2TTS training directory.")
    
    # --- PATH OVERRIDES ---
    parser.add_argument("--syn_dir", type=Path, default=argparse.SUPPRESS, 
                        help="Custom path to synthesizer-level data. Defaults to <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("--voc_dir", type=Path, default=argparse.SUPPRESS, 
                        help="Custom path to vocoder-level GTA data. Defaults to <datasets_root>/SV2TTS/vocoder/.")
    parser.add_argument("-m", "--models_dir", type=Path, default="saved_models", 
                        help="Destination for saved weights, rolling backups, and audio samples.")
    
    # --- TRAINING LOGIC ---
    parser.add_argument("-g", "--ground_truth", action="store_true", 
                        help="Bypass GTA spectrograms and train directly on original ground truth data.")
    parser.add_argument("-s", "--save_every", type=int, default=1000, 
                        help="Iteration interval for weight persistence on disk.")
    parser.add_argument("-b", "--backup_every", type=int, default=25000, 
                        help="Frequency of immutable state backups.")
    parser.add_argument("-f", "--force_restart", action="store_true", 
                        help="Initialize weights from scratch, ignoring existing checkpoints.")
    
    args = parser.parse_args()

    # --- ARCHITECTURAL RESOLUTION ---
    if not hasattr(args, "syn_dir"):
        args.syn_dir = args.datasets_root / "SV2TTS" / "synthesizer"
    if not hasattr(args, "voc_dir"):
        args.voc_dir = args.datasets_root / "SV2TTS" / "vocoder"
    
    # Finalize workspace preparations.
    del args.datasets_root
    args.models_dir.mkdir(exist_ok=True)

    # --- EXECUTION ---
    print_args(args, parser)
    print("🤝🏻 Scholarly Partnership: Amey Thakur & Mega Satish")
    print("🚀 Initiating Vocoder Training Pipeline - Optimizing wave reconstruction...")
    
    # Delegate to the WaveRNN training module.
    train(**vars(args))
