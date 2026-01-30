# ==================================================================================================
# DEEPFAKE AUDIO - vocoder_preprocess.py (Spectrogram Realignment for Waveform Synthesis)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This script prepares the Ground-Truth Aligned (GTA) mel-spectrograms required 1or 
# training the Waveform Vocoder (WaveRNN). By passing training audio through the 
# synthesizer and capturing the resulting spectrograms, we ensure that the vocoder 
# learns to reconstruct audio from the specific spectral artifacts produced by 
# the synthesis engine, rather than just perfect ground truth data.
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
import os
from pathlib import Path

# --- SYNTHETIC PIPELINE MODULES ---
from synthesizer.hparams import hparams
from synthesizer.synthesize import run_synthesis
from utils.argutils import print_args

if __name__ == "__main__":
    # --- INTERFACE CONFIGURATION ---
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        """Custom formatter to preserve scholarly descriptions."""
        pass

    parser = argparse.ArgumentParser(
        description="GTA Pre-processor: Generates synthesizer-aligned spectrograms for vocoder training.",
        formatter_class=MyFormatter
    )
    
    # --- PATH SPECIFICATIONS ---
    parser.add_argument("datasets_root", type=Path, 
                        help="Root directory of the SV2TTS data structure.")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="Path to the trained synthesizer weights used for GTA generation.")
    parser.add_argument("-i", "--in_dir", type=Path, default=argparse.SUPPRESS, 
                        help="Input directory (synthesizer-level data). Defaults to <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, 
                        help="Output directory for GTA features. Defaults to <datasets_root>/SV2TTS/vocoder/.")
    
    # --- COMPUTE PARAMETERS ---
    parser.add_argument("--hparams", default="", 
                        help="Acoustic hyperparameter overrides.")
    parser.add_argument("--cpu", action="store_true", 
                        help="Enforce CPU-only processing for GTA synthesis.")
    
    args = parser.parse_args()

    # --- ARCHITECTURAL ORCHESTRATION ---
    print_args(args, parser)
    print("🤝🏻 Scholarly Partnership: Amey Thakur & Mega Satish")
    print("🚀 Extracting Ground-Truth Aligned mel-spectrograms...")
    
    modified_hp = hparams.parse(args.hparams)

    # Resolution of default paths within the SV2TTS ecosystem.
    if not hasattr(args, "in_dir"):
        args.in_dir = args.datasets_root / "SV2TTS" / "synthesizer"
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root / "SV2TTS" / "vocoder"

    # Hardware masking for CPU-constrained environments.
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Execute the GTA synthesis engine.
    run_synthesis(args.in_dir, args.out_dir, args.syn_model_fpath, modified_hp)
