# ==================================================================================================
# DEEPFAKE AUDIO - synthesizer_preprocess_audio.py (Acoustic Feature Alignment)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This script orchestrates the ground-truth audio preprocessing for the Synthesizer. 
# It transforms raw speech utterances into time-aligned Mel-Spectrograms, which are 
# essential for training the Tacotron 2-based synthesis architecture. This process 
# ensures that the synthesizer can learn the relationship between textual tokens 
# and acoustic spectral features.
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

from synthesizer.preprocess import preprocess_dataset
from synthesizer.hparams import hparams
from utils.argutils import print_args
from pathlib import Path
import argparse

if __name__ == "__main__":
    # --- INTERFACE COMMANDS ---
    parser = argparse.ArgumentParser(
        description="Synthesizer Audio Pre-processor: Transforms speech data into training-ready spectrograms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- PATH SPECIFICATIONS ---
    parser.add_argument("datasets_root", type=Path, 
                        help="Root directory containing the LibriSpeech/TTS datasets.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, 
                        help="Destination for the synthesized spectrograms and embeds. Defaults to <datasets_root>/SV2TTS/synthesizer/")
    
    # --- PERFORMANCE & OPTIMIZATION ---
    parser.add_argument("-n", "--n_processes", type=int, default=4, 
                        help="Degree of parallelism for the preprocessing pipeline.")
    parser.add_argument("-s", "--skip_existing", action="store_true", 
                        help="Bypass utterances that have already been materialized on disk.")
    parser.add_argument("--hparams", type=str, default="", 
                        help="Acoustic hyperparameter overrides (comma-separated).")
    parser.add_argument("--no_alignments", action="store_true", 
                        help="Fallback mode for datasets lacking temporal alignment metadata.")
    
    # --- DATASET MANAGEMENT ---
    parser.add_argument("--datasets_name", type=str, default="LibriSpeech", 
                        help="Target dataset identifier to prioritize.")
    parser.add_argument("--subfolders", type=str, default="train-clean-100,train-clean-360", 
                        help="Specific sub-corpora to target within the dataset root.")
    
    args = parser.parse_args()

    # --- ARCHITECTURAL ORCHESTRATION ---
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "synthesizer")

    # Verify workspace integrity.
    assert args.datasets_root.exists(), "Fatal: datasets_root not found. 🤝🏻 Verify pathing."
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # --- EXECUTION ---
    print_args(args, parser)
    print("🤝🏻 Scholarly Partnership: Amey Thakur & Mega Satish")
    print("🚀 Initiating Mel-Spectrogram Extraction Pipeline...")
    
    args.hparams = hparams.parse(args.hparams)
    preprocess_dataset(**vars(args))
