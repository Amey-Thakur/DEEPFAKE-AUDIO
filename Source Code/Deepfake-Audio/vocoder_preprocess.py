"""
Deepfake Audio - Vocoder Preprocessing Script
---------------------------------------------
This script transforms the synthesizer's output (mel spectrograms) into
Ground Truth Aligned (GTA) spectrograms suitable for training the Vocoder.

The Vocoder (WaveRNN) learns to invert mel spectrograms back into time-domain waveforms.
Training on GTA spectrograms (which contain synthesized noise/artifacts) rather than purely
ground-truth spectrograms helps the Vocoder refine its generation quality and robustly
handle the imperfections of the Synthesizer output.

Authors:
    - Amey Thakur (https://github.com/Amey-Thakur)
    - Mega Satish (https://github.com/msatmod)

Repository:
    - https://github.com/Amey-Thakur/DEEPFAKE-AUDIO

Release Date:
    - February 06, 2021

License:
    - MIT License

Description:
    The preprocessing workflow:
    1.  **Alignment**: Uses the Synthesizer model to generate spectrograms from ground-truth text.
    2.  **Teacher Forcing**: Constrains the generation using the ground-truth audio embeddings.
    3.  **GTA Generation**: Produces spectrograms that mimic the distribution of real synthesis outputs.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

# Standard Library Imports
from utils.argutils import print_args

# Internal Modules
from synthesizer.synthesize import run_synthesis
from synthesizer.hparams import hparams


def main() -> None:
    """
    Main execution routine for vocoder preprocessing (GTA generation).
    
    Loads a pretrained Synthesizer and generates aligned spectrograms for all
    utterances in the training dataset.
    """
    
    # Custom formatter for better CLI help description
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass
    
    parser = argparse.ArgumentParser(
        description="Generates Ground Truth Aligned (GTA) spectrograms for Vocoder training.",
        formatter_class=MyFormatter
    )
    
    # -------------------------------------------------------------------------
    # Arguments
    # -------------------------------------------------------------------------
    parser.add_argument("datasets_root", type=str, help=\
        "Path to the root directory containing the SV2TTS structure. "
        "Used to infer input/output paths if not explicitly provided.")
    
    parser.add_argument("--model_dir", type=str, 
                        default="synthesizer/saved_models/logs-pretrained/", help=\
        "Path to the directory containing the pretrained Synthesizer checkpoint.")
    
    parser.add_argument("-i", "--in_dir", type=str, default=argparse.SUPPRESS, help= \
        "Direct path to the input synthesizer directory (containing mels, wavs, embeds). "
        "Defaults to <datasets_root>/SV2TTS/synthesizer/.")
        
    parser.add_argument("-o", "--out_dir", type=str, default=argparse.SUPPRESS, help= \
        "Direct path to the output vocoder directory for GTA mels. "
        "Defaults to <datasets_root>/SV2TTS/vocoder/.")
    
    parser.add_argument("--hparams", default="",
                        help="Hyperparameter overrides as a comma-separated list of name=value pairs.")
                        
    parser.add_argument("--no_trim", action="store_true", help=\
        "Disable silence trimming (not recommended). "
        "Must match the setting used during synthesizer preprocessing.")
        
    args = parser.parse_args()
    print_args(args, parser)
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    modified_hp = hparams.parse(args.hparams)
    
    # Infer directories from datasets_root if not provided
    if not hasattr(args, "in_dir"):
        args.in_dir = os.path.join(args.datasets_root, "SV2TTS", "synthesizer")
        
    if not hasattr(args, "out_dir"):
        args.out_dir = os.path.join(args.datasets_root, "SV2TTS", "vocoder")
    
    # -------------------------------------------------------------------------
    # Dependency Check
    # -------------------------------------------------------------------------
    # Verify webrtcvad availability needed for audio processing utils
    if not args.no_trim:
        try:
            import webrtcvad
        except ImportError:
            raise ModuleNotFoundError(
                "Package 'webrtcvad' not found. This package is required for silence removal. "
                "Please install it via 'pip install webrtcvad'."
            )
            
    del args.no_trim

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------
    # Run GTA synthesis
    run_synthesis(args.in_dir, args.out_dir, args.model_dir, modified_hp)


if __name__ == "__main__":
    main()

