"""
Deepfake Audio - Synthesizer Audio Preprocessing Script
-------------------------------------------------------
This script prepares the audio data for the Synthesizer (Tacotron 2) model.
It processes raw audio datasets (e.g., LibriSpeech) into mel spectrograms and
associated audio waveforms, which serve as the ground truth for training.

The Synthesizer requires pairs of (text, mel spectrogram) for training, but to
enable end-to-end voice cloning, we also extract embeddings and vocoder calibration
data during this stage.

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
    The pipeline performs the following tasks:
    1.  **Audio Loading**: Reads raw audio files from the dataset.
    2.  **Silence Removal**: Trims non-speech segments using `webrtcvad`.
    3.  **Mel Spectrogram Computation**: Converts time-domain signals to the mel-frequency domain.
    4.  **Dataset Partitioning**: Organizes data for efficient batch loading during training.
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Standard Library Imports
from utils.argutils import print_args

# Internal Modules
from synthesizer.preprocess import preprocess_librispeech
from synthesizer.hparams import hparams


def main() -> None:
    """
    Main execution routine for synthesizer audio preprocessing.
    
    Orchestrates the data preparation workflow, including argument parsing,
    dependency verification, and invoking the dataset-specific preprocessing logic.
    """
    
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files for Synthesizer training. Generates mel spectrograms "
                    "and aligns them with transcriptions. Also prepares audio for Vocoder fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the root directory containing your LibriSpeech/TTS datasets.")
    
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Output directory for generated training artifacts (mel spectrograms, audio, embeddings). "
        "Defaults to <datasets_root>/SV2TTS/synthesizer/.")
    
    parser.add_argument("-n", "--n_processes", type=int, default=None, help=\
        "Number of parallel worker processes. Defaults to CPU count.")
    
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Skip processing for files that already exist in the output directory. "
        "Useful for resuming interrupted jobs.")
    
    parser.add_argument("--hparams", type=str, default="", help=\
        "Hyperparameter overrides as a comma-separated list of name=value pairs "
        "(e.g., 'sample_rate=22050,ref_level_db=20').")
    
    parser.add_argument("--no_trim", action="store_true", help=\
        "Disable silence trimming. NOT RECOMMENDED. Clean audio is essential for TTS quality.")
        
    args = parser.parse_args()

    # =========================================================================
    # Configuration Setup
    # =========================================================================
    # Set default output directory if not specified
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "synthesizer")

    # Validate inputs
    if not args.datasets_root.exists():
        raise FileNotFoundError(f"Datasets root not found: {args.datasets_root}")
        
    # Create output structure
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # =========================================================================
    # Dependency Verification
    # =========================================================================
    # Check for 'webrtcvad' for silence trimming
    if not args.no_trim:
        try:
            import webrtcvad
        except ImportError:
            raise ModuleNotFoundError(
                "Package 'webrtcvad' not found. This package is required for silence removal. "
                "Please install it via 'pip install webrtcvad'. "
                "If using Windows and installation fails, consider installing C++ build tools "
                "or using --no_trim (not recommended)."
            )
            
    # Clean up namespace (though sub-functions might accept strict kwargs, 
    # we follow the pattern of only passing what's needed or strictly kwargs)
    del args.no_trim

    # =========================================================================
    # Execution
    # =========================================================================
    print_args(args, parser)
    
    # Update hyperparameters with any command-line overrides
    if args.hparams:
        args.hparams = hparams.parse(args.hparams)
    
    # Invoke the LibriSpeech preprocessing routine
    # Note: Although named 'preprocess_librispeech', this structure is often adapted
    # for other aligned TTS datasets.
    preprocess_librispeech(**vars(args))


if __name__ == "__main__":
    main()
