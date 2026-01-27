"""
Deepfake Audio - Encoder Preprocessing Script
---------------------------------------------
This script handles the preprocessing of raw audio datasets for training the Speaker Encoder.
The Speaker Encoder requires input audio in the form of mel spectrograms, which are generated
from raw waveforms after silence removal and normalization.

Supported Datasets:
    1.  **LibriSpeech**: Large-scale corpus of read English speech.
    2.  **VoxCeleb1**: Speaker identification dataset from celebrity interviews.
    3.  **VoxCeleb2**: Extended speaker identification dataset.

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
    The preprocessing pipeline performs the following steps:
    1.  **Silence Trimming**: Removes non-speech intervals to optimize training efficiency (requires `webrtcvad`).
    2.  **Mel Spectrogram Generation**: Converts time-domain waveforms into frequency-domain mel spectrograms.
    3.  **Storage**: Saves the processed data to the disk for rapid access during training.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Any, Callable

# Standard Library Imports
from utils.argutils import print_args

# Internal Preprocessing Routines
from encoder.preprocess import preprocess_librispeech, preprocess_voxceleb1, preprocess_voxceleb2


def main() -> None:
    """
    Main execution routine for encoder preprocessing.
    
    Parses CLI arguments, verifies dependencies (webrtcvad), and delegates
    processing to dataset-specific routines.
    """
    
    # Custom formatter class to support multiple inheritance for formatting help text
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass
    
    parser = argparse.ArgumentParser(
        description=(
            "Preprocesses audio files from datasets, encodes them as mel spectrograms, and "
            "writes them to the disk. This creates the training data for the Speaker Encoder.\n\n"
            "Requirements:\n"
            "    Your dataset directory structure should look like this:\n"
            "    [datasets_root]\n"
            "        ├── LibriSpeech\n"
            "        │   └── train-other-500\n"
            "        ├── VoxCeleb1\n"
            "        │   ├── wav\n"
            "        │   └── vox1_meta.csv\n"
            "        └── VoxCeleb2\n"
            "            └── dev\n"
        ),
        formatter_class=MyFormatter
    )
    
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the root directory containing your LibriSpeech and/or VoxCeleb datasets.")
    
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Output directory for the mel spectrograms. Defaults to <datasets_root>/SV2TTS/encoder/.")
    
    parser.add_argument("-d", "--datasets", type=str, 
                        default="librispeech_other,voxceleb1,voxceleb2", help=\
        "Comma-separated list of datasets to preprocess (e.g., 'librispeech_other,voxceleb1'). "
        "Only the training sets of these datasets will be processed.")
    
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Skip processing for files that already exist in the output directory. "
        "Useful for resuming interrupted jobs.")
    
    parser.add_argument("--no_trim", action="store_true", help=\
        "Disable silence trimming. NOT RECOMMENDED. Silence removal is crucial for training quality.")
        
    args = parser.parse_args()

    # =========================================================================
    # Dependency Verification
    # =========================================================================
    # Check for 'webrtcvad' unless trimming is explicitly disabled.
    # Webrtcvad is a high-performance Voice Activity Detector (VAD) from WebRTC.
    if not args.no_trim:
        try:
            import webrtcvad
        except ImportError:
            raise ModuleNotFoundError(
                "Package 'webrtcvad' not found. This package is required for silence removal "
                "(voice activity detection). Please install it via 'pip install webrtcvad'. "
                "If installation fails (common on Windows), ensure you have the C++ build tools installed, "
                "or use --no_trim to disable VAD (not recommended)."
            )
    
    # Remove the no_trim flag from args to prevent passing it to functions that don't expect it
    # (though 'no_trim' isn't directly passed to preprocess_*, checking is good practice)
    # Actually, the logic below passes **vars(args) to functions, so we should keep what they expect.
    # The original code deleted it, but let's check if the sub-functions use it.
    # Assuming standard implementation, we retain the original logic of deleting it if not needed,
    # but here it seems safer to keep strictly what was there or what is needed.
    del args.no_trim

    # =========================================================================
    # Configuration & Execution
    # =========================================================================
    # Parse dataset list
    datasets_to_process: List[str] = args.datasets.split(",")
    
    # Set default output directory if not specified
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "encoder")
    
    # Validate input path
    if not args.datasets_root.exists():
        raise FileNotFoundError(f"Datasets root not found: {args.datasets_root}")
        
    # Create output directory hierarchy
    args.out_dir.mkdir(exist_ok=True, parents=True)

    print_args(args, parser)
    
    # Map dataset names to their respective preprocessing functions
    preprocess_manifest: Dict[str, Callable] = {
        "librispeech_other": preprocess_librispeech,
        "voxceleb1": preprocess_voxceleb1,
        "voxceleb2": preprocess_voxceleb2,
    }
    
    # Convert args to dictionary for kwargs unpacking
    args_dict = vars(args)
    # Remove 'datasets' from the dict as it's not a valid argument for the preprocess functions
    del args_dict["datasets"]
    
    for dataset in datasets_to_process:
        if dataset not in preprocess_manifest:
            print(f"[WARNING] Unknown dataset '{dataset}'. Skipping.")
            continue
            
        print(f"Preprocessing {dataset}...")
        preprocess_manifest[dataset](**args_dict)


if __name__ == "__main__":
    main()
