# ==================================================================================================
# DEEPFAKE AUDIO - encoder_preprocess.py (Acoustic Feature Extraction)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This script serves as the primary data ingestion layer for the Speaker Encoder. It 
# processes raw waveforms from massive speech datasets (LibriSpeech, VoxCeleb) and extracts 
# high-dimensional Mel-Spectrogram features. These features are essentially "voice prints" 
# that allow the encoder to learn speaker-independent embedding representations.
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

from encoder.preprocess import preprocess_librispeech, preprocess_voxceleb1, preprocess_voxceleb2
from utils.argutils import print_args
from pathlib import Path
import argparse

if __name__ == "__main__":
    # --- INTERFACE CONFIGURATION ---
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        """Custom formatter to preserve formatting in the help description."""
        pass

    parser = argparse.ArgumentParser(
        description="Acoustic Pre-processor: Normalizes raw datasets into mel-spectrogram arrays.\n"
                    "Required datasets: LibriSpeech, VoxCeleb1, or VoxCeleb2.",
        formatter_class=MyFormatter
    )
    
    # --- PATH DEFINITIONS ---
    parser.add_argument("datasets_root", type=Path, 
                        help="Root directory where raw datasets are extracted.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, 
                        help="Destination for processed neural features. Defaults to <datasets_root>/SV2TTS/encoder/")
    
    # --- PROCESSING PARAMETERS ---
    parser.add_argument("-d", "--datasets", type=str,
                        default="librispeech_other,voxceleb1,voxceleb2", 
                        help="Comma-separated identifiers of datasets to include in the pipeline.")
    parser.add_argument("-s", "--skip_existing", action="store_true", 
                        help="Optimize by skipping files that have already been materialized on disk.")
    parser.add_argument("--no_trim", action="store_true", 
                        help="Inhibit silent period removal (Voice Activity Detection). Not recommended for high-fidelity training.")
    
    args = parser.parse_args()

    # --- HARDWARE & VAD VALIDATION ---
    # We verify the presence of 'webrtcvad' as it is critical for ensuring non-silent training samples.
    if not hasattr(args, "no_trim") or not args.no_trim:
        try:
            import webrtcvad
        except:
            print("⚠️ Scholarly Warning: 'webrtcvad' not detected. This is required for speech silence removal.")
            raise ModuleNotFoundError("Please install 'webrtcvad' or use --no_trim for a degraded run.")

    # --- ARCHITECTURAL ORCHESTRATION ---
    args.datasets = args.datasets.split(",")
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "encoder")
    
    assert args.datasets_root.exists(), "Fatal: datasets_root not found. 🤝🏻 Ensure pathing."
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # --- EXECUTION ---
    print_args(args, parser)
    print("🤝🏻 Scholarly Partnership: Amey Thakur & Mega Satish")
    
    preprocess_func = {
        "librispeech_other": preprocess_librispeech,
        "voxceleb1": preprocess_voxceleb1,
        "voxceleb2": preprocess_voxceleb2,
    }
    
    args_dict = vars(args)
    datasets_to_process = args_dict.pop("datasets")
    
    for dataset in datasets_to_process:
        print("\n🚀 Initiating Neural Feature Extraction for: %s" % dataset)
        preprocess_func[dataset](**args_dict)
