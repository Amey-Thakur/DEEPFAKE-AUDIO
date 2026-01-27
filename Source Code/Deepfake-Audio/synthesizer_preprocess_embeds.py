"""
Deepfake Audio - Synthesizer Embedding Generation Script
--------------------------------------------------------
This script generates speaker embeddings for the Synthesizer using a pre-trained Speaker Encoder.
These embeddings condition the Text-To-Speech model, enabling it to modulate its output
to match the target speaker's voice characteristics.

This is the second stage of the Synthesizer data preparation pipeline, executed
after `synthesizer_preprocess_audio.py`.

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
    The script performs the following:
    1.  **Audio Ingestion**: Loads the audio clips processed in the previous stage.
    2.  **Encoder Inference**: Runs the Speaker Encoder on each clip to produce a fixed-dimensional embedding.
    3.  **Data Alignment**: Associates each embedding with its corresponding spectrogram and text text for training.
"""

import argparse
from pathlib import Path
from typing import Optional

# Standard Library Imports
from utils.argutils import print_args

# Internal Modules
from synthesizer.preprocess import create_embeddings


def main() -> None:
    """
    Main execution routine for synthesizer embedding generation.
    
    Loads a pre-trained encoder and runs inference across the synthesis dataset
    to create the speaker conditioning vectors.
    """
    
    parser = argparse.ArgumentParser(
        description="Creates speaker embeddings for the Synthesizer using a trained Encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("synthesizer_root", type=Path, help=\
        "Path to the synthesizer training data directory (output of synthesizer_preprocess_audio.py). "
        "Typically: <datasets_root>/SV2TTS/synthesizer/.")
    
    parser.add_argument("-e", "--encoder_model_fpath", type=Path, 
                        default="encoder/saved_models/pretrained.pt", help=\
        "Path to the trained Speaker Encoder model checkpoint (e.g., 'saved_models/pretrained.pt').")
    
    parser.add_argument("-n", "--n_processes", type=int, default=4, help= \
        "Number of parallel worker processes. Each process loads an instance of the Encoder. "
        "Reduce this value if you encounter CUDA out-of-memory errors.")
        
    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------
    print_args(args, parser)
    
    # Validate input directory
    if not args.synthesizer_root.exists():
         raise FileNotFoundError(f"Synthesizer root not found: {args.synthesizer_root}")

    # Generate embeddings
    create_embeddings(**vars(args))


if __name__ == "__main__":
    main()
