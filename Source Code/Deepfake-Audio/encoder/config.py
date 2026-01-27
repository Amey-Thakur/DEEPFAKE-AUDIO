"""
Deepfake Audio - Encoder Dataset Configuration
----------------------------------------------
This module defines the directory structures and file paths for the datasets
supported by the Speaker Encoder training pipeline.

Supported Datasets:
    - LibriSpeech: A large corpus of read English speech.
    - LibriTTS: A text-to-speech dataset derived from LibriSpeech.
    - VoxCeleb: A large-scale speaker identification dataset.

Authors:
    - Amey Thakur (https://github.com/Amey-Thakur)
    - Mega Satish (https://github.com/msatmod)

Repository:
    - https://github.com/Amey-Thakur/DEEPFAKE-AUDIO

Release Date:
    - February 06, 2021

License:
    - MIT License
"""

from typing import Dict, List

# -----------------------------------------------------------------------------
# LibriSpeech Dataset Configuration
# -----------------------------------------------------------------------------
librispeech_datasets: Dict[str, Dict[str, List[str]]] = {
    "train": {
        "clean": ["LibriSpeech/train-clean-100", "LibriSpeech/train-clean-360"],
        "other": ["LibriSpeech/train-other-500"]
    },
    "test": {
        "clean": ["LibriSpeech/test-clean"],
        "other": ["LibriSpeech/test-other"]
    },
    "dev": {
        "clean": ["LibriSpeech/dev-clean"],
        "other": ["LibriSpeech/dev-other"]
    },
}

# -----------------------------------------------------------------------------
# LibriTTS Dataset Configuration
# -----------------------------------------------------------------------------
libritts_datasets: Dict[str, Dict[str, List[str]]] = {
    "train": {
        "clean": ["LibriTTS/train-clean-100", "LibriTTS/train-clean-360"],
        "other": ["LibriTTS/train-other-500"]
    },
    "test": {
        "clean": ["LibriTTS/test-clean"],
        "other": ["LibriTTS/test-other"]
    },
    "dev": {
        "clean": ["LibriTTS/dev-clean"],
        "other": ["LibriTTS/dev-other"]
    },
}

# -----------------------------------------------------------------------------
# VoxCeleb Dataset Configuration
# -----------------------------------------------------------------------------
voxceleb_datasets: Dict[str, Dict[str, List[str]]] = {
    "voxceleb1": {
        "train": ["VoxCeleb1/wav"],
        "test": ["VoxCeleb1/test_wav"]
    },
    "voxceleb2": {
        "train": ["VoxCeleb2/dev/aac"],
        "test": ["VoxCeleb2/test_wav"]
    }
}

# -----------------------------------------------------------------------------
# Miscellaneous Configuration
# -----------------------------------------------------------------------------
other_datasets: List[str] = [
    "LJSpeech-1.1",
    "VCTK-Corpus/wav48",
]

anglophone_nationalities: List[str] = ["australia", "canada", "ireland", "uk", "usa"]
