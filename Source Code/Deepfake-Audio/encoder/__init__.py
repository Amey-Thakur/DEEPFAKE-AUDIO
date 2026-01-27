"""
Deepfake Audio - Encoder Package
--------------------------------
This package contains the implementation of the Speaker Encoder, a neural network
trained to generate fixed-dimensional embeddings (d-vectors) from speech utterances.
These embeddings capture speaker-specific characteristics and are used to condition
the Synthesizer.

components:
    - audio.py: Signal processing utilities (VAD, Spectrograms).
    - model.py: The LSTM-based Speaker Encoder architecture.
    - train.py: Training loop and optimization logic.
    - inference.py: Inference interface for generating embeddings.
    - params_*.py: Hyperparameter configurations.

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
