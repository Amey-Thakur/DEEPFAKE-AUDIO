# ==================================================================================================
# DEEPFAKE AUDIO - encoder/data_objects/utterance.py (Vocal Unit Representation)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module defines the 'Utterance' class, representing a single spoken phrase 
# or acoustic segment. It provides mechanisms for loading preprocessed Mel-scale 
# filterbanks from the disk and handles stochastic temporal cropping (random 
# partials) to increase data variety during training.
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

import numpy as np

class Utterance:
    """
    Acoustic Data Container:
    Manages the lifecycle of a single vocal sample, from disk retrieval to 
    stochastic temporal segmentation.
    """
    def __init__(self, frames_fpath, wave_fpath):
        self.frames_fpath = frames_fpath
        self.wave_fpath = wave_fpath
        
    def get_frames(self):
        """Deserializes the Mel-Spectrogram matrix from the filesystem."""
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames):
        """
        Spatio-Temporal Cropping:
        Cuts a random segment of 'n_frames' from the full utterance.
        This technique acts as a form of temporal data augmentation.
        """
        frames = self.get_frames()
        if frames.shape[0] == n_frames:
            start = 0
        else:
            # Stochastic offset selection
            start = np.random.randint(0, frames.shape[0] - n_frames)
        
        end = start + n_frames
        return frames[start:end], (start, end)