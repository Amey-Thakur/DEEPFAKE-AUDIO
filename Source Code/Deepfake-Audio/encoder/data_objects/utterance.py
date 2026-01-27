"""
Deepfake Audio - Utterance Data Object
--------------------------------------
Represents a single audio utterance. 
Handles lazy loading of Mel Spectrogram frames and random cropping.

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

from typing import Tuple
from pathlib import Path
import numpy as np


class Utterance:
    """
    Holds reference to an utterance's spectrogram frames and its original waveform.
    Allows sampling fixed-length partials for training.
    """
    
    def __init__(self, frames_fpath: Path, wave_fpath: Path):
        self.frames_fpath = frames_fpath
        self.wave_fpath = wave_fpath
        
    def get_frames(self) -> np.ndarray:
        """Loads and returns the mel spectrogram frames from disk."""
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crops the frames into a partial utterance of specified length.
        
        Args:
            n_frames: Duration of the partial utterance in frames.
            
        Returns:
            Tuple containing:
            1. The cropped frames (numpy array).
            2. Tuple `(start, end)` indices of the crop.
        """
        frames = self.get_frames()
        
        if frames.shape[0] == n_frames:
            start = 0
        else:
            # Random starting point ensuring full length fits
            start = np.random.randint(0, frames.shape[0] - n_frames)
            
        end = start + n_frames
        return frames[start:end], (start, end)