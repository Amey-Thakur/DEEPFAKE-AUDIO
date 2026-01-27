"""
Deepfake Audio - Speaker Batch
------------------------------
Represents a batch of data for training, containing embeddings/partials 
for multiple speakers.

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

from typing import List
import numpy as np
from encoder.data_objects.speaker import Speaker


class SpeakerBatch:
    """
    A batch containing data for multiple speakers. Each speaker provides a set of
    partial utterances (spectrogram frames).
    """
    
    def __init__(self, speakers: List[Speaker], utterances_per_speaker: int, n_frames: int):
        self.speakers = speakers
        
        # Collect random partials for each speaker in the batch
        self.partials = {s: s.random_partial(utterances_per_speaker, n_frames) for s in speakers}
        
        # Stack all frames into a single numpy array
        # Shape: (n_speakers * n_utterances, n_frames, mel_n_channels)
        self.data = np.array([frames for s in speakers for _, frames, _ in self.partials[s]])
