# ==================================================================================================
# DEEPFAKE AUDIO - encoder/data_objects/speaker_batch.py (Neural Batch Collation)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module defines the SpeakerBatch class, which aggregates multiple speakers 
# and their respective partial utterances into a unified tensor structure. It 
# facilitates the high-throughput gradient descent cycles required for the 
# GE2E loss optimization.
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
from typing import List
from encoder.data_objects.speaker import Speaker

class SpeakerBatch:
    """
    Categorical Batch Orchestrator:
    Collates acoustic data for B speakers, each with M utterances, into a 
    consistent [B*M, T, C] matrix for neural ingestion.
    """
    def __init__(self, speakers: List[Speaker], utterances_per_speaker: int, n_frames: int):
        self.speakers = speakers
        
        # Parallel Identity Sampling
        self.partials = {s: s.random_partial(utterances_per_speaker, n_frames) for s in speakers}

        # Sparse-to-Dense Materialization: (n_speakers * n_utterances, n_frames, mel_channels)
        self.data = np.array([frames for s in speakers for _, frames, _ in self.partials[s]])
