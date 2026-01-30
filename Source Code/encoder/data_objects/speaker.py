# ==================================================================================================
# DEEPFAKE AUDIO - encoder/data_objects/speaker.py (Categorical Identity Representation)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements the 'Speaker' abstraction, encapulating all linguistic 
# and acoustic metadata for a single individual. It manages the retrieval 
# and segmented sampling of utterances, acting as a gateway to the serialized 
# Mel-Spectrograms used in neural distillation.
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

from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.utterance import Utterance
from pathlib import Path

class Speaker:
    """
    Categorical Data Container:
    Aggregates all speech samples associated with a unique institutional speaker ID.
    """
    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None
        
    def _load_utterances(self):
        """Lazy-loading of utterance metadata from the serialized index (_sources.txt)."""
        with self.root.joinpath("_sources.txt").open("r") as sources_file:
            sources = [l.split(",") for l in sources_file]
        
        # Identity Mapping: frames_fname -> original_wave_fpath
        sources = {frames_fname: wave_fpath for frames_fname, wave_fpath in sources}
        self.utterances = [Utterance(self.root.joinpath(f), w) for f, w in sources.items()]
        self.utterance_cycler = RandomCycler(self.utterances)
               
    def random_partial(self, count, n_frames):
        """
        Samples a batch of <count> unique partial utterances.
        Ensures diverse temporal coverage within the speaker's available vocal range.
        """
        if self.utterances is None:
            self._load_utterances()

        # Stochastic selection of utterances
        utterances = self.utterance_cycler.sample(count)

        # Spatio-temporal cropping: (utterance, frames, crop_range)
        a = [(u,) + u.random_partial(n_frames) for u in utterances]

        return a
