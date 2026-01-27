"""
Deepfake Audio - Speaker Data Object
------------------------------------
Represents a single speaker and their associated audio utterances.
Manages the loading and random sampling of partial utterances for training purposes.

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

from pathlib import Path
from typing import List, Optional, Tuple, Any

from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.utterance import Utterance


class Speaker:
    """
    Encapsulates a speaker's data, including their identity and a collection of
    audio utterances.
    """
    
    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        self.utterances: Optional[List[Utterance]] = None
        self.utterance_cycler: Optional[RandomCycler] = None
        
    def _load_utterances(self):
        """Loads the metadata for all utterances belonging to this speaker."""
        with self.root.joinpath("_sources.txt").open("r") as sources_file:
            sources = [l.strip().split(",") for l in sources_file]
            
        # Map generated npy frame file to original wav file
        sources = {frames_fname: wave_fpath for frames_fname, wave_fpath in sources}
        self.utterances = [Utterance(self.root.joinpath(f), w) for f, w in sources.items()]
        self.utterance_cycler = RandomCycler(self.utterances)
               
    def random_partial(self, count: int, n_frames: int) -> List[Tuple[Utterance, Any, Any]]:
        """
        Samples a batch of <count> unique partial utterances from the disk.
        Ensures consistent coverage of the speaker's data over multiple calls.
        
        Args:
            count: Number of partial utterances to sample.
            n_frames: Number of frames in each partial utterance.
            
        Returns:
            A list of tuples: (Utterance, frames, range_tuple)
        """
        if self.utterances is None:
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)

        # Generate partials for sampled utterances
        # Each 'u' is an Utterance object
        return [(u,) + u.random_partial(n_frames) for u in utterances]
