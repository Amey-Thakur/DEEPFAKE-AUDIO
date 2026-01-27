"""
Deepfake Audio - Speaker Verification Dataset
---------------------------------------------
PyTorch Dataset definition for Speaker Verification.
Iterates over random batches of speakers.

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
from typing import List
from torch.utils.data import Dataset, DataLoader

from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.speaker_batch import SpeakerBatch
from encoder.data_objects.speaker import Speaker
from encoder.params_data import partials_n_frames


class SpeakerVerificationDataset(Dataset):
    """
    A PyTorch Dataset that yields individual speakers from the dataset.
    The order of speakers is randomized using RandomCycler.
    """
    
    def __init__(self, datasets_root: Path):
        self.root = datasets_root
        
        # Identify all valid speaker directories
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        
        if len(speaker_dirs) == 0:
            raise RuntimeError(
                "No speakers found. Ensure you are pointing to the directory "
                "containing all preprocessed speaker directories."
            )
            
        self.speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]
        self.speaker_cycler = RandomCycler(self.speakers)

    def __len__(self):
        # Virtually infinite dataset to allow continuous training
        return int(1e10)
        
    def __getitem__(self, index):
        return next(self.speaker_cycler)
    
    def get_logs(self) -> str:
        """Retrieves concatenated logs from the dataset directory."""
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string
    
    
class SpeakerVerificationDataLoader(DataLoader):
    """
    Custom DataLoader that batches speakers together using a specialized collate function.
    """
    
    def __init__(self, dataset, speakers_per_batch, utterances_per_speaker, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
                 worker_init_fn=None):
        self.utterances_per_speaker = utterances_per_speaker

        super().__init__(
            dataset=dataset, 
            batch_size=speakers_per_batch, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, speakers: List[Speaker]) -> SpeakerBatch:
        """
        Batches a list of speakers into a SpeakerBatch object.
        """
        return SpeakerBatch(speakers, self.utterances_per_speaker, partials_n_frames) 
    