# ==================================================================================================
# DEEPFAKE AUDIO - encoder/data_objects/speaker_verification_dataset.py (PyTorch Data Layer)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements the PyTorch Dataset and DataLoader abstractions tailored 
# for Speaker Verification. It manages the discovery of speaker directories, 
# categorical sampling via RandomCycler, and high-performance batch collation.
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
from encoder.data_objects.speaker_batch import SpeakerBatch
from encoder.data_objects.speaker import Speaker
from encoder.params_data import partials_n_frames
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class SpeakerVerificationDataset(Dataset):
    """
    Neural Corpus Interface:
    Scans a root directory for processed speaker identities and provides 
    an infinite stochastic stream of categorical data.
    """
    def __init__(self, datasets_root: Path):
        self.root = datasets_root
        
        # Identity Discovery
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        if len(speaker_dirs) == 0:
            raise Exception("⚠️ Technical Alert: No speakers detected in %s." % self.root)
        
        self.speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]
        self.speaker_cycler = RandomCycler(self.speakers)

    def __len__(self):
        """Returns a high constant to simulate an infinite stream for the DataLoader."""
        return int(1e10)
        
    def __getitem__(self, index):
        """Retrieves the next stochastic categorical identity."""
        return next(self.speaker_cycler)
    
    def get_logs(self):
        """Aggregates all preprocessing logs into a single analytical string."""
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string
    
class SpeakerVerificationDataLoader(DataLoader):
    """
    High-Throughput Orchestrator:
    Custom DataLoader designed to yield SpeakerBatch objects containing 
    diverse identities and utterances.
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
            collate_fn=self.collate, # Custom collation for GE2E loss
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, speakers):
        """Constructs a SpeakerBatch from a set of sampled identities."""
        return SpeakerBatch(speakers, self.utterances_per_speaker, partials_n_frames) 
    