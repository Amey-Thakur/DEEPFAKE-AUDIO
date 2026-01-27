"""
Deepfake Audio - Encoder Data Preprocessing
-------------------------------------------
This module handles the preprocessing of raw audio datasets (LibriSpeech, VoxCeleb, etc.)
into a format suitable for training the Speaker Encoder.

It performs the following:
1.  **Normalization**: Standardizes volume and sampling rate.
2.  **Silence Removal**: Trims non-speech segments using VAD.
3.  **Mel Spectrogram Generation**: Computes features for training.
4.  **Data Organization**: Structures the output for efficient loading.

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

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

import numpy as np
from multiprocess.pool import ThreadPool
from tqdm import tqdm

# Internal Modules
from encoder import audio
from encoder.config import librispeech_datasets, anglophone_nationalites
from encoder.params_data import sampling_rate, partials_n_frames


class DatasetLog:
    """
    Manages logging of dataset creation metadata, including parameters,
    statistics on audio duration, and references to source files.
    """
    
    def __init__(self, root: Path, name: str):
        self.text_file = open(root.joinpath(f"Log_{name.replace('/', '_')}.txt"), "w")
        self.sample_data: Dict[str, List[float]] = {}
        
        start_time = datetime.now().strftime("%A %d %B %Y at %H:%M")
        self.write_line(f"Creating dataset {name} on {start_time}")
        self.write_line("-----")
        self._log_params()
        
    def _log_params(self):
        """Logs the hyperparameter configuration used during preprocessing."""
        from encoder import params_data
        self.write_line("Parameter values:")
        for param_name in (p for p in dir(params_data) if not p.startswith("__")):
            value = getattr(params_data, param_name)
            self.write_line(f"\t{param_name}: {value}")
        self.write_line("-----")
    
    def write_line(self, line: str):
        """Writes a single line to the log file."""
        self.text_file.write(f"{line}\n")
        
    def add_sample(self, **kwargs):
        """Records statistics for a processed sample."""
        for param_name, value in kwargs.items():
            if param_name not in self.sample_data:
                self.sample_data[param_name] = []
            self.sample_data[param_name].append(value)
            
    def finalize(self):
        """Writes summary statistics and closes the log file."""
        self.write_line("Statistics:")
        for param_name, values in self.sample_data.items():
            self.write_line(f"\t{param_name}:")
            self.write_line(f"\t\tmin {np.min(values):.3f}, max {np.max(values):.3f}")
            self.write_line(f"\t\tmean {np.mean(values):.3f}, median {np.median(values):.3f}")
        self.write_line("-----")
        end_time = datetime.now().strftime("%A %d %B %Y at %H:%M")
        self.write_line(f"Finished on {end_time}")
        self.text_file.close()

        
def _init_preprocess_dataset(dataset_name: str, datasets_root: Path, out_dir: Path) -> Tuple[Optional[Path], Optional[DatasetLog]]:
    """Initializes the output directory and logger for a specific dataset."""
    dataset_root = datasets_root.joinpath(dataset_name)
    if not dataset_root.exists():
        print(f"Couldn't find {dataset_root}, skipping this dataset.")
        return None, None
    return dataset_root, DatasetLog(out_dir, dataset_name)


def _preprocess_speaker_dirs(speaker_dirs: List[Path], dataset_name: str, datasets_root: Path, 
                             out_dir: Path, extension: str, skip_existing: bool, logger: DatasetLog):
    """
    Iterates over speaker directories, processing audio files in parallel.
    """
    print(f"{dataset_name}: Preprocessing data for {len(speaker_dirs)} speakers.")
    
    # Function to preprocess utterances for one speaker
    def preprocess_speaker(speaker_dir: Path):
        # Give a name to the speaker that includes its dataset
        # e.g., LibriSpeech_train-clean-100_1234
        speaker_name = "_".join(speaker_dir.relative_to(datasets_root).parts)
        
        # Create output directory
        speaker_out_dir = out_dir.joinpath(speaker_name)
        speaker_out_dir.mkdir(exist_ok=True)
        sources_fpath = speaker_out_dir.joinpath("_sources.txt")
        
        # Check for existing work if resuming
        existing_fnames = set()
        if sources_fpath.exists():
            try:
                with sources_fpath.open("r") as sources_file:
                    existing_fnames = {line.split(",")[0] for line in sources_file}
            except Exception:
                existing_fnames = set()
        
        # Append to existing sources file or create new one
        sources_file = sources_fpath.open("a" if skip_existing else "w")
        
        # Gather all audio files recursively
        for in_fpath in speaker_dir.glob(f"**/*.{extension}"):
            # Construct output filename
            # e.g. chapter_verse -> chapter_verse.npy
            out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)
            out_fname = out_fname.replace(f".{extension}", ".npy")
            
            if skip_existing and out_fname in existing_fnames:
                continue
                
            # Load and preprocess the waveform
            wav = audio.preprocess_wav(in_fpath)
            if len(wav) == 0:
                continue
            
            # Create mel spectrogram
            frames = audio.wav_to_mel_spectrogram(wav)
            
            # Discard short utterances that can't form a full partial
            if len(frames) < partials_n_frames:
                continue
            
            # Save to disk
            out_fpath = speaker_out_dir.joinpath(out_fname)
            np.save(out_fpath, frames)
            
            # Log metadata
            logger.add_sample(duration=len(wav) / sampling_rate)
            sources_file.write(f"{out_fname},{in_fpath}\n")
        
        sources_file.close()
    
    # Multiprocess the speakers
    # Using ThreadPool is generally safe here as most work is I/O or numpy/scipy calls releasing GIL
    with ThreadPool(8) as pool:
        list(tqdm(pool.imap(preprocess_speaker, speaker_dirs), dataset_name, len(speaker_dirs),
                  unit="speakers"))
                  
    logger.finalize()
    print(f"Done preprocessing {dataset_name}.\n")


def preprocess_librispeech(datasets_root: Path, out_dir: Path, skip_existing: bool = False):
    """Preprocesses the LibriSpeech dataset."""
    for dataset_name in librispeech_datasets["train"]["other"]:
        # Initialize
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return 
        
        # Process speakers
        speaker_dirs = list(dataset_root.glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "flac",
                                 skip_existing, logger)


def preprocess_voxceleb1(datasets_root: Path, out_dir: Path, skip_existing: bool = False):
    """Preprocesses the VoxCeleb1 dataset, filtering for Anglophone speakers."""
    dataset_name = "VoxCeleb1"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

    # Parse metadata to filter by nationality
    meta_path = dataset_root.joinpath("vox1_meta.csv")
    if not meta_path.exists():
        print(f"Metadata file not found at {meta_path}. Cannot filter speakers.")
        return

    with meta_path.open("r") as metafile:
        metadata = [line.split("\t") for line in metafile][1:]
    
    # ID is index 0, Nationality is index 3
    nationalities = {line[0]: line[3] for line in metadata}
    
    # Filter speakers
    keep_speaker_ids = [sid for sid, nat in nationalities.items() if 
                        nat.lower() in anglophone_nationalites]
                        
    print(f"VoxCeleb1: using samples from {len(keep_speaker_ids)} (presumed anglophone) "
          f"speakers out of {len(nationalities)}.")
    
    # Locate speaker directories
    speaker_dirs = dataset_root.joinpath("wav").glob("*")
    speaker_dirs = [sd for sd in speaker_dirs if sd.name in keep_speaker_ids]
    
    print(f"VoxCeleb1: found {len(speaker_dirs)} anglophone speakers on disk, "
          f"{len(keep_speaker_ids) - len(speaker_dirs)} missing (normal if subset).")

    # Process
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                             skip_existing, logger)


def preprocess_voxceleb2(datasets_root: Path, out_dir: Path, skip_existing: bool = False):
    """Preprocesses the VoxCeleb2 dataset."""
    dataset_name = "VoxCeleb2"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return
    
    # VoxCeleb2 structure: dev/aac/idXXXX
    speaker_dirs = list(dataset_root.joinpath("dev", "aac").glob("*"))
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "m4a",
                             skip_existing, logger)
