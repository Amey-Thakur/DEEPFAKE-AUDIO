# ==================================================================================================
# DEEPFAKE AUDIO - encoder/preprocess.py (Acoustic Feature Extraction Engine)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements the internal orchestration for dataset preprocessing. 
# It provides highly-parallelized functions to traverse raw speech corpora 
# (LibriSpeech, VoxCeleb), extract speaker-specific metadata, and materialize 
# normalized Mel-Spectrograms onto the disk for high-throughput training.
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

from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from tqdm import tqdm

# --- INTERNAL SIGNAL UTILITIES ---
from encoder import audio
from encoder.config import librispeech_datasets, anglophone_nationalites
from encoder.params_data import *

_AUDIO_EXTENSIONS = ("wav", "flac", "m4a", "mp3")

class DatasetLog:
    """
    Experimental Ledger: Records the metadata and parameter state of a 
    preprocessing run for reproducibility.
    """
    def __init__(self, root, name):
        self.text_file = open(Path(root, "Log_%s.txt" % name.replace("/", "_")), "w")
        self.sample_data = dict()

        start_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("🚀 Initiating Preprocessing Cycle for: %s on %s" % (name, start_time))
        self.write_line("-----")
        self._log_params()

    def _log_params(self):
        """Archives the effective hyperparameters in the log."""
        from encoder import params_data
        self.write_line("Acoustic Hyperparameters:")
        for param_name in (p for p in dir(params_data) if not p.startswith("__")):
            value = getattr(params_data, param_name)
            self.write_line("\t%s: %s" % (param_name, value))
        self.write_line("-----")

    def write_line(self, line):
        self.text_file.write("%s\n" % line)

    def add_sample(self, **kwargs):
        """Accumulates statistical distributions of processed samples."""
        for param_name, value in kwargs.items():
            if not param_name in self.sample_data:
                self.sample_data[param_name] = []
            self.sample_data[param_name].append(value)

    def finalize(self):
        """Computes and writes dataset-wide statistics before closing."""
        self.write_line("Neural Statistical Profile:")
        for param_name, values in self.sample_data.items():
            self.write_line("\t%s:" % param_name)
            self.write_line("\t\tmin %.3f, max %.3f" % (np.min(values), np.max(values)))
            self.write_line("\t\tmean %.3f, median %.3f" % (np.mean(values), np.median(values)))
        self.write_line("-----")
        end_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("🤝🏻 Cycle Completed on %s" % end_time)
        self.text_file.close()

def _init_preprocess_dataset(dataset_name, datasets_root, out_dir) -> (Path, DatasetLog):
    """Integrity check and log initialization for a specific corpus."""
    dataset_root = datasets_root.joinpath(dataset_name)
    if not dataset_root.exists():
        print("⚠️ Scholarly Alert: Dataset %s not found. Skipping." % dataset_root)
        return None, None
    return dataset_root, DatasetLog(out_dir, dataset_name)

def _preprocess_speaker(speaker_dir: Path, datasets_root: Path, out_dir: Path, skip_existing: bool):
    """
    Utterance Orchestration: Processes all vocal samples for a single speaker.
    Extracts Mel-Spectrograms and archives source mappings.
    """
    speaker_name = "_".join(speaker_dir.relative_to(datasets_root).parts)
    speaker_out_dir = out_dir.joinpath(speaker_name)
    speaker_out_dir.mkdir(exist_ok=True)
    sources_fpath = speaker_out_dir.joinpath("_sources.txt")

    # Resilience: Load existing sources index
    existing_fnames = {}
    if sources_fpath.exists():
        try:
            with sources_fpath.open("r") as sources_file:
                existing_fnames = {line.split(",")[0] for line in sources_file}
        except:
            pass

    # Process utterances recursively
    sources_file = sources_fpath.open("a" if skip_existing else "w")
    audio_durs = []
    for extension in _AUDIO_EXTENSIONS:
        for in_fpath in speaker_dir.glob("**/*.%s" % extension):
            out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)
            out_fname = out_fname.replace(".%s" % extension, ".npy")
            
            if skip_existing and out_fname in existing_fnames:
                continue

            # Signal Normalization
            wav = audio.preprocess_wav(in_fpath)
            if len(wav) == 0:
                continue

            # Spectral Analysis
            frames = audio.wav_to_mel_spectrogram(wav)
            if len(frames) < partials_n_frames:
                continue

            # Materialization
            out_fpath = speaker_out_dir.joinpath(out_fname)
            np.save(out_fpath, frames)
            sources_file.write("%s,%s\n" % (out_fname, in_fpath))
            audio_durs.append(len(wav) / sampling_rate)

    sources_file.close()
    return audio_durs

def _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger):
    """Degree of parallelism management for speaker-level processing."""
    print("🤝🏻 Processing %d speakers from: %s" % (len(speaker_dirs), dataset_name))

    work_fn = partial(_preprocess_speaker, datasets_root=datasets_root, out_dir=out_dir, skip_existing=skip_existing)
    with Pool(4) as pool:
        tasks = pool.imap(work_fn, speaker_dirs)
        for sample_durs in tqdm(tasks, dataset_name, len(speaker_dirs), unit="speakers"):
            for sample_dur in sample_durs:
                logger.add_sample(duration=sample_dur)

    logger.finalize()

def preprocess_librispeech(datasets_root: Path, out_dir: Path, skip_existing=False):
    """Pipeline for LibriSpeech: A massive corpus of read English speech."""
    for dataset_name in librispeech_datasets["train"]["other"]:
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root: continue
        
        speaker_dirs = list(dataset_root.glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger)

def preprocess_voxceleb1(datasets_root: Path, out_dir: Path, skip_existing=False):
    """Pipeline for VoxCeleb1: Celebrity voices with multi-national diversity."""
    dataset_name = "VoxCeleb1"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root: return

    # Meta-Guided Filtering for Anglophone Speakers
    with dataset_root.joinpath("vox1_meta.csv").open("r") as metafile:
        metadata = [line.split("\t") for line in metafile][1:]

    nationalities = {line[0]: line[3] for line in metadata}
    keep_speaker_ids = [s_id for s_id, nat in nationalities.items() if nat.lower() in anglophone_nationalites]
    
    speaker_dirs = [d for d in dataset_root.joinpath("wav").glob("*") if d.name in keep_speaker_ids]
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger)

def preprocess_voxceleb2(datasets_root: Path, out_dir: Path, skip_existing=False):
    """Pipeline for VoxCeleb2: Broadscale celebrity speech data."""
    dataset_name = "VoxCeleb2"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root: return

    speaker_dirs = list(dataset_root.joinpath("dev", "aac").glob("*"))
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger)
