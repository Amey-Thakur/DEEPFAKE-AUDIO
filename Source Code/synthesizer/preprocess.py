# ==================================================================================================
# DEEPFAKE AUDIO - synthesizer/preprocess.py (Corpus Orchestration & Normalization)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements the data preprocessing pipeline for the synthesizer. 
# It handles the ingestion of raw speech datasets (LibriSpeech, LibriTTS), 
# audio-text alignment check, noise reduction via logMMSE, silence-based 
# temporal segmentation, and the serialization of Mel-Spectrograms and 
# speaker embeddings.
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

from multiprocessing.pool import Pool
from synthesizer import audio
from functools import partial
from itertools import chain
from encoder import inference as encoder
from pathlib import Path
from utils import logmmse
from tqdm import tqdm
import numpy as np
import librosa

def preprocess_dataset(datasets_root: Path, out_dir: Path, n_processes: int, skip_existing: bool, hparams,
                       no_alignments: bool, datasets_name: str, subfolders: str):
    """
    Mass Corpus Processing:
    Orchestrates the conversion of a raw dataset into neural-ready formats.
    """
    dataset_root = datasets_root.joinpath(datasets_name)
    input_dirs = [dataset_root.joinpath(subfolder.strip()) for subfolder in subfolders.split(",")]
    print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
    assert all(input_dir.exists() for input_dir in input_dirs)

    # Physical Layer Initialization
    out_dir.joinpath("mels").mkdir(exist_ok=True)
    out_dir.joinpath("audio").mkdir(exist_ok=True)

    # Telemetry: Tracking the metadata stream
    metadata_fpath = out_dir.joinpath("train.txt")
    metadata_file = metadata_fpath.open("a" if skip_existing else "w", encoding="utf-8")

    # Distributed Execution: Multiprocessing for high-throughput I/O
    speaker_dirs = list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))
    func = partial(preprocess_speaker, out_dir=out_dir, skip_existing=skip_existing,
                   hparams=hparams, no_alignments=no_alignments)
    job = Pool(n_processes).imap(func, speaker_dirs)
    
    for speaker_metadata in tqdm(job, datasets_name, len(speaker_dirs), unit="speakers"):
        for metadatum in speaker_metadata:
            metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
    metadata_file.close()

    # Analytical Validation: Dataset statistics report
    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    sample_rate = hparams.sample_rate
    hours = (timesteps / sample_rate) / 3600
    print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
          (len(metadata), mel_frames, timesteps, hours))

def preprocess_speaker(speaker_dir, out_dir: Path, skip_existing: bool, hparams, no_alignments: bool):
    """Identifies and processes all vocal samples for a specific individual."""
    metadata = []
    for book_dir in speaker_dir.glob("*"):
        if no_alignments:
            # Flexible format ingestion: WAV, FLAC, MP3
            extensions = ["*.wav", "*.flac", "*.mp3"]
            for extension in extensions:
                wav_fpaths = book_dir.glob(extension)

                for wav_fpath in wav_fpaths:
                    wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
                    if hparams.rescale:
                        wav = wav / np.abs(wav).max() * hparams.rescaling_max

                    # Linguistic Linkage: Locating transcriptions
                    text_fpath = wav_fpath.with_suffix(".txt")
                    if not text_fpath.exists():
                        text_fpath = wav_fpath.with_suffix(".normalized.txt")
                        assert text_fpath.exists()
                    
                    with text_fpath.open("r") as text_file:
                        text = "".join([line for line in text_file])
                        text = text.replace("\"", "").strip()

                    metadata.append(process_utterance(wav, text, out_dir, str(wav_fpath.with_suffix("").name),
                                                      skip_existing, hparams))
        else:
            # Alignment-guided segmentation (LibriSpeech support)
            try:
                alignments_fpath = next(book_dir.glob("*.alignment.txt"))
                with alignments_fpath.open("r") as alignments_file:
                    alignments = [line.rstrip().split(" ") for line in alignments_file]
            except StopIteration:
                continue

            for wav_fname, words, end_times in alignments:
                wav_fpath = book_dir.joinpath(wav_fname + ".flac")
                assert wav_fpath.exists()
                words = words.replace("\"", "").split(",")
                end_times = list(map(float, end_times.replace("\"", "").split(",")))

                # Sub-utterance Fractionation
                wavs, texts = split_on_silences(wav_fpath, words, end_times, hparams)
                for i, (wav, text) in enumerate(zip(wavs, texts)):
                    sub_basename = "%s_%02d" % (wav_fname, i)
                    metadata.append(process_utterance(wav, text, out_dir, sub_basename,
                                                      skip_existing, hparams))

    return [m for m in metadata if m is not None]

def split_on_silences(wav_fpath, words, end_times, hparams):
    """
    Spatio-Temporal Segmentation:
    Fractures long utterances into smaller chunks based on acoustic silences.
    Includes noise profiling and reduction via logMMSE.
    """
    wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    words = np.array(words)
    start_times = np.array([0.0] + end_times[:-1])
    end_times = np.array(end_times)
    
    # Silence Detection Mask
    mask = (words == "") & (end_times - start_times >= hparams.silence_min_duration_split)
    mask[0] = mask[-1] = True
    breaks = np.where(mask)[0]

    # Acoustic Enhancement: Noise reduction
    silence_times = [[start_times[i], end_times[i]] for i in breaks]
    silence_times = (np.array(silence_times) * hparams.sample_rate).astype(np.int)
    noisy_wav = np.concatenate([wav[stime[0]:stime[1]] for stime in silence_times])
    if len(noisy_wav) > hparams.sample_rate * 0.02:
        profile = logmmse.profile_noise(noisy_wav, hparams.sample_rate)
        wav = logmmse.denoise(wav, profile, eta=0)

    # Segment Re-aggregation: Ensures mini-utterances meet minimum duration
    segments = list(zip(breaks[:-1], breaks[1:]))
    segment_durations = [start_times[end] - end_times[start] for start, end in segments]
    i = 0
    while i < len(segments) and len(segments) > 1:
        if segment_durations[i] < hparams.utterance_min_duration:
            left_duration = float("inf") if i == 0 else segment_durations[i - 1]
            right_duration = float("inf") if i == len(segments) - 1 else segment_durations[i + 1]
            joined_duration = segment_durations[i] + min(left_duration, right_duration)

            if joined_duration > hparams.hop_size * hparams.max_mel_frames / hparams.sample_rate:
                i += 1
                continue

            j = i - 1 if left_duration <= right_duration else i
            segments[j] = (segments[j][0], segments[j + 1][1])
            segment_durations[j] = joined_duration
            del segments[j + 1], segment_durations[j + 1]
        else:
            i += 1

    # Materialization: Final split segments
    segment_times = [[end_times[start], start_times[end]] for start, end in segments]
    segment_times = (np.array(segment_times) * hparams.sample_rate).astype(np.int)
    wavs = [wav[segment_time[0]:segment_time[1]] for segment_time in segment_times]
    texts = [" ".join(words[start + 1:end]).replace("  ", " ") for start, end in segments]

    return wavs, texts

def process_utterance(wav: np.ndarray, text: str, out_dir: Path, basename: str,
                      skip_existing: bool, hparams):
    """
    Unit Feature Extraction:
    Extracts and serializes features for a single utterance. 
    Applies VAD-based silence trimming before Mel-Spectrogram generation.
    """
    mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % basename)
    wav_fpath = out_dir.joinpath("audio", "audio-%s.npy" % basename)
    if skip_existing and mel_fpath.exists() and wav_fpath.exists():
        return None

    # VAD-based refinement
    if hparams.trim_silence:
        wav = encoder.preprocess_wav(wav, normalize=False, trim_silence=True)

    # Validity filtering
    if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
        return None

    # Feature Distillation: Mel-Spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None

    # Persistence Layer: Serialization as .npy arrays
    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)

    return wav_fpath.name, mel_fpath.name, "embed-%s.npy" % basename, len(wav), mel_frames, text

def embed_utterance(fpaths, encoder_model_fpath):
    """Neural Projection: Generates a fixed-dimensional speaker embedding."""
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    wav_fpath, embed_fpath = fpaths
    wav = np.load(wav_fpath)
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    np.save(embed_fpath, embed, allow_pickle=False)

def create_embeddings(synthesizer_root: Path, encoder_model_fpath: Path, n_processes: int):
    """Identity Ingestion: Orchestrates high-speed embedding generation for all samples."""
    wav_dir = synthesizer_root.joinpath("audio")
    metadata_fpath = synthesizer_root.joinpath("train.txt")
    assert wav_dir.exists() and metadata_fpath.exists()
    embed_dir = synthesizer_root.joinpath("embeds")
    embed_dir.mkdir(exist_ok=True)

    with metadata_fpath.open("r") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
        fpaths = [(wav_dir.joinpath(m[0]), embed_dir.joinpath(m[2])) for m in metadata]

    # Distributed Embedding Loop
    func = partial(embed_utterance, encoder_model_fpath=encoder_model_fpath)
    job = Pool(n_processes).imap(func, fpaths)
    list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))

