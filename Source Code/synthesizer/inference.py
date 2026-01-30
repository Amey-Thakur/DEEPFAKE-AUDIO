# ==================================================================================================
# DEEPFAKE AUDIO - synthesizer/inference.py (Acoustic Distillation Engine)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements the Synthesizer class, which orchestrates the loading 
# and inference of the Tacotron model. it converts textual sequences and speaker 
# embeddings (identity vectors) into high-resolution Mel-Spectrograms, enabling 
# zero-shot multispeaker voice synthesis.
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

import torch
from synthesizer import audio
from synthesizer.hparams import hparams
from synthesizer.models.tacotron import Tacotron
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import text_to_sequence
from vocoder.display import simple_table
from pathlib import Path
from typing import Union, List
import numpy as np
import librosa

class Synthesizer:
    """
    Neural Voice Orchestrator:
    Manages the lifecycle of the Tacotron model and provides high-level APIs 
    for text-to-spectrogram synthesis.
    """
    sample_rate = hparams.sample_rate
    hparams = hparams

    def __init__(self, model_fpath: Path, verbose=True):
        """
        Lazy Initialization:
        The model remains un-instantiated until the first synthesis request or 
        explicit load() call to conserve memory.
        """
        self.model_fpath = model_fpath
        self.verbose = verbose

        # Hardware Auto-detection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        if self.verbose:
            print("Synthesizer using device:", self.device)

        self._model = None

    def is_loaded(self):
        """Verifies if the neural weights are resident in memory."""
        return self._model is not None

    def load(self):
        """Materializes the Tacotron architecture and loads pre-trained checkpoints."""
        self._model = Tacotron(embed_dims=hparams.tts_embed_dims,
                               num_chars=len(symbols),
                               encoder_dims=hparams.tts_encoder_dims,
                               decoder_dims=hparams.tts_decoder_dims,
                               n_mels=hparams.num_mels,
                               fft_bins=hparams.num_mels,
                               postnet_dims=hparams.tts_postnet_dims,
                               encoder_K=hparams.tts_encoder_K,
                               lstm_dims=hparams.tts_lstm_dims,
                               postnet_K=hparams.tts_postnet_K,
                               num_highways=hparams.tts_num_highways,
                               dropout=hparams.tts_dropout,
                               stop_threshold=hparams.tts_stop_threshold,
                               speaker_embedding_size=hparams.speaker_embedding_size).to(self.device)

        self._model.load(self.model_fpath)
        self._model.eval()

        if self.verbose:
            print("Loaded synthesizer \"%s\" trained to step %d" % 
                  (self.model_fpath.name, self._model.state_dict()["step"]))

    def synthesize_spectrograms(self, texts: List[str],
                                embeddings: Union[np.ndarray, List[np.ndarray]],
                                return_alignments=False):
        """
        Batch Inference:
        Converts a list of texts and identity embeddings into Mel-Spectrograms.
        """
        if not self.is_loaded():
            self.load()

        # Text-to-Phoneme Preprocessing
        inputs = [text_to_sequence(text.strip(), hparams.tts_cleaner_names) for text in texts]
        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        # Batch Orchestration
        batched_inputs = [inputs[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(inputs), hparams.synthesis_batch_size)]
        batched_embeds = [embeddings[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(embeddings), hparams.synthesis_batch_size)]

        specs = []
        for i, batch in enumerate(batched_inputs, 1):
            if self.verbose:
                print(f"\n| Generating {i}/{len(batched_inputs)}")

            # Temporal Padding for batch consistency
            text_lens = [len(text) for text in batch]
            max_text_len = max(text_lens)
            chars = [pad1d(text, max_text_len) for text in batch]
            chars = np.stack(chars)

            # Identity Vector Aggregation
            speaker_embeds = np.stack(batched_embeds[i-1])

            # Tensor Conversion
            chars = torch.tensor(chars).long().to(self.device)
            speaker_embeddings = torch.tensor(speaker_embeds).float().to(self.device)

            # Neural Forward Pass
            _, mels, alignments = self._model.generate(chars, speaker_embeddings)
            mels = mels.detach().cpu().numpy()
            
            for m in mels:
                # Stochastic Silence Trimming: Removes tail-end artifacts
                while np.max(m[:, -1]) < hparams.tts_stop_threshold:
                    m = m[:, :-1]
                specs.append(m)

        if self.verbose:
            print("\n\nDone.\n")
        return (specs, alignments) if return_alignments else specs

    @staticmethod
    def load_preprocess_wav(fpath):
        """Standardized Audio Ingestion: Loads and rescales audio for synthesis consistency."""
        wav = librosa.load(str(fpath), hparams.sample_rate)[0]
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max
        return wav

    @staticmethod
    def make_spectrogram(fpath_or_wav: Union[str, Path, np.ndarray]):
        """Acoustic Transformation: Generates a Mel-Spectrogram from audio input."""
        if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
            wav = Synthesizer.load_preprocess_wav(fpath_or_wav)
        else:
            wav = fpath_or_wav

        mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
        return mel_spectrogram

    @staticmethod
    def griffin_lim(mel):
        """Acoustic Reconstruction: Approximates waveform from Mel space via Griffin-Lim."""
        return audio.inv_mel_spectrogram(mel, hparams)

def pad1d(x, max_len, pad_value=0):
    """Utility for 1D sequence padding."""
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)
