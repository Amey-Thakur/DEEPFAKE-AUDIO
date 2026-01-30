# ==================================================================================================
# DEEPFAKE AUDIO - synthesizer/hparams.py (Heuristic Hyperparameter Registry)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module defines the global hyperparameter registry (HParams) for the 
# synthesizer and vocoder. It centralizes architectural constants, signal 
# processing thresholds, and training schedules, ensuring consistent behavioral 
# configuration across the SV2TTS pipeline.
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

import ast
import pprint

class HParams(object):
    """
    Dynamic Configuration Container:
    A dot-accessible dictionary wrapper for managing experimental parameters.
    """
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def __setitem__(self, key, value): setattr(self, key, value)
    def __getitem__(self, key): return getattr(self, key)
    def __repr__(self): return pprint.pformat(self.__dict__)

    def parse(self, string):
        """Analytical override of hyperparameters via string injection (e.g., CLI)."""
        if len(string) > 0:
            overrides = [s.split("=") for s in string.split(",")]
            keys, values = zip(*overrides)
            keys = list(map(str.strip, keys))
            values = list(map(str.strip, values))
            for k in keys:
                self.__dict__[k] = ast.literal_eval(values[keys.index(k)])
        return self

# -- GLOBAL HEURISTICS --
hparams = HParams(
        ### Signal Processing (Fundamental Acoustic Constraints)
        sample_rate = 16000,
        n_fft = 800,
        num_mels = 80,
        hop_size = 200,                             # Tacotron 12.5ms frame shift
        win_size = 800,                             # Tacotron 50ms frame length
        fmin = 55,
        min_level_db = -100,
        ref_level_db = 20,
        max_abs_value = 4.,                         # Optimization stability constraint
        preemphasis = 0.97,                         # High-frequency accentuation coefficient
        preemphasize = True,

        ### Tacotron TTS Architecture (Neural Dimensionality)
        tts_embed_dims = 512,                       # Character representation depth
        tts_encoder_dims = 256,
        tts_decoder_dims = 128,
        tts_postnet_dims = 512,
        tts_encoder_K = 5,
        tts_lstm_dims = 1024,
        tts_postnet_K = 5,
        tts_num_highways = 4,
        tts_dropout = 0.5,
        tts_cleaner_names = ["english_cleaners"],
        tts_stop_threshold = -3.4,                  # End-of-sequence termination heuristic

        ### Progressive Training Schedule
        tts_schedule = [(2,  1e-3,  20_000,  12),   # Format: (r, lr, step, batch_size)
                        (2,  5e-4,  40_000,  12),   # r = reduction factor
                        (2,  2e-4,  80_000,  12),   # lr = learning rate
                        (2,  1e-4, 160_000,  12),   # step = target training step
                        (2,  3e-5, 320_000,  12),   # batch_size = sample count per GPU pass
                        (2,  1e-5, 640_000,  12)],

        tts_clip_grad_norm = 1.0,                   # Robustness: Prevent gradient explosions
        tts_eval_interval = 500,                    # Frequency of acoustic evaluation
        tts_eval_num_samples = 1,                   # Evaluation sampling density

        ### Data Preprocessing Logic
        max_mel_frames = 900,
        rescale = True,
        rescaling_max = 0.9,
        synthesis_batch_size = 16,                  # Inference parallelism

        ### Telemetry and Phase Recovery
        signal_normalization = True,
        power = 1.5,
        griffin_lim_iters = 60,                     # Iterative phase approximation quality

        ### Advanced Audio Engineering
        fmax = 7600,                                # Nyquist-constrained limit
        allow_clipping_in_normalization = True,     # Resiliency handle
        clip_mels_length = True,                    # Temporal standardization
        use_lws = False,                            # Fast phase recovery toggle
        symmetric_mels = True,                      # Bipolar vs. Unipolar Mel range
        trim_silence = True,                        # Ambient noise reduction

        ### SV2TTS Multispeaker Configuration
        speaker_embedding_size = 256,               # Identity vector dimensionality
        silence_min_duration_split = 0.4,           # Linguistic segmentation heuristic
        utterance_min_duration = 1.6,               # Temporal validity threshold
        )

def hparams_debug_string():
    """Diagnostic utility for observing active configuration state."""
    return str(hparams)
