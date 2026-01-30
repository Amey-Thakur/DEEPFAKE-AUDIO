# ==================================================================================================
# DEEPFAKE AUDIO - vocoder/hparams.py (Hyperparameter Configuration)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module defines the architectural and training hyperparameters for the 
# WaveRNN vocoder. It ensures consistent audio settings between the 
# synthesizer and vocoder, managing bit-depth, sampling rates, and neural 
# layer dimensions.
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

from synthesizer.hparams import hparams as _syn_hp

# Signal Conditioning: Core audio parameters synchronized with the synthesizer backend
sample_rate = _syn_hp.sample_rate
n_fft = _syn_hp.n_fft
num_mels = _syn_hp.num_mels
hop_length = _syn_hp.hop_size
win_length = _syn_hp.win_size
fmin = _syn_hp.fmin
min_level_db = _syn_hp.min_level_db
ref_level_db = _syn_hp.ref_level_db
mel_max_abs_value = _syn_hp.max_abs_value
preemphasis = _syn_hp.preemphasis
apply_preemphasis = _syn_hp.preemphasize

# Quantization: Bit depth of the output pulse-code modulation (PCM) signal
bits = 9                            # 9-bit resolution typically used for Mu-Law
mu_law = True                       # Non-linear quantization to preserve dynamic range

# Neural Architecture: WaveRNN structural parameters
voc_mode = 'RAW'                    # Synthesis Mode: 'RAW' (Softmax) or 'MOL' (Logistic Mixture)
voc_upsample_factors = (5, 5, 8)    # Transposed Conv factors (Product must equal hop_length)
voc_rnn_dims = 512                  # Hidden dimensions for the Gated Recurrent Unit (GRU)
voc_fc_dims = 512                   # Fully connected layer dimensions
voc_compute_dims = 128              # Embedding dimensions for categorical inputs
voc_res_out_dims = 128              # Residual block output channels
voc_res_blocks = 10                 # Depth of the residual sub-network

# Training Protocol: Weights optimization and data augmentation defaults
voc_batch_size = 100
voc_lr = 1e-4
voc_gen_at_checkpoint = 5           # Samples to generate for qualitative assessment
voc_pad = 2                         # Visual padding for the ResNet receptive field
voc_seq_len = hop_length * 5        # Optimization window size (multiples of hop_length)

# Inference Strategy: Parameters for real-time waveform synthesis
voc_gen_batched = True              # Enables high-speed parallel generation
voc_target = 8000                   # Chunk size for batched inference
voc_overlap = 400                   # Crossfade overlap to prevent boundary artifacts
