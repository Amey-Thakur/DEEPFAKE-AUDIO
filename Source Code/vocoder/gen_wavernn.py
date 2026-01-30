# ==================================================================================================
# DEEPFAKE AUDIO - vocoder/gen_wavernn.py (Waveform Generation Script)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This script provides a simplified interface for generating test samples 
# from a trained WaveRNN model. It iterates through a test set, generates 
# waveforms from Mel-Spectrograms, and saves them for qualitative analysis 
# and verification.
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

from vocoder.models.fatchord_version import WaveRNN
from vocoder.audio import *

def gen_testset(model: WaveRNN, test_set, samples, batched, target, overlap, save_path):
    """
    Regression Test Runner:
    Systematically generates audio from a test suite to monitor model performance.
    """
    k = model.get_step() // 1000

    for i, (m, x) in enumerate(test_set, 1):
        if i > samples: break

        print('\n| Generating Sample: %i/%i' % (i, samples))
        x = x[0].numpy()
        bits = 16 if hp.voc_mode == 'MOL' else hp.bits

        # Amplitudinal Restoration
        if hp.mu_law and hp.voc_mode != 'MOL':
            x = decode_mu_law(x, 2**bits, from_labels=True)
        else:
            x = label_2_float(x, bits)

        # Persistence: Save ground truth and generated cloned result
        save_wav(x, save_path.joinpath("%dk_steps_%d_target.wav" % (k, i)))
        
        batch_str = "gen_batched_target%d_overlap%d" % (target, overlap) if batched else "gen_not_batched"
        save_str = save_path.joinpath("%dk_steps_%d_%s.wav" % (k, i, batch_str))

        wav = model.generate(m, batched, target, overlap, hp.mu_law)
        save_wav(wav, save_str)

