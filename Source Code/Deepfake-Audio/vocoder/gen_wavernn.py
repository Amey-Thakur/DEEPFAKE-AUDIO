"""
Deepfake Audio - WaveRNN Generation Script
------------------------------------------
Script for generating audio waveforms from mel spectrograms using the WaveRNN model.
Used for testing and evaluation.

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
from vocoder.models.fatchord_version import  WaveRNN
from vocoder.audio import *


def gen_testset(model: WaveRNN, test_set, samples, batched, target, overlap, save_path: Path) -> None:
    """
    Generates audio for a set of test samples using the WaveRNN model.

    Args:
        model: Trained WaveRNN model.
        test_set: Dataset containing test samples (mel, wav).
        samples: Number of samples to generate.
        batched: Whether to use batched generation (faster).
        target: Target number of samples per batch entry.
        overlap: Overlap between batches.
        save_path: Directory to save the generated audio files.
    """
    k = model.get_step() // 1000

    for i, (m, x) in enumerate(test_set, 1):
        if i > samples: 
            break

        print('\n| Generating: %i/%i' % (i, samples))

        x = x[0].numpy()

        bits = 16 if hp.voc_mode == 'MOL' else hp.bits

        if hp.mu_law and hp.voc_mode != 'MOL' :
            x = decode_mu_law(x, 2**bits, from_labels=True)
        else :
            x = label_2_float(x, bits)

        save_wav(x, save_path.joinpath("%dk_steps_%d_target.wav" % (k, i)))
        
        batch_str = "gen_batched_target%d_overlap%d" % (target, overlap) if batched else \
            "gen_not_batched"
        save_str = save_path.joinpath("%dk_steps_%d_%s.wav" % (k, i, batch_str))

        wav = model.generate(m, batched, target, overlap, hp.mu_law)
        save_wav(wav, save_str)

