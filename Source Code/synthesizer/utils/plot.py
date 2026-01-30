# ==================================================================================================
# DEEPFAKE AUDIO - synthesizer/utils/plot.py (Neural Visualization Engine)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module provides visualization utilities for the synthesizer. It includes 
# functions for plotting attention alignments (showing the model's focus 
# during synthesis) and Mel-Spectrograms (predictive vs. target), 
# facilitating model diagnostics and qualitative evaluation.
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

import numpy as np

def split_title_line(title_text, max_words=5):
    """
    Title Formatter:
    Splits long strings into multiple lines for enhanced legibility in plots.
    """
    seq = title_text.split()
    return "\n".join([" ".join(seq[i:i + max_words]) for i in range(0, len(seq), max_words)])

def plot_alignment(alignment, path, title=None, split_title=False, max_len=None):
    """
    Attention Visualizer:
    Generates a heat map of the attention weights, illustrating the temporal 
    alignment between encoder (text) and decoder (audio) timesteps.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if max_len is not None:
        alignment = alignment[:, :max_len]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    im = ax.imshow(
        alignment,
        aspect="auto",
        origin="lower",
        interpolation="none")
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"

    if split_title:
        title = split_title_line(title)

    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()

def plot_spectrogram(pred_spectrogram, path, title=None, split_title=False, target_spectrogram=None, max_len=None, auto_aspect=False):
    """
    Spectrographic Comparison:
    Plots the predicted Mel-Spectrogram alongside the ground-truth target (if provided) 
    to visualize reconstruction accuracy and model fidelity.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if max_len is not None:
        target_spectrogram = target_spectrogram[:max_len]
        pred_spectrogram = pred_spectrogram[:max_len]

    if split_title:
        title = split_title_line(title)

    fig = plt.figure(figsize=(10, 8))
    fig.text(0.5, 0.18, title, horizontalalignment="center", fontsize=16)

    # Render target spectrogram for reference
    if target_spectrogram is not None:
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)

        if auto_aspect:
            im = ax1.imshow(np.rot90(target_spectrogram), aspect="auto", interpolation="none")
        else:
            im = ax1.imshow(np.rot90(target_spectrogram), interpolation="none")
        ax1.set_title("Target Mel-Spectrogram")
        fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
        ax2.set_title("Predicted Mel-Spectrogram")
    else:
        ax2 = fig.add_subplot(211)

    # Render predicted spectrogram
    if auto_aspect:
        im = ax2.imshow(np.rot90(pred_spectrogram), aspect="auto", interpolation="none")
    else:
        im = ax2.imshow(np.rot90(pred_spectrogram), interpolation="none")
    fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)

    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()
