# ==================================================================================================
# DEEPFAKE AUDIO - vocoder/display.py (Console Monitoring Engine)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module provides styling and progress visualization for the vocoder 
# training and generation processes. It includes progress bars, formatted 
# tables, and Matplotlib routines for saving attention maps and spectrograms 
# during model checkpoints.
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

import time
import numpy as np
import sys

def progbar(i, n, size=16):
    """Diagnostic UI: Renders a character-based progress bar for console feedback."""
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar

def stream(message):
    """Dynamic Output: Overwrites the current console line for live status updates."""
    try:
        sys.stdout.write("\r{%s}" % message)
    except:
        # Fallback: Cleanse non-ASCII characters to prevent encoding errors
        message = ''.join(i for i in message if ord(i)<128)
        sys.stdout.write("\r{%s}" % message)

def simple_table(item_tuples):
    """Information Grid: Prints a structured ASCII table for model parameters or stats."""
    border_pattern = '+---------------------------------------'
    whitespace = '                                            '
    headings, cells, = [], []

    for item in item_tuples:
        heading, cell = str(item[0]), str(item[1])
        pad_head = True if len(heading) < len(cell) else False
        pad = whitespace[:abs(len(heading) - len(cell))]
        pad_left = pad[:len(pad)//2]
        pad_right = pad[len(pad)//2:]

        if pad_head:
            heading = pad_left + heading + pad_right
        else:
            cell = pad_left + cell + pad_right

        headings += [heading]
        cells += [cell]

    border, head, body = '', '', ''
    for i in range(len(item_tuples)):
        temp_head = f'| {headings[i]} '
        temp_body = f'| {cells[i]} '
        border += border_pattern[:len(temp_head)]
        head += temp_head
        body += temp_body

        if i == len(item_tuples) - 1:
            head += '|'
            body += '|'
            border += '+'

    print(f"{border}\n{head}\n{border}\n{body}\n{border}\n ")

def time_since(started):
    """Temporal Tracker: Formats elapsed time into a human-readable string."""
    elapsed = time.time() - started
    m = int(elapsed // 60)
    s = int(elapsed % 60)
    if m >= 60:
        h = int(m // 60)
        m = m % 60
        return f'{h}h {m}m {s}s'
    else:
        return f'{m}m {s}s'

def save_attention(attn, path):
    """Attentive Visualizer: Generates and saves a PNG representation of attention maps."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(attn.T, interpolation='nearest', aspect='auto')
    fig.savefig(f'{path}.png', bbox_inches='tight')
    plt.close(fig)

def save_spectrogram(M, path, length=None):
    """Signal Visualizer: Persists Mel-Spectrogram snapshots to the filesystem."""
    import matplotlib.pyplot as plt
    M = np.flip(M, axis=0)
    if length: M = M[:, :length]
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(M, interpolation='nearest', aspect='auto')
    fig.savefig(f'{path}.png', bbox_inches='tight')
    plt.close(fig)

def plot(array):
    """Signal Debugger: Generates a large-scale plot for 1D signal exploration."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(30, 5))
    ax = fig.add_subplot(111)
    ax.xaxis.label.set_color('grey')
    ax.yaxis.label.set_color('grey')
    ax.xaxis.label.set_fontsize(23)
    ax.yaxis.label.set_fontsize(23)
    ax.tick_params(axis='x', colors='grey', labelsize=23)
    ax.tick_params(axis='y', colors='grey', labelsize=23)
    plt.plot(array)

def plot_spec(M):
    """Interactive Visualizer: Displays a spectrogram using standard pyplot routing."""
    import matplotlib.pyplot as plt
    M = np.flip(M, axis=0)
    plt.figure(figsize=(18,4))
    plt.imshow(M, interpolation='nearest', aspect='auto')
    plt.show()

