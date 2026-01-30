# ==================================================================================================
# DEEPFAKE AUDIO - synthesizer/utils/text.py (Linguistic Tokenization Engine)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements the text-to-sequence transformation logic. It handles 
# cleaning, ARPAbet embedding (via curly brace detection), and numeric ID 
# mapping, converting human-readable text into neural embeddings for the 
# synthesizer.
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

from synthesizer.utils.symbols import symbols
from synthesizer.utils import cleaners
import re

# Neural Mapping: Bidi dictionary for character/ID conversion
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# ARPAbet Detection: Identifies phonetic sequences in curly braces
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

def text_to_sequence(text, cleaner_names):
    """
    Symbolic Ingestion:
    Converts raw text into a categorical sequence of token IDs. 
    Supports hybrid text/ARPAbet inputs via curly brace tagging.
    """
    sequence = []

    # Sequential Parsing Segment by Segment
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    # Termination: Append EOS token
    sequence.append(_symbol_to_id["~"])
    return sequence

def sequence_to_text(sequence):
    """Linguistic Restoration: Decodes token ID sequences back into human-readable text."""
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Handle ARPAbet re-bracketing
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")

def _clean_text(text, cleaner_names):
    """Pipeline Orchestration: Runs text through specified normalization filters."""
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text

def _symbols_to_sequence(symbols):
    """ID Conversion: Maps a list of character tokens to their numeric analogues."""
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]

def _arpabet_to_sequence(text):
    """Phonetic Encoding: Converts space-delimited ARPAbet codes into ID sequences."""
    return _symbols_to_sequence(["@" + s for s in text.split()])

def _should_keep_symbol(s):
    """Filter Logic: Ensures only valid tokens are included in the sequence."""
    return s in _symbol_to_id and s not in ("_", "~")
