# ==================================================================================================
# DEEPFAKE AUDIO - synthesizer/utils/_cmudict.py (Phonetic Lexicon Interface)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module provides a thin interface for the CMU Pronouncing Dictionary. 
# It facilitates the mapping of English words to their ARPAbet phonetic 
# representations, supporting high-fidelity phoneme-to-speech synthesis.
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

import re

# ARPAbet Token Set: Standard phonetic symbols for English
valid_symbols = [
  "AA", "AA0", "AA1", "AA2", "AE", "AE0", "AE1", "AE2", "AH", "AH0", "AH1", "AH2",
  "AO", "AO0", "AO1", "AO2", "AW", "AW0", "AW1", "AW2", "AY", "AY0", "AY1", "AY2",
  "B", "CH", "D", "DH", "EH", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2", "EY",
  "EY0", "EY1", "EY2", "F", "G", "HH", "IH", "IH0", "IH1", "IH2", "IY", "IY0", "IY1",
  "IY2", "JH", "K", "L", "M", "N", "NG", "OW", "OW0", "OW1", "OW2", "OY", "OY0",
  "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH", "UH0", "UH1", "UH2", "UW",
  "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH"
]

_valid_symbol_set = set(valid_symbols)

class CMUDict:
    """
    Phonetic Registry:
    Encapsulates the CMU Dictionary data for fast lookups.
    Ref: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
    """
    def __init__(self, file_or_path, keep_ambiguous=True):
        if isinstance(file_or_path, str):
            with open(file_or_path, encoding="latin-1") as f:
                entries = _parse_cmudict(f)
        else:
            entries = _parse_cmudict(file_or_path)
            
        if not keep_ambiguous:
            entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
        self._entries = entries

    def __len__(self):
        return len(self._entries)

    def lookup(self, word):
        """Linguistic Inquiry: Returns ARPAbet pronunciations for a specific word."""
        return self._entries.get(word.upper())

_alt_re = re.compile(r"\([0-9]+\)")

def _parse_cmudict(file):
    """Parses raw text stream into a structured phonetic dictionary."""
    cmudict = {}
    for line in file:
        if len(line) and (line[0] >= "A" and line[0] <= "Z" or line[0] == "'"):
            parts = line.split("  ")
            word = re.sub(_alt_re, "", parts[0])
            pronunciation = _get_pronunciation(parts[1])
            if pronunciation:
                if word in cmudict:
                    cmudict[word].append(pronunciation)
                else:
                    cmudict[word] = [pronunciation]
    return cmudict

def _get_pronunciation(s):
    """Validates and formats the ARPAbet symbol sequence."""
    parts = s.strip().split(" ")
    for part in parts:
        if part not in _valid_symbol_set:
            return None
    return " ".join(parts)
