# ==================================================================================================
# DEEPFAKE AUDIO - synthesizer/utils/cleaners.py (Linguistic Preprocessing)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements text normalization pipelines (cleaners) that process 
# input text before tokenization. It includes regex-based abbreviation 
# expansion, number normalization, and ASCII transliteration, ensuring 
# consistent textual ingestion for both training and inference.
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
from unidecode import unidecode
from synthesizer.utils.numbers import normalize_numbers

# Whitespace normalization pattern
_whitespace_re = re.compile(r"\s+")

# Lexicon Expansion: Titles and professional abbreviations
_abbreviations = [(re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1]) for x in [
    ("mrs", "misess"),
    ("mr", "mister"),
    ("dr", "doctor"),
    ("st", "saint"),
    ("co", "company"),
    ("jr", "junior"),
    ("maj", "major"),
    ("gen", "general"),
    ("drs", "doctors"),
    ("rev", "reverend"),
    ("lt", "lieutenant"),
    ("hon", "honorable"),
    ("sgt", "sergeant"),
    ("capt", "captain"),
    ("esq", "esquire"),
    ("ltd", "limited"),
    ("col", "colonel"),
    ("ft", "fort"),
]]

def expand_abbreviations(text):
    """Semantic Translation: Replaces common abbreviations with their full spoken forms."""
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def expand_numbers(text):
    """Numeric Distillation: Orchestrates number-to-words conversion."""
    return normalize_numbers(text)

def lowercase(text):
    """Canonicalization: Standardizes input tokens to lowercase."""
    return text.lower()

def collapse_whitespace(text):
    """Structural Normalization: Eliminates redundant spacing."""
    return re.sub(_whitespace_re, " ", text)

def convert_to_ascii(text):
    """Script Refinement: Transliterates non-ASCII characters via Unidecode."""
    return unidecode(text)

def basic_cleaners(text):
    """Base Pipeline: Lowercases and collapses whitespace; no transliteration."""
    return collapse_whitespace(lowercase(text))

def transliteration_cleaners(text):
    """International Pipeline: Translates non-English scripts to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    return collapse_whitespace(text)

def english_cleaners(text):
    """Advanced English Pipeline: Full normalization suite for Anglophonic synthesis."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    return collapse_whitespace(text)
