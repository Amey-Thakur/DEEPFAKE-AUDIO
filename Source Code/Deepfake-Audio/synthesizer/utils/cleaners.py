"""
Deepfake Audio - Text Cleaners
------------------------------
Text cleaning pipelines for English and non-English text.
Includes transliteration, number expansion, and abbreviation expansion.
These cleaners are critical for normalizing input text before it is converted
to phonemes or processed by the synthesizer.

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

import re
from unidecode import unidecode
from .numbers import normalize_numbers

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
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
  """Expands common abbreviations (e.g. 'Dr.' -> 'Doctor')."""
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  """Expands number strings into words using the numbers module."""
  return normalize_numbers(text)


def lowercase(text):
  """Converts text to lowercase."""
  return text.lower()


def collapse_whitespace(text):
  """Replaces multiple whitespace characters with a single space."""
  return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
  """Converts unicode text to ASCII using unidecode transliteration."""
  return unidecode(text)


def basic_cleaners(text):
  """
  Basic text cleaning pipeline.
  Lowercases and collapses whitespace without transliteration.
  """
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  """
  Pipeline for non-English text.
  Transliterates text to ASCII, lowercases it, and collapses whitespace.
  Useful for languages that can be reasonably approximated by ASCII characters.
  """
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def english_cleaners(text):
  """
  Comprehensive pipeline for English text.
  1. Converts to ASCII.
  2. Lowercases.
  3. Expands numbers to words.
  4. Expands abbreviations.
  5. Collapses whitespace.
  """
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)
  return text
