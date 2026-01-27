"""
Deepfake Audio - Number Normalization
-------------------------------------
Functions to expand numbers, currency, and abbreviations into spoken words.
This module is crucial for normalizing text input for TTS, converting numerical
representations into their verbal counterparts (e.g., "10" -> "ten").

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
import inflect

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _remove_commas(m):
  """Removes commas from a matched number string."""
  return m.group(1).replace(",", "")


def _expand_decimal_point(m):
  """Expands a decimal number match into words (e.g. '3.14' -> '3 point 14')."""
  return m.group(1).replace(".", " point ")


def _expand_dollars(m):
  """Expands a currency match into words (e.g. '$5.99' -> 'five dollars, ninety-nine cents')."""
  match = m.group(1)
  parts = match.split(".")
  if len(parts) > 2:
    return match + " dollars"  # Unexpected format
  dollars = int(parts[0]) if parts[0] else 0
  cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
  if dollars and cents:
    dollar_unit = "dollar" if dollars == 1 else "dollars"
    cent_unit = "cent" if cents == 1 else "cents"
    return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
  elif dollars:
    dollar_unit = "dollar" if dollars == 1 else "dollars"
    return "%s %s" % (dollars, dollar_unit)
  elif cents:
    cent_unit = "cent" if cents == 1 else "cents"
    return "%s %s" % (cents, cent_unit)
  else:
    return "zero dollars"


def _expand_ordinal(m):
  """Expands ordinal numbers into words (e.g. '1st' -> 'first')."""
  return _inflect.number_to_words(m.group(0))


def _expand_number(m):
  """Expands cardinal numbers into words (e.g. '42' -> 'forty-two').
  Special handling for years between 1000 and 3000 (e.g. 2020 -> 'twenty twenty')."""
  num = int(m.group(0))
  if num > 1000 and num < 3000:
    if num == 2000:
      return "two thousand"
    elif num > 2000 and num < 2010:
      return "two thousand " + _inflect.number_to_words(num % 100)
    elif num % 100 == 0:
      return _inflect.number_to_words(num // 100) + " hundred"
    else:
      return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
  else:
    return _inflect.number_to_words(num, andword="")


def normalize_numbers(text):
  """
  Main entry point for number normalization.
  Applies a sequence of regex substitutions to expand various number formats.
  
  Args:
      text: The input text containing numbers.
      
  Returns:
      The normalized text with numbers converted to words.
  """
  text = re.sub(_comma_number_re, _remove_commas, text)
  text = re.sub(_pounds_re, r"\1 pounds", text)
  text = re.sub(_dollars_re, _expand_dollars, text)
  text = re.sub(_decimal_number_re, _expand_decimal_point, text)
  text = re.sub(_ordinal_re, _expand_ordinal, text)
  text = re.sub(_number_re, _expand_number, text)
  return text
