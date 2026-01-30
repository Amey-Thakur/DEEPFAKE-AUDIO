# ==================================================================================================
# DEEPFAKE AUDIO - synthesizer/utils/numbers.py (Numeric Expansion Engine)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements the logic for expanding numeric expressions into 
# their verbal equivalents. it uses the 'inflect' library to normalize 
# integers, ordinals, currencies (USD/GBP), and decimals, ensuring natural-sounding 
# synthetic speech for numeric data.
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
import inflect

# Inflection engine for number-to-words conversion
_inflect = inflect.engine()

# Syntactic patterns for numeric normalization
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")

def _remove_commas(m):
    """Utility: Strips formatting commas from numeric strings."""
    return m.group(1).replace(",", "")

def _expand_decimal_point(m):
    """Linguistic Translation: Replaces decimal dots with verbal 'point'."""
    return m.group(1).replace(".", " point ")

def _expand_dollars(m):
    """Currency Localization: Converts monetary expressions ($) into verbal phrases."""
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"
        
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        return "%s %s" % (dollars, "dollar" if dollars == 1 else "dollars")
    elif cents:
        return "%s %s" % (cents, "cent" if cents == 1 else "cents")
    else:
        return "zero dollars"

def _expand_ordinal(m):
    """Sequential Alignment: Converts ordinals (1st, 2nd) into full words."""
    return _inflect.number_to_words(m.group(0))

def _expand_number(m):
    """
    Mathematical Expansion:
    Converts integers into natural verbal sequences.
    Handles specific year-like representations for values between 1000 and 3000.
    """
    num = int(m.group(0))
    if 1000 < num < 3000:
        if num == 2000:
            return "two thousand"
        elif 2000 < num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")

def normalize_numbers(text):
    """Orchestration: Sequential application of all numeric expansion filters."""
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
