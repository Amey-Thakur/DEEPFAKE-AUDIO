# ==================================================================================================
# DEEPFAKE AUDIO - synthesizer/utils/symbols.py (Character Token Registry)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module defines the canonical set of text symbols (tokens) used for 
# synthesis. It includes standard ASCII characters, padding, and EOS tokens, 
# ensuring consistent character-to-index mapping for the neural encoder.
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

# Core Tokens
_pad        = "_" # Alignment padding
_eos        = "~" # End of Sequence token
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? "

# Exported Symbol Set: Unified sequence of valid input tokens
symbols = [_pad, _eos] + list(_characters)
# _arpabet appended here if used
