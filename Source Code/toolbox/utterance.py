# ==================================================================================================
# DEEPFAKE AUDIO - toolbox/utterance.py (Utterance Signature Definition)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module defines the Utterance data structure used within the toolbox to 
# store speech samples, their corresponding Mel-Spectrograms, and 256D 
# speaker embeddings.
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

from collections import namedtuple

# Data Container: Canonical representation of a speech sample in the toolbox
Utterance = namedtuple("Utterance", "name speaker_name wav spec embed partial_embeds synth")

# Functional Overrides: Ensure identity based on filename for set operations
Utterance.__eq__ = lambda x, y: x.name == y.name
Utterance.__hash__ = lambda x: hash(x.name)
