# ==================================================================================================
# DEEPFAKE AUDIO - encoder/data_objects/__init__.py (Data Layer Initialization)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This initialization script exposes the primary data orchestration classes for 
# speaker verification. It facilitates structured access to datasets and 
# loaders, abstraction layers that underpin the neural training pipeline.
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

from encoder.data_objects.speaker_verification_dataset import SpeakerVerificationDataset
from encoder.data_objects.speaker_verification_dataset import SpeakerVerificationDataLoader
