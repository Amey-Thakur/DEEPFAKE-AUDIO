# ==================================================================================================
# DEEPFAKE AUDIO - tests/test_ci_smoke.py (Integrity Verification Suite)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This script performs "Smoke Testing" to verify the basic integrity of the neural 
# pipeline and its dependencies. It ensures that critical third-party libraries 
# (Torch, Librosa, etc.) and internal project modules are correctly resolved 
# within the execution environment, acting as a gatekeeper for CI/CD workflows.
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

import os
import sys

# Ensure the repository root is on sys.path for imports like `import encoder`
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_third_party_imports():
    import librosa  # noqa: F401
    import numpy  # noqa: F401
    import soundfile  # noqa: F401
    import torch  # noqa: F401


def test_project_imports():
    import encoder  # noqa: F401
    import synthesizer  # noqa: F401
    import vocoder  # noqa: F401
