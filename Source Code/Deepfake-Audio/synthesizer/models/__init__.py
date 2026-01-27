"""
Deepfake Audio - Synthesizer Models Package
-------------------------------------------
This package contains the Tacotron-2 model definitions and sub-modules.

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
from .tacotron import Tacotron


def create_model(name, hparams):
  if name == "Tacotron":
    return Tacotron(hparams)
  else:
    raise Exception("Unknown model: " + name)
