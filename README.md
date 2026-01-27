# DEEPFAKE-AUDIO

👉 An audio deepfake occurs when a "cloned" voice, potentially indistinguishable from the original subject, is used to produce synthetic audio.

- [Google Colaboratory](https://github.com/Amey-Thakur/DEEPFAKE-AUDIO/blob/main/Source%20Code/DEEPFAKE_AUDIO.ipynb)
- [Kaggle](https://www.kaggle.com/ameythakur20/deepfake-audio)
- [Model](https://drive.google.com/uc?id=1n1sPXvT34yXFLT47QZA6FIRGrwMeSsZc)
- [Project Demo](https://youtu.be/i3wnBcbHDbs)

---

### Overview

This project implements **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis (SV2TTS)**, utilizing a real-time vocoder for high-fidelity audio synthesis.

SV2TTS is a three-stage deep learning framework that generates a numerical representation of a voice from just a few seconds of audio, which is then used to condition a text-to-speech model for generating new vocalizations.

### Technical Implementation

The system is built on three core pillars derived from seminal research papers:

| Designation | Application | Implementation Source |
| :--- | :--- | :--- |
| **SV2TTS** | Transfer Learning for Multispeaker TTS | [1806.04558](https://arxiv.org/pdf/1806.04558.pdf) |
| **WaveRNN** | Efficient Neural Audio Synthesis (Vocoder) | [1802.08435](https://arxiv.org/pdf/1802.08435.pdf) |
| **Tacotron 2** | Mel Spectrogram Prediction (Synthesizer) | [1712.05884](https://arxiv.org/pdf/1712.05884.pdf) |
| **GE2E** | Generalized End-To-End Speaker Verification | [1710.10467](https://arxiv.org/pdf/1710.10467.pdf) |

---

<div align="center">

  **Created to study the functional mechanics of Deepfake Audio.**

  **Project Authors: Amey Thakur and Mega Satish**

  [**✌️ Back To Engineering ✌️**](https://github.com/Amey-Thakur/COMPUTER-ENGINEERING)

</div>
