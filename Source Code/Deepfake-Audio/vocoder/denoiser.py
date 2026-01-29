from vocoder.models.fatchord_version import WaveRNN
from vocoder import hparams as hp
import torch
import numpy as np

def denoise(wav, v, device):
    """
    Denoises the waveform using a bias adaptation method.
    It estimates the bias from a silent spectrogram and subtracts it.
    
    Args:
        wav: The generated audio waveform (numpy array)
        v: The trained WaveRNN model
        device: The torch device (cpu or cuda)
    """
    # Create a zero-spectrogram (silence)
    mel_zeros = np.zeros(shape=(1, hp.num_mels, 88))
    mel_zeros = torch.tensor(mel_zeros).float().to(device)

    # Generate silence waveform to capture the bias/noise profile
    with torch.no_grad():
        bias_wav = v.generate(mel_zeros, batched=True, target=11000, overlap=1100, mu_law=hp.mu_law)
    
    # Scale the bias to match the waveform length
    bias_wav = bias_wav.astype(np.float32)
    
    # If bias is shorter, tile it. If longer, crop it.
    if len(bias_wav) < len(wav):
        bias_wav = np.tile(bias_wav, int(np.ceil(len(wav) / len(bias_wav))))[:len(wav)]
    else:
        bias_wav = bias_wav[:len(wav)]
        
    # Subtract bias (simple spectral subtraction equivalent in waveform domain if strictly additive)
    # For WaveRNN, we simply subtract the "silence noise"
    denoised_wav = wav.astype(np.float32) - bias_wav.astype(np.float32)
    
    return denoised_wav
