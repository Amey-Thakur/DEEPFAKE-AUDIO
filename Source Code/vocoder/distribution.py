# ==================================================================================================
# DEEPFAKE AUDIO - vocoder/distribution.py (Probabilistic Modeling Engine)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements the discretized mixture of logistics (MoL) 
# distribution, essential for modeling the stochastic nature of speech 
# waveforms. It provides the loss functions and sampling routines used by 
# the vocoder to generate high-fidelity output.
#
# 👤 AUTHORS
# - Amey Thakur (https://github.com/Amey-Thakur)
# - Mega Satish (https://github.com/msatmod)
#
# 🤝🏻 CREDITS
# Original Real-Time Voice Cloning methodology by CorentinJ
# Repository: https://github.com/CorentinJ/Real-Time-Voice-Cloning
# Mixture implementation adapted from: https://github.com/r9y9/wavenet_vocoder
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

import numpy as np
import torch
import torch.nn.functional as F

def log_sum_exp(x):
    """Numeric Stabilizer: Computes log(sum(exp(x))) preventing overflow/underflow."""
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def discretized_mix_logistic_loss(y_hat, y, num_classes=65536, log_scale_min=None, reduce=True):
    """
    MoL Loss Objective:
    Computes the negative log-likelihood for a discretized mixture of logistics, 
    optimizing model parameters for high-fidelity audio reconstruction.
    """
    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))
    y_hat = y_hat.permute(0,2,1)
    assert y_hat.dim() == 3
    assert y_hat.size(1) % 3 == 0
    nr_mix = y_hat.size(1) // 3

    y_hat = y_hat.transpose(1, 2)

    # Dimensional Decomposition: Probs, Means, and Scales
    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix:2 * nr_mix]
    log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min)

    y = y.expand_as(means)

    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # Logistic Mapping logic for boundary conditions
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * \
        torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
        (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
    inner_cond = (y > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (y < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    log_probs = log_probs + F.log_softmax(logit_probs, -1)

    if reduce:
        return -torch.mean(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)

def sample_from_discretized_mix_logistic(y, log_scale_min=None):
    """
    Probabilistic Sampling:
    Picks samples from the neural MoL distribution to generate one audio frame.
    """
    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))
    assert y.size(1) % 3 == 0
    nr_mix = y.size(1) // 3

    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]

    # Softmax Gumbel-style Mixture Sampling
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=-1)

    # Feature Extraction for selected mixture
    one_hot = to_one_hot(argmax, nr_mix)
    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.clamp(torch.sum(
        y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1), min=log_scale_min)
    
    # Stochastic Amalgamation: Inverse CDF sampling from selected logistic component
    u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    return torch.clamp(x, min=-1., max=1.)

def to_one_hot(tensor, n, fill_with=1.):
    """Categorical Transformer: Standard One-Hot encoding for latent selection."""
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot
