# ==================================================================================================
# DEEPFAKE AUDIO - synthesizer/utils/__init__.py (Utility Namespace)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module provides foundational utilities for the synthesizer, including 
# a DataParallel workaround for distributed training and a ValueWindow class 
# for stable metric observability.
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

import torch

_output_ref = None
_replicas_ref = None

def data_parallel_workaround(model, *input):
    """
    Distributed Proxy:
    Handles manual model replication and data scattering to bypass known 
    Python-level DataParallel bottlenecks.
    """
    global _output_ref
    global _replicas_ref
    device_ids = list(range(torch.cuda.device_count()))
    output_device = device_ids[0]
    replicas = torch.nn.parallel.replicate(model, device_ids)
    inputs = torch.nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = torch.nn.parallel.parallel_apply(replicas, inputs)
    y_hat = torch.nn.parallel.gather(outputs, output_device)
    _output_ref = outputs
    _replicas_ref = replicas
    return y_hat

class ValueWindow():
    """
    Stochastic Metric Observer:
    Maintains a sliding window of values to compute stable statistical averages 
    during training telemetry.
    """
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        """Archives a new value and maintains the fixed window aperture."""
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        """Returns the aggregate sum of values in the current window."""
        return sum(self._values)

    @property
    def count(self):
        """Returns the current density of the window (number of elements)."""
        return len(self._values)

    @property
    def average(self):
        """Calculates the temporal mean of the historical window."""
        return self.sum / max(1, self.count)

    def reset(self):
        """Clears the value archive for a fresh telemetry session."""
        self._values = []
