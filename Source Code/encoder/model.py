# ==================================================================================================
# DEEPFAKE AUDIO - encoder/model.py (Neural Architecture Definition)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module defines the SpeakerEncoder class, a three-layer LSTM-based neural 
# network inspired by the 'Generalized End-to-End Loss for Speaker Verification' 
# research. It maps variable-length speech features into fixed-dimensional 
# embeddings (d-vectors) that represent the unique vocal characteristics of the 
# speaker, enabling zero-shot voice cloning.
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

from encoder.params_model import *
from encoder.params_data import *
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq
from torch import nn
import numpy as np
import torch

class SpeakerEncoder(nn.Module):
    """
    Spatio-Temporal Identity Extractor:
    An LSTM architecture designed to condense acoustic feature sequences into 
    latent speaker representations.
    """
    def __init__(self, device, loss_device):
        super().__init__()
        self.loss_device = loss_device
        
        # --- RECURRENT BACKBONE ---
        # Multi-layer LSTM to capture temporal acoustic dependencies.
        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=model_hidden_size, 
                            num_layers=model_num_layers, 
                            batch_first=True).to(device)
        
        # --- PROJECTION LAYER ---
        # Maps the final hidden state to the d-vector space.
        self.linear = nn.Linear(in_features=model_hidden_size, 
                                out_features=model_embedding_size).to(device)
        self.relu = torch.nn.ReLU().to(device)
        
        # --- COSINE SIMILARITY SCALE & BIAS ---
        # Learnable parameters to transform cosine similarities into optimized logit ranges.
        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(loss_device)

        # Optimization Criterion
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)
        
    def do_gradient_ops(self):
        """Manages gradient scaling and norm clipping for stable training dynamics."""
        # Sensitivity reduction for similarity parameters
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        # Global Gradient Constraint
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, utterances, hidden_init=None):
        """
        Neural Distillation:
        Processes a batch of mel-spectrograms [B, T, C] and returns d-vectors [B, E].
        """
        # Sequential temporal extraction
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # State aggregation: Extract identity from the final LSTM layer's last state
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        # L2-Normalization: Project onto the identity hypersphere (unit length)
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        

        return embeds
    
    def similarity_matrix(self, embeds):
        """
        Geometric Contrast: Computes the GE2E similarity matrix.
        Quantifies the proximity of d-vectors to speaker centroids.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Inclusive centroids: Mean identity representation per speaker
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        # Exclusive centroids: LOO (Leave-One-Out) means to avoid biased similarity scoring
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        # Similarity calculation via Dot Product (efficient Cosine Similarity equivalent)
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(self.loss_device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        # Scaling towards cross-entropy optimization
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix
    
    def loss(self, embeds):
        """
        Discriminant Optimization:
        Computes GE2E Softmax Loss and monitors Equal Error Rate (EER).
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Global Similarity Awareness
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        
        # Target Generation (Diagonal Mapping)
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        loss = self.loss_fn(sim_matrix, target)
        
        # Equal Error Rate (Diagnostic Telemetry)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # Statistical Error Estimation
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer
