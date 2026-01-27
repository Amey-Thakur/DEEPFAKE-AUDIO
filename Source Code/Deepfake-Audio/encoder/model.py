"""
Deepfake Audio - Encoder Model Architecture
-------------------------------------------
This module implements the Speaker Encoder network based on a multi-layer Long Short-Term Memory (LSTM)
architecture. It is trained using the Generalized End-to-End (GE2E) loss function to produce
embeddings where utterances from the same speaker are close in the embedding space (high cosine similarity),
while utterances from different speakers are far apart.

References:
    - "Generalized End-to-End Loss for Speaker Verification", Wan et al., 2018.

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

from typing import Tuple, Optional

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from torch import nn
from torch.nn.utils import clip_grad_norm_

# Internal Modules
from encoder.params_model import *
from encoder.params_data import mel_n_channels


class SpeakerEncoder(nn.Module):
    """
    The Speaker Encoder model capable of generating fixed-size embeddings (d-vectors) from
    variable-length audio segments.
    """
    
    def __init__(self, device: torch.device, loss_device: torch.device):
        """
        Initializes the Speaker Encoder.
        
        Args:
            device: The device (CPU/GPU) to run the forward pass on.
            loss_device: The device to compute the GE2E loss on (often CPU is efficient for this).
        """
        super().__init__()
        self.loss_device = loss_device
        
        # Network Architecture
        # 3-layer LSTM with projection
        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=model_hidden_size, 
                            num_layers=model_num_layers, 
                            batch_first=True).to(device)
        
        # Linear projection to the final embedding dimension
        self.linear = nn.Linear(in_features=model_hidden_size, 
                                out_features=model_embedding_size).to(device)
        
        self.relu = torch.nn.ReLU().to(device)
        
        # Cosine similarity scaling parameters (w and b in the paper)
        # These are learnable parameters that scale the cosine similarity before the softmax.
        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(loss_device)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)
        
    def do_gradient_ops(self):
        """
        Performs gradient scaling and clipping before the optimization step.
        Scaling the similarity parameters' gradients helps stabilize training.
        """
        # Gradient scale for geometric parameters
        if self.similarity_weight.grad is not None:
            self.similarity_weight.grad *= 0.01
        if self.similarity_bias.grad is not None:
            self.similarity_bias.grad *= 0.01
            
        # Gradient clipping to prevent exploding gradients in LSTM
        clip_grad_norm_(self.parameters(), 3.0, norm_type=2)
    
    def forward(self, utterances: torch.Tensor, hidden_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the embeddings for a batch of input spectrograms.
        
        Args:
            utterances: Batch of mel spectrograms. Shape: (batch_size, n_frames, n_channels).
            hidden_init: Initial hidden state for the LSTM. Shape: (num_layers, batch_size, hidden_size).
                         Defaults to zero if None.
                         
        Returns:
            torch.Tensor: L2-normalized embeddings. Shape: (batch_size, embedding_size).
        """
        # Pass input through LSTM layers
        # out: (batch_size, seq_len, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size) - hidden state of last time step
        # cell: (num_layers, batch_size, hidden_size) - cell state of last time step
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # Use the final hidden state of the last LSTM layer as the utterance representation
        # hidden[-1] shape: (batch_size, hidden_size)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        # L2-normalize the embeddings (crucial for cosine similarity-based loss)
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)
        
        return embeds
    
    def similarity_matrix(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Computes the similarity matrix for the GE2E loss.
        
        Calculates the cosine similarity between each embedding and the centroids of all
        speakers (inclusive and exclusive of the embedding itself).
        
        Args:
            embeds: Batch of embeddings. Shape: (speakers_per_batch, utterances_per_speaker, embedding_size).
            
        Returns:
            torch.Tensor: Similarity matrix. Shape: (speakers_per_batch, utterances_per_speaker, speakers_per_batch).
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Inclusive centroids: Mean embedding for each speaker including the current utterance
        # Shape: (speakers_per_batch, 1, embedding_size)
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        # Exclusive centroids: Mean embedding for each speaker excluding the current utterance
        # (Used for the positive class to avoid trivial solution)
        # Shape: (speakers_per_batch, utterances_per_speaker, embedding_size)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        # Similarity Matrix Construction
        # We calculate the cosine similarity (dot product of normalized vectors)
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(self.loss_device)
        
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=int)
        
        for j in range(speakers_per_batch):
            # For different speakers (j != k), use inclusive centroid (standard logic)
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            
            # For the same speaker (j == k), use exclusive centroid (avoid self-reinforcement)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        # Scale and shift signals (learnable w, b)
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix
    
    def loss(self, embeds: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Computes the Generalized End-to-End (GE2E) loss.
        
        Args:
            embeds: Embeddings reshaped as (speakers, utterances, embedding_dim).
            
        Returns:
            Tuple[torch.Tensor, float]: (Cross entropy loss value, Equal Error Rate).
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Calculate Similarity Matrix
        sim_matrix = self.similarity_matrix(embeds)
        
        # Reshape for CrossEntropyLoss
        # Input: (batch_size, number_of_classes) -> (speakers * utterances, speakers)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        
        # Ground Truth Labels: Each utterance belongs to its speaker index
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        
        # Compute Loss
        loss = self.loss_fn(sim_matrix, target)
        
        # Calculate Equal Error Rate (EER) for monitoring (detached from graph)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # Calculate EER using ROC curve
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer