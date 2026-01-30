# ==================================================================================================
# DEEPFAKE AUDIO - synthesizer/models/tacotron.py (Neural Acoustic Architecture)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements the Tacotron-2 architecture, a neural sequence-to-sequence 
# model for speech synthesis. It features a CBHG-based encoder for robust text 
# representation and a location-sensitive attention-based decoder for generating 
# high-resolution Mel-Spectrograms. The implementation is optimized for 
# multispeaker conditioning via identity vector injection.
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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Union

class HighwayNetwork(nn.Module):
    """Gated Residual Connection: Facilitates deep gradient flow via highway gates."""
    def __init__(self, size):
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)
        self.W1.bias.data.fill_(0.)

    def forward(self, x):
        x1 = self.W1(x)
        x2 = self.W2(x)
        g = torch.sigmoid(x2)
        y = g * F.relu(x1) + (1. - g) * x
        return y

class Encoder(nn.Module):
    """
    Linguistic Encoder:
    Transforms text sequences into abstract phonological representations. 
    Includes PreNet for dimensionality reduction and CBHG for feature extraction.
    """
    def __init__(self, embed_dims, num_chars, encoder_dims, K, num_highways, dropout):
        super().__init__()
        prenet_dims = (encoder_dims, encoder_dims)
        cbhg_channels = encoder_dims
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.pre_net = PreNet(embed_dims, fc1_dims=prenet_dims[0], fc2_dims=prenet_dims[1],
                              dropout=dropout)
        self.cbhg = CBHG(K=K, in_channels=cbhg_channels, channels=cbhg_channels,
                         proj_channels=[cbhg_channels, cbhg_channels],
                         num_highways=num_highways)

    def forward(self, x, speaker_embedding=None):
        x = self.embedding(x)
        x = self.pre_net(x)
        x.transpose_(1, 2)
        x = self.cbhg(x)
        if speaker_embedding is not None:
            x = self.add_speaker_embedding(x, speaker_embedding)
        return x

    def add_speaker_embedding(self, x, speaker_embedding):
        """SV2TTS: Identity Injection - Tile and concat speaker embedding to encoder output."""
        batch_size = x.size()[0]
        num_chars = x.size()[1]

        idx = 0 if speaker_embedding.dim() == 1 else 1
        speaker_embedding_size = speaker_embedding.size()[idx]
        e = speaker_embedding.repeat_interleave(num_chars, dim=idx)
        e = e.reshape(batch_size, speaker_embedding_size, num_chars).transpose(1, 2)

        return torch.cat((x, e), 2)

class BatchNormConv(nn.Module):
    """Standardized 1D Convolution with Batch Normalization."""
    def __init__(self, in_channels, out_channels, kernel, relu=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)

class CBHG(nn.Module):
    """
    Neural Feature Extractor:
    1D Convolution Bank + Highway Network + Bidirectional GRU.
    Inspired by Google's Tacotron architecture for robust feature learning.
    """
    def __init__(self, K, in_channels, channels, proj_channels, num_highways):
        super().__init__()
        self._to_flatten = []
        self.bank_kernels = [i for i in range(1, K + 1)]
        self.conv1d_bank = nn.ModuleList([BatchNormConv(in_channels, channels, k) for k in self.bank_kernels])
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.conv_project1 = BatchNormConv(len(self.bank_kernels) * channels, proj_channels[0], 3)
        self.conv_project2 = BatchNormConv(proj_channels[0], proj_channels[1], 3, relu=False)

        self.highway_mismatch = proj_channels[-1] != channels
        if self.highway_mismatch:
            self.pre_highway = nn.Linear(proj_channels[-1], channels, bias=False)

        self.highways = nn.ModuleList([HighwayNetwork(channels) for _ in range(num_highways)])
        self.rnn = nn.GRU(channels, channels // 2, batch_first=True, bidirectional=True)
        self._to_flatten.append(self.rnn)
        self._flatten_parameters()

    def forward(self, x):
        self._flatten_parameters()
        residual = x
        seq_len = x.size(-1)
        
        # Convolve through bank and pool
        conv_bank = torch.cat([conv(x)[:, :, :seq_len] for conv in self.conv1d_bank], dim=1)
        x = self.maxpool(conv_bank)[:, :, :seq_len]

        # Projections & Residual Connection
        x = self.conv_project1(x)
        x = self.conv_project2(x)
        x = x + residual

        # High-level feature gating
        x = x.transpose(1, 2)
        if self.highway_mismatch: x = self.pre_highway(x)
        for h in self.highways: x = h(x)

        # Bidirectional Temporal Modeling
        x, _ = self.rnn(x)
        return x

    def _flatten_parameters(self):
        [m.flatten_parameters() for m in self._to_flatten]

class PreNet(nn.Module):
    """Neural Bottleneck: Initial projection for input sequences."""
    def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.p = dropout

    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), self.p, training=True)
        x = F.dropout(F.relu(self.fc2(x)), self.p, training=True)
        return x

class Attention(nn.Module):
    """Dot-Product Attention mechanism."""
    def __init__(self, attn_dims):
        super().__init__()
        self.W = nn.Linear(attn_dims, attn_dims, bias=False)
        self.v = nn.Linear(attn_dims, 1, bias=False)

    def forward(self, encoder_seq_proj, query, t):
        query_proj = self.W(query).unsqueeze(1)
        u = self.v(torch.tanh(encoder_seq_proj + query_proj))
        scores = F.softmax(u, dim=1)
        return scores.transpose(1, 2)

class LSA(nn.Module):
    """Location Sensitive Attention: Incorporates historical attention for hybrid alignment."""
    def __init__(self, attn_dim, kernel_size=31, filters=32):
        super().__init__()
        self.conv = nn.Conv1d(1, filters, padding=(kernel_size - 1) // 2, kernel_size=kernel_size, bias=True)
        self.L = nn.Linear(filters, attn_dim, bias=False)
        self.W = nn.Linear(attn_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.cumulative = None
        self.attention = None

    def init_attention(self, encoder_seq_proj):
        device = next(self.parameters()).device
        b, t, c = encoder_seq_proj.size()
        self.cumulative = torch.zeros(b, t, device=device)
        self.attention = torch.zeros(b, t, device=device)

    def forward(self, encoder_seq_proj, query, t, chars):
        if t == 0: self.init_attention(encoder_seq_proj)
        processed_query = self.W(query).unsqueeze(1)
        location = self.cumulative.unsqueeze(1)
        processed_loc = self.L(self.conv(location).transpose(1, 2))
        u = self.v(torch.tanh(processed_query + encoder_seq_proj + processed_loc)).squeeze(-1)

        # Zero-masking for padding tokens
        u = u * (chars != 0).float()
        scores = F.softmax(u, dim=1)
        self.attention = scores
        self.cumulative = self.cumulative + self.attention
        return scores.unsqueeze(-1).transpose(1, 2)

class Decoder(nn.Module):
    """
    Autoregressive Decoder:
    Synthesizes Mel-Spectrogram frames sequentially. 
    Includes Zoneout-regularized LSTMs for robust speech patterns.
    """
    max_r = 20
    def __init__(self, n_mels, encoder_dims, decoder_dims, lstm_dims,
                 dropout, speaker_embedding_size):
        super().__init__()
        self.register_buffer("r", torch.tensor(1, dtype=torch.int))
        self.n_mels = n_mels
        prenet_dims = (decoder_dims * 2, decoder_dims * 2)
        self.prenet = PreNet(n_mels, fc1_dims=prenet_dims[0], fc2_dims=prenet_dims[1],
                             dropout=dropout)
        self.attn_net = LSA(decoder_dims)
        self.attn_rnn = nn.GRUCell(encoder_dims + prenet_dims[1] + speaker_embedding_size, decoder_dims)
        self.rnn_input = nn.Linear(encoder_dims + decoder_dims + speaker_embedding_size, lstm_dims)
        self.res_rnn1 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.res_rnn2 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.mel_proj = nn.Linear(lstm_dims, n_mels * self.max_r, bias=False)
        self.stop_proj = nn.Linear(encoder_dims + speaker_embedding_size + lstm_dims, 1)

    def zoneout(self, prev, current, p=0.1):
        """Neural Zoneout: Stochastic hidden state persistence."""
        device = next(self.parameters()).device
        mask = torch.zeros(prev.size(), device=device).bernoulli_(p)
        return prev * mask + current * (1 - mask)

    def forward(self, encoder_seq, encoder_seq_proj, prenet_in,
                hidden_states, cell_states, context_vec, t, chars):
        batch_size = encoder_seq.size(0)
        attn_hidden, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states

        # Neural feedback loop
        prenet_out = self.prenet(prenet_in)
        attn_hidden = self.attn_rnn(torch.cat([context_vec, prenet_out], dim=-1).squeeze(1), attn_hidden)
        scores = self.attn_net(encoder_seq_proj, attn_hidden, t, chars)
        context_vec = (scores @ encoder_seq).squeeze(1)

        # Residual RNN layers
        x = self.rnn_input(torch.cat([context_vec, attn_hidden], dim=1))
        
        rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell))
        rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next) if self.training else rnn1_hidden_next
        x = x + rnn1_hidden

        rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell))
        rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next) if self.training else rnn2_hidden_next
        x = x + rnn2_hidden

        # Multi-frame projection
        mels = self.mel_proj(x).view(batch_size, self.n_mels, self.max_r)[:, :, :self.r]
        
        # Stop token prediction
        stop_tokens = torch.sigmoid(self.stop_proj(torch.cat((x, context_vec), dim=1)))

        return mels, scores, (attn_hidden, rnn1_hidden, rnn2_hidden), (rnn1_cell, rnn2_cell), context_vec, stop_tokens

class Tacotron(nn.Module):
    """
    The Tacotron Model:
    Unified orchestrator for end-to-end multispeaker TTS.
    """
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, 
                 fft_bins, postnet_dims, encoder_K, lstm_dims, postnet_K, num_highways,
                 dropout, stop_threshold, speaker_embedding_size):
        super().__init__()
        self.n_mels, self.lstm_dims, self.encoder_dims = n_mels, lstm_dims, encoder_dims
        self.decoder_dims, self.speaker_embedding_size = decoder_dims, speaker_embedding_size
        
        # Architecture sub-layers
        self.encoder = Encoder(embed_dims, num_chars, encoder_dims, encoder_K, num_highways, dropout)
        self.encoder_proj = nn.Linear(encoder_dims + speaker_embedding_size, decoder_dims, bias=False)
        self.decoder = Decoder(n_mels, encoder_dims, decoder_dims, lstm_dims, dropout, speaker_embedding_size)
        self.postnet = CBHG(postnet_K, n_mels, postnet_dims, [postnet_dims, fft_bins], num_highways)
        self.post_proj = nn.Linear(postnet_dims, fft_bins, bias=False)

        self.init_model()
        self.num_params()
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
        self.register_buffer("stop_threshold", torch.tensor(stop_threshold, dtype=torch.float32))

    @property
    def r(self): return self.decoder.r.item()
    @r.setter
    def r(self, value): self.decoder.r = self.decoder.r.new_tensor(value, requires_grad=False)

    def forward(self, x, m, speaker_embedding):
        """Neural Training Pass."""
        device = next(self.parameters()).device
        self.step += 1
        batch_size, _, steps = m.size()

        # State Initialization
        hidden_states = (torch.zeros(batch_size, self.decoder_dims, device=device),
                         torch.zeros(batch_size, self.lstm_dims, device=device),
                         torch.zeros(batch_size, self.lstm_dims, device=device))
        cell_states = (torch.zeros(batch_size, self.lstm_dims, device=device),
                        torch.zeros(batch_size, self.lstm_dims, device=device))
        
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)
        context_vec = torch.zeros(batch_size, self.encoder_dims + self.speaker_embedding_size, device=device)

        # Neural Encoding
        encoder_seq = self.encoder(x, speaker_embedding)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        mel_outputs, attn_scores, stop_outputs = [], [], []

        # Autoregressive Decoding Loop
        for t in range(0, steps, self.r):
            prenet_in = m[:, :, t - 1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec, stop_tokens = \
                self.decoder(encoder_seq, encoder_seq_proj, prenet_in, hidden_states, cell_states, context_vec, t, x)
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)
            stop_outputs.extend([stop_tokens] * self.r)

        mel_outputs = torch.cat(mel_outputs, dim=2)
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out).transpose(1, 2)

        return mel_outputs, linear, torch.cat(attn_scores, 1), torch.cat(stop_outputs, 1)

    def generate(self, x, speaker_embedding=None, steps=2000):
        """Neural Inference Pass: Synthesizes speech from text and identity."""
        self.eval()
        device = next(self.parameters()).device
        batch_size, _ = x.size()

        # State Initialization for inference
        hidden_states = (torch.zeros(batch_size, self.decoder_dims, device=device),
                         torch.zeros(batch_size, self.lstm_dims, device=device),
                         torch.zeros(batch_size, self.lstm_dims, device=device))
        cell_states = (torch.zeros(batch_size, self.lstm_dims, device=device),
                        torch.zeros(batch_size, self.lstm_dims, device=device))
        
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)
        context_vec = torch.zeros(batch_size, self.encoder_dims + self.speaker_embedding_size, device=device)

        encoder_seq = self.encoder(x, speaker_embedding)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        mel_outputs, attn_scores = [], []

        for t in range(0, steps, self.r):
            prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec, stop_tokens = \
                self.decoder(encoder_seq, encoder_seq_proj, prenet_in, hidden_states, cell_states, context_vec, t, x)
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)
            if (stop_tokens > 0.5).all() and t > 10: break

        mel_outputs = torch.cat(mel_outputs, dim=2)
        linear = self.post_proj(self.postnet(mel_outputs)).transpose(1, 2)
        self.train()
        return mel_outputs, linear, torch.cat(attn_scores, 1)

    def init_model(self):
        """Xavier Weight Initialization."""
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def get_step(self): return self.step.data.item()

    def load(self, path, optimizer=None):
        """Neural Restoration: Loads model and optimizer states."""
        device = next(self.parameters()).device
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
        self.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

    def save(self, path, optimizer=None):
        """Neural Persistence: Saves model checkpoints."""
        state = {"model_state": self.state_dict()}
        if optimizer: state["optimizer_state"] = optimizer.state_dict()
        torch.save(state, str(path))

    def num_params(self, print_out=True):
        """Analytical evaluation of model complexity."""
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        count = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out: print("Trainable Parameters: %.3fM" % count)
        return count
