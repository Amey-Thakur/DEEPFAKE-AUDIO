# ==================================================================================================
# DEEPFAKE AUDIO - vocoder/models/fatchord_version.py (WaveRNN Architecture)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements the "Fatchord" variant of WaveRNN. It utilizes 
# upsampling networks, residual Mel-ResNets, and a dual-GRU recurrent core 
# to synthesize waveforms from Mel-Spectrograms. It supports both RAW 
# (softmax) and MOL (Logistic Mixture) output modes.
#
# 👤 AUTHORS
# - Amey Thakur (https://github.com/Amey-Thakur)
# - Mega Satish (https://github.com/msatmod)
#
# 🤝🏻 CREDITS
# Original Real-Time Voice Cloning methodology by CorentinJ
# Repository: https://github.com/CorentinJ/Real-Time-Voice-Cloning
# Fatchord WaveRNN original implementation reference
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
import torch.nn as nn
import torch.nn.functional as F
from vocoder.distribution import sample_from_discretized_mix_logistic
from vocoder.display import *
from vocoder.audio import *

class ResBlock(nn.Module):
    """Neural Backbone: Implements a 1D residual block with batch normalization."""
    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        residual = x
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))
        return x + residual

class MelResNet(nn.Module):
    """
    Feature Refinement:
    Applies a series of residual blocks to refine Mel-Spectrogram features.
    """
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims, pad):
        super().__init__()
        k_size = pad * 2 + 1
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=k_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList([ResBlock(compute_dims) for _ in range(res_blocks)])
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.batch_norm(self.conv_in(x)))
        for f in self.layers: x = f(x)
        return self.conv_out(x)

class Stretch2d(nn.Module):
    """Signal Expansion: Nearest-neighbor upsampling for 2D tensors."""
    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)

class UpsampleNetwork(nn.Module):
    """
    Temporal Pyramid:
    Upsamples Mel-Spectrogram features to match the audio sampling resolution.
    """
    def __init__(self, feat_dims, upsample_scales, compute_dims, res_blocks, res_out_dims, pad):
        super().__init__()
        total_scale = np.cumprod(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / k_size[1])
            self.up_layers.extend([stretch, conv])

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux).squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers: m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)

class WaveRNN(nn.Module):
    """
    Neural Orchestration:
    Primary class for the WaveRNN vocoder, managing upsampling and recurrent generation.
    """
    def __init__(self, rnn_dims, fc_dims, bits, pad, upsample_factors,
                 feat_dims, compute_dims, res_out_dims, res_blocks,
                 hop_length, sample_rate, mode='RAW'):
        super().__init__()
        self.mode = mode
        self.pad = pad
        self.n_classes = 2 ** bits if mode == 'RAW' else 30
        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims, res_blocks, res_out_dims, pad)
        self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)
        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)

        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
        self.num_params()

    def forward(self, x, mels):
        """Neural Training Step: Process a batch of audio sequences."""
        self.step += 1
        bsize = x.size(0)
        device = x.device
        h1 = torch.zeros(1, bsize, self.rnn_dims).to(device)
        h2 = torch.zeros(1, bsize, self.rnn_dims).to(device)
        
        mels, aux = self.upsample(mels)

        aux_idx = [self.aux_dims * i for i in range(5)]
        a1, a2, a3, a4 = (aux[:, :, aux_idx[i]:aux_idx[i+1]] for i in range(4))

        x = self.I(torch.cat([x.unsqueeze(-1), mels, a1], dim=2))
        res = x
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x, _ = self.rnn2(torch.cat([x, a2], dim=2), h2)

        x = F.relu(self.fc1(torch.cat([x+res, a3], dim=2)))
        x = F.relu(self.fc2(torch.cat([x, a4], dim=2)))
        return self.fc3(x)

    def generate(self, mels, batched, target, overlap, mu_law, progress_callback=None):
        """Autoregressive Synthesis: Generates audio waveforms from Mel-Spectrograms."""
        mu_law = mu_law if self.mode == 'RAW' else False
        progress_callback = progress_callback or self.gen_display
        self.eval()
        
        start = time.time()
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        # Align to hop_length
        target = (target // self.hop_length) * self.hop_length
        overlap = (overlap // self.hop_length) * self.hop_length

        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            mels = mels.to(device)
            wave_len = (mels.size(-1) - 1) * self.hop_length
            
            # 1. Pad and Fold MELs (Zero-Copy)
            mel_step = (target + overlap) // self.hop_length
            mel_size = (target + 2 * overlap) // self.hop_length + 2 * self.pad
            mels_padded = self.pad_tensor(mels.transpose(1, 2), pad=self.pad, side='both')
            mels_folded = mels_padded.unfold(1, mel_size, mel_step).squeeze(0).transpose(1, 2)
            
            num_folds = mels_folded.size(0)
            mini_batch_size = 16
            all_outputs = []

            for b in range(0, num_folds, mini_batch_size):
                m_batch = mels_folded[b:b + mini_batch_size]
                cur_b_size = m_batch.size(0)
                
                # i. Upsample current mini-batch
                # upsample expects (B, Mel, T)
                m_batch, aux_batch = self.upsample(m_batch.transpose(1, 2))
                
                seq_len = m_batch.size(1)
                h1 = torch.zeros(cur_b_size, self.rnn_dims).to(device)
                h2 = torch.zeros(cur_b_size, self.rnn_dims).to(device)
                x = torch.zeros(cur_b_size, 1).to(device)
                
                d = self.aux_dims
                aux_split = [aux_batch[:, :, d * i:d * (i + 1)] for i in range(4)]
                
                batch_output = []
                for i in range(seq_len):
                    m_t = m_batch[:, i, :]
                    a1_t, a2_t, a3_t, a4_t = (a[:, i, :] for a in aux_split)

                    x = self.I(torch.cat([x, m_t, a1_t], dim=1))
                    h1 = rnn1(x, h1)

                    x = x + h1
                    h2 = rnn2(torch.cat([x, a2_t], dim=1), h2)

                    x = F.relu(self.fc1(torch.cat([x + h2, a3_t], dim=1)))
                    x = F.relu(self.fc2(torch.cat([x, a4_t], dim=1)))
                    logits = self.fc3(x)

                    if self.mode == 'MOL':
                        sample = sample_from_discretized_mix_logistic(logits.unsqueeze(0).transpose(1, 2))
                        batch_output.append(sample.view(-1))
                        x = sample.transpose(0, 1).to(device)
                    elif self.mode == 'RAW':
                        posterior = F.softmax(logits, dim=1)
                        distrib = torch.distributions.Categorical(posterior)
                        sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                        batch_output.append(sample)
                        x = sample.unsqueeze(-1)

                    if i % 100 == 0:
                        gen_rate = (i + 1) / (time.time() - start) * (b + cur_b_size) / 1000
                        progress_callback(b * seq_len + i, num_folds * seq_len, cur_b_size, gen_rate)
                
                all_outputs.append(torch.stack(batch_output).transpose(0, 1).cpu())
                # Explicitly clear memory
                del m_batch, aux_batch, h1, h2, x, aux_split, batch_output
            
        output = torch.cat(all_outputs).numpy().astype(np.float64)
        if batched: output = self.xfade_and_unfold(output, target, overlap)
        else: output = output[0]

        if mu_law: output = decode_mu_law(output, self.n_classes, False)
        if hp.apply_preemphasis: output = de_emphasis(output)

        # Dynamic Fade-out
        fade_out = np.linspace(1, 0, 20 * self.hop_length)
        output = output[:wave_len]
        if len(output) >= 20 * self.hop_length:
            output[-20 * self.hop_length:] *= fade_out
        self.train()
        return output

    def gen_display(self, i, seq_len, b_size, gen_rate):
        """Diagnostic Monitor: Updates console progress during generation."""
        pbar = progbar(i, seq_len)
        stream(f'| {pbar} {i*b_size}/{seq_len*b_size} | Batch Size: {b_size} | Gen Rate: {gen_rate:.1f}kHz | ')

    def get_gru_cell(self, gru):
        """Cell Extractor: Converts a GRU layer into a GRUCell for manual stepping."""
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def pad_tensor(self, x, pad, side='both'):
        """Asset Conditioning: Pads the temporal axis of a tensor."""
        b, t, c = x.size()
        total = t + (2 * pad if side == 'both' else pad)
        padded = torch.zeros(b, total, c).to(x.device)
        if side == 'before' or side == 'both': padded[:, pad:pad+t, :] = x
        elif side == 'after': padded[:, :t, :] = x
        return padded

    def fold_with_overlap(self, x, target, overlap):
        """Batch Optimizer: Folds a long sequence into overlapping batches for inference."""
        _, total_len, features = x.size()
        
        # Correctly calculate num_folds using ceil-like logic
        num_folds = (total_len - overlap) // (target + overlap)
        if (total_len - overlap) % (target + overlap) != 0:
            num_folds += 1
        
        # Pad the tensor to fit the full extent of the calculated folds
        expected_len = num_folds * (target + overlap) + overlap
        if total_len != expected_len:
            x = self.pad_tensor(x, expected_len - total_len, side='after')
        
        folded = torch.zeros(num_folds, target + 2 * overlap, features).to(x.device)
        for i in range(num_folds):
            start = i * (target + overlap)
            folded[i] = x[:, start:start + target + 2 * overlap, :]
        return folded

    def xfade_and_unfold(self, y, target, overlap):
        """Signal Reconstruction: Crossfades overlapping batches back into a 1D sequence."""
        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap
        
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.concatenate([np.zeros(silence_len), np.sqrt(0.5 * (1 + t))])
        fade_out = np.concatenate([np.sqrt(0.5 * (1 - t)), np.zeros(silence_len)])

        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out
        unfolded = np.zeros(total_len, dtype=np.float64)
        for i in range(num_folds):
            start = i * (target + overlap)
            unfolded[start:start + target + 2 * overlap] += y[i]
        return unfolded

    def get_step(self):
        """Metric Retrieval: Returns the current global training step."""
        return self.step.data.item()

    def checkpoint(self, model_dir, optimizer):
        """Persistence: Saves a model checkpoint with the current step count."""
        self.save(model_dir.joinpath("checkpoint_%dk_steps.pt" % (self.get_step() // 1000)), optimizer)

    def log(self, path, msg):
        """Diagnostic Logging: Appends messages to a text log file."""
        with open(path, 'a') as f: print(msg, file=f)

    def load(self, path, optimizer):
        """Restoration: Loads weights and optimizer state from a file."""
        checkpoint = torch.load(path)
        if "optimizer_state" in checkpoint:
            self.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else: self.load_state_dict(checkpoint)

    def save(self, path, optimizer):
        """Persistence: Saves the model and optimizer state."""
        torch.save({"model_state": self.state_dict(), "optimizer_state": optimizer.state_dict()}, path)

    def num_params(self, print_out=True):
        """Architectural Audit: Logs the total number of trainable parameters."""
        parameters = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.parameters())]) / 1_000_000
        if print_out: print('Audit: Trainable Parameters: %.3fM' % parameters)
