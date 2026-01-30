# ==================================================================================================
# DEEPFAKE AUDIO - encoder/visualizations.py (Neural Telemetry & Projections)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements the visual diagnostic engine for training monitoring. 
# It utilizes Visdom for real-time loss/EER plotting and UMAP (Uniform Manifold 
# Approximation and Projection) to visualize the high-dimensional speaker 
# identity space, allowing researchers to observe the clustering of d-vectors.
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

from datetime import datetime
from time import perf_counter as timer
import numpy as np
import umap
import visdom

# --- PROJECT CORE MODULES ---
from encoder.data_objects.speaker_verification_dataset import SpeakerVerificationDataset

# --- AESTHETIC CONFIGURATION (Categorical Color Map) ---
colormap = np.array([
    [76, 255, 0],   [0, 127, 70],   [255, 0, 0],   [255, 217, 38],
    [0, 135, 255],  [165, 0, 165],  [255, 167, 255], [0, 255, 255],
    [255, 96, 38],  [142, 76, 0],   [33, 0, 127],  [0, 0, 0],
    [183, 183, 183],
], dtype=float) / 255

class Visualizations:
    """
    Experimental Dashboard:
    Provides a real-time window into the model's convergence and identity manifold.
    """
    def __init__(self, env_name=None, update_every=10, server="http://localhost", disabled=False):
        # Neural Tracking State
        self.last_update_timestamp = timer()
        self.update_every = update_every
        self.step_times, self.losses, self.eers = [], [], []
        
        # Operational Mode
        self.disabled = disabled
        if self.disabled: return

        # Temporal Versioning of Environment
        now = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = now if env_name is None else "%s (%s)" % (env_name, now)

        # Visdom Server Handshake
        try:
            self.vis = visdom.Visdom(server, env=self.env_name, raise_exceptions=True)
        except ConnectionError:
            raise Exception("⚠️ Technical Blocker: Visdom server not detected. Start it via 'visdom' CLI.")

        # Dashboard Window Handles
        self.loss_win = self.eer_win = self.implementation_win = self.projection_win = None
        self.implementation_string = ""

    def log_params(self):
        """Archives model and data hyperparameters in the Visdom environment."""
        if self.disabled: return
        from encoder import params_data, params_model
        
        param_string = "<b>🧬 Neural Architecture Parameters</b>:<br>"
        for p in (n for n in dir(params_model) if not n.startswith("__")):
            param_string += "\t%s: %s<br>" % (p, getattr(params_model, p))
            
        param_string += "<br><b>🔊 Acoustic Signal Parameters</b>:<br>"
        for p in (n for n in dir(params_data) if not n.startswith("__")):
            param_string += "\t%s: %s<br>" % (p, getattr(params_data, p))
            
        self.vis.text(param_string, opts={"title": "Hyperparameters (Static Snapshot)"})

    def log_dataset(self, dataset: SpeakerVerificationDataset):
        """Documents the corpus profile being utilized for training."""
        if self.disabled: return
        ds_string = "<b>📚 Corpus Overview</b>: %d Registered Speakers<br>" % len(dataset.speakers)
        ds_string += dataset.get_logs().replace("\n", "<br>")
        self.vis.text(ds_string, opts={"title": "Dataset Manifest"})

    def log_implementation(self, params):
        """Metadata regarding the execution environment (e.g., CUDA device)."""
        if self.disabled: return
        impl_str = "".join("<b>%s</b>: %s<br>" % (p, v) for p, v in params.items())
        self.implementation_string = impl_str
        self.implementation_win = self.vis.text(impl_str, opts={"title": "Training Environment"})

    def update(self, loss, eer, step):
        """Calculates and plots rolling averages of optimization metrics."""
        now = timer()
        self.step_times.append(1000 * (now - self.last_update_timestamp))
        self.last_update_timestamp = now
        self.losses.append(loss)
        self.eers.append(eer)
        print(".", end="") # Silent progress heartbeat

        if step % self.update_every != 0: return # Stratified reporting
        
        t_str = "Step Latency: mean %dms, std %dms" % (int(np.mean(self.step_times)), int(np.std(self.step_times)))
        print("\n📈 Step %d | Loss: %.4f | EER: %.4f | %s" % (step, np.mean(self.losses), np.mean(self.eers), t_str))
        
        if not self.disabled:
            # Loss Convergence Plot
            self.loss_win = self.vis.line([np.mean(self.losses)], [step], win=self.loss_win,
                                           update="append" if self.loss_win else None,
                                           opts=dict(legend=["Avg. GE2E Loss"], xlabel="Step", ylabel="Loss", title="Training Loss"))
            # Error Rate Plot
            self.eer_win = self.vis.line([np.mean(self.eers)], [step], win=self.eer_win,
                                          update="append" if self.eer_win else None,
                                          opts=dict(legend=["Avg. EER"], xlabel="Step", ylabel="EER", title="Equal Error Rate"))
            
            if self.implementation_win:
                self.vis.text(self.implementation_string + ("<b>%s</b>" % t_str),
                              win=self.implementation_win, opts={"title": "Training Environment"})

        # Memory Cleanup for the next window
        self.losses.clear(); self.eers.clear(); self.step_times.clear()

    def draw_projections(self, embeds, utterances_per_speaker, step, out_fpath=None, max_speakers=10):
        """
        Manifold Mapping:
        Projects high-dimensional d-vectors onto a 2D plane for visual cluster analysis.
        """
        import matplotlib.pyplot as plt
        max_speakers = min(max_speakers, len(colormap))
        embeds = embeds[:max_speakers * utterances_per_speaker]

        # Categorical Color Orchestration
        n_speakers = len(embeds) // utterances_per_speaker
        colors = [colormap[i] for i in np.repeat(np.arange(n_speakers), utterances_per_speaker)]

        # UMAP Non-Linear Dimensionality Reduction
        reducer = umap.UMAP()
        projected = reducer.fit_transform(embeds)
        
        plt.scatter(projected[:, 0], projected[:, 1], c=colors)
        plt.gca().set_aspect("equal", "datalim")
        plt.title("🌌 Identity Manifold (UMAP) - Step %d" % step)
        
        if not self.disabled:
            self.projection_win = self.vis.matplot(plt, win=self.projection_win)
        if out_fpath:
            plt.savefig(out_fpath)
        plt.clf()

    def save(self):
        """Persists the Visdom environment state to the server."""
        if not self.disabled:
            self.vis.save([self.env_name])
