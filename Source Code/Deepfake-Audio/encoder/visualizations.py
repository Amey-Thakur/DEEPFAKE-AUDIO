"""
Deepfake Audio - Encoder Training Visualization
-----------------------------------------------
This module provides a real-time visualization interface using Visdom.
It plots training metrics (Loss, EER) and projects speaker embeddings
into 2D space using UMAP to visualize cluster separation.

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

from datetime import datetime
from time import perf_counter as timer
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import umap
import visdom

from encoder.data_objects.speaker_verification_dataset import SpeakerVerificationDataset

# Distinct colors for visualizing different speakers in embedding space
COLORMAP = np.array([
    [76, 255, 0],
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
], dtype=float) / 255 


class Visualizations:
    """
    Wraps the Visdom client to handle plotting and logging during training.
    """
    
    def __init__(self, env_name: str = None, update_every: int = 10, 
                 server: str = "http://localhost", disabled: bool = False):
        
        # Tracking metrics
        self.last_update_timestamp = timer()
        self.update_every = update_every
        self.step_times = []
        self.losses = []
        self.eers = []
        
        self.disabled = disabled
        if self.disabled:
             return 

        print(f"Updating the visualizations every {update_every} steps.")
        
        # Set the environment name
        now = datetime.now().strftime("%d-%m %Hh%M")
        if env_name is None:
            self.env_name = now
        else:
            self.env_name = f"{env_name} ({now})"
        
        # Connect to Visdom server
        try:
            self.vis = visdom.Visdom(server, env=self.env_name, raise_exceptions=True)
        except ConnectionError:
            raise ConnectionError(
                "No visdom server detected. Run 'visdom' in your CLI to start it."
            )
        
        # Window handles
        self.loss_win = None
        self.eer_win = None
        self.implementation_win = None
        self.projection_win = None
        self.implementation_string = ""
        
    def log_params(self):
        """Display model and data hyperparameters in the Visdom UI."""
        if self.disabled: return

        from encoder import params_data
        from encoder import params_model
        
        param_string = "<b>Model parameters</b>:<br>"
        for param_name in (p for p in dir(params_model) if not p.startswith("__")):
            value = getattr(params_model, param_name)
            param_string += f"\t{param_name}: {value}<br>"
            
        param_string += "<b>Data parameters</b>:<br>"
        for param_name in (p for p in dir(params_data) if not p.startswith("__")):
            value = getattr(params_data, param_name)
            param_string += f"\t{param_name}: {value}<br>"
            
        self.vis.text(param_string, opts={"title": "Parameters"})
        
    def log_dataset(self, dataset: SpeakerVerificationDataset):
        """Display dataset statistics in the Visdom UI."""
        if self.disabled: return
        
        dataset_string = f"<b>Speakers</b>: {len(dataset.speakers)}\n"
        dataset_string += "\n" + dataset.get_logs()
        dataset_string = dataset_string.replace("\n", "<br>")
        self.vis.text(dataset_string, opts={"title": "Dataset"})
        
    def log_implementation(self, params: Dict[str, Any]):
        """Display implementation details (like hardware) in the Visdom UI."""
        if self.disabled: return
        
        implementation_string = ""
        for param, value in params.items():
            implementation_string += f"<b>{param}</b>: {value}\n"
        
        implementation_string = implementation_string.replace("\n", "<br>")
        self.implementation_string = implementation_string
        self.implementation_win = self.vis.text(
            implementation_string, 
            opts={"title": "Training implementation"}
        )

    def update(self, loss: float, eer: float, step: int):
        """Updates internal metrics and refreshes plots if the interval is reached."""
        
        # Update tracking data
        now = timer()
        self.step_times.append(1000 * (now - self.last_update_timestamp))
        self.last_update_timestamp = now
        self.losses.append(loss)
        self.eers.append(eer)
        
        print(".", end="")
        
        # Update plots only every <update_every> steps
        if step % self.update_every != 0:
            return
            
        time_string = f"Step time:  mean: {int(np.mean(self.step_times)):5d}ms  std: {int(np.std(self.step_times)):5d}ms"
        print(f"\nStep {step:6d}   Loss: {np.mean(self.losses):.4f}   EER: {np.mean(self.eers):.4f}   {time_string}")
        
        if not self.disabled:
             # Plot Loss
            self.loss_win = self.vis.line(
                [np.mean(self.losses)],
                [step],
                win=self.loss_win,
                update="append" if self.loss_win else None,
                opts=dict(
                    legend=["Avg. loss"],
                    xlabel="Step",
                    ylabel="Loss",
                    title="Loss",
                )
            )
            # Plot EER
            self.eer_win = self.vis.line(
                [np.mean(self.eers)],
                [step],
                win=self.eer_win,
                update="append" if self.eer_win else None,
                opts=dict(
                    legend=["Avg. EER"],
                    xlabel="Step",
                    ylabel="EER",
                    title="Equal error rate"
                )
            )
            
            # Update status text
            if self.implementation_win is not None:
                self.vis.text(
                    self.implementation_string + f"<b>{time_string}</b>", 
                    win=self.implementation_win,
                    opts={"title": "Training implementation"},
                )

        # Reset accumulators
        self.losses.clear()
        self.eers.clear()
        self.step_times.clear()
        
    def draw_projections(self, embeds: np.ndarray, utterances_per_speaker: int, 
                         step: int, out_fpath: Optional[str] = None, max_speakers: int = 10):
        """
        Projects high-dimensional embeddings to 2D using UMAP and plots them.
        This provides a visual check of how well the model separates speakers.
        """
        max_speakers = min(max_speakers, len(COLORMAP))
        embeds = embeds[:max_speakers * utterances_per_speaker]
        
        n_speakers = len(embeds) // utterances_per_speaker
        ground_truth = np.repeat(np.arange(n_speakers), utterances_per_speaker)
        colors = [COLORMAP[i] for i in ground_truth]
        
        # Dimensionality Reduction
        reducer = umap.UMAP()
        projected = reducer.fit_transform(embeds)
        
        # Plotting
        plt.figure(figsize=(10, 10))
        plt.scatter(projected[:, 0], projected[:, 1], c=colors)
        plt.gca().set_aspect("equal", "datalim")
        plt.title(f"UMAP projection (step {step})")
        
        if not self.disabled:
            self.projection_win = self.vis.matplot(plt, win=self.projection_win)
            
        if out_fpath is not None:
            plt.savefig(out_fpath)
            
        plt.clf()
        plt.close()
        
    def save(self):
        """Saves the Visdom environment state."""
        if not self.disabled:
            self.vis.save([self.env_name])
        