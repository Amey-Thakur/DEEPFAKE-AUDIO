"""
Deepfake Audio - Encoder Training Loop
--------------------------------------
This module handles the training process for the Speaker Encoder.
It manages data loading, model optimization, logging, visualization,
and checkpoint management.

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

from pathlib import Path
import torch

from encoder.visualizations import Visualizations
from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.params_model import *
from encoder.model import SpeakerEncoder
from utils.profiler import Profiler


def sync(device: torch.device) -> None:
    """
    Forces synchronization between CPU and GPU to ensure accurate profiling time measurements.
    
    Args:
        device: The PyTorch device to synchronize.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    else:
        torch.cpu.synchronize(device)
    

def train(run_id: str, clean_data_root: Path, models_dir: Path, umap_every: int, save_every: int,
          backup_every: int, vis_every: int, force_restart: bool, visdom_server: str,
          no_visdom: bool) -> None:
    """
    Main training function for the Speaker Encoder.
    
    Args:
        run_id: Unique identifier for the training run.
        clean_data_root: Path to the preprocessed training data.
        models_dir: Directory to save model checkpoints and backups.
        umap_every: Interval (steps) for generating UMAP projections.
        save_every: Interval (steps) for saving the current model state.
        backup_every: Interval (steps) for creating historical backups.
        vis_every: Interval (steps) for updating visualization plots.
        force_restart: If True, ignores existing checkpoints and starts from scratch.
        visdom_server: URL for the Visdom visualization server.
        no_visdom: If True, disables Visdom logging.
    """
    # -------------------------------------------------------------------------
    # Data Preparation
    # -------------------------------------------------------------------------
    dataset = SpeakerVerificationDataset(clean_data_root)
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=8,
    )
    
    # -------------------------------------------------------------------------
    # Computing Infrastructure
    # -------------------------------------------------------------------------
    # Setup device for forward pass (GPU preferred) and loss computation (CPU often preferred for GE2E)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Loss computation is explicitly on CPU to avoid potential CUDA bottlenecks with the specific
    # operations in GE2E or to save GPU memory.
    loss_device = torch.device("cpu") 
    
    # -------------------------------------------------------------------------
    # Model & Optimizer Initialization
    # -------------------------------------------------------------------------
    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    init_step = 1
    
    # Configure Paths
    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "_backups")

    # Load Checkpoint logic
    if not force_restart:
        if state_fpath.exists():
            print(f"Found existing model \"{run_id}\", loading it and resuming training.")
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print(f"No model \"{run_id}\" found, starting training from scratch.")
    else:
        print("Starting the training from scratch.")
        
    model.train()
    
    # -------------------------------------------------------------------------
    # Visualization Setup
    # -------------------------------------------------------------------------
    vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    vis.log_dataset(dataset)
    vis.log_params()
    
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    vis.log_implementation({"Device": device_name})
    
    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    profiler = Profiler(summarize_every=10, disabled=False)
    
    for step, speaker_batch in enumerate(loader, init_step):
        profiler.tick("Blocking, waiting for batch (threaded)")
        
        # 1. Forward Pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)
        profiler.tick(f"Data to {device}")
        
        embeds = model(inputs)
        sync(device)
        profiler.tick("Forward pass")
        
        # 2. Loss Computation
        embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)
        sync(loss_device)
        profiler.tick("Loss")

        # 3. Backward Pass & Optimization
        model.zero_grad()
        loss.backward()
        profiler.tick("Backward pass")
        
        model.do_gradient_ops()
        optimizer.step()
        profiler.tick("Parameter update")
        
        # 4. Visualization Updates
        vis.update(loss.item(), eer, step)
        
        # 5. UMAP Projections (Computational intensive, done less frequently)
        if umap_every != 0 and step % umap_every == 0:
            print(f"Drawing and saving projections (step {step})")
            backup_dir.mkdir(exist_ok=True)
            projection_fpath = backup_dir.joinpath(f"{run_id}_umap_{step:06d}.png")
            embeds = embeds.detach().cpu().numpy()
            vis.draw_projections(embeds, utterances_per_speaker, step, projection_fpath)
            vis.save()

        # 6. Checkpointing
        if save_every != 0 and step % save_every == 0:
            print(f"Saving the model (step {step})")
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)
            
        # 7. Backups
        if backup_every != 0 and step % backup_every == 0:
            print(f"Making a backup (step {step})")
            backup_dir.mkdir(exist_ok=True)
            backup_fpath = backup_dir.joinpath(f"{run_id}_bak_{step:06d}.pt")
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, backup_fpath)
            
        profiler.tick("Extras (visualizations, saving)")
