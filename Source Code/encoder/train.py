# ==================================================================================================
# DEEPFAKE AUDIO - encoder/train.py (Neural Identity Optimization Cycle)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module orchestrates the complete training cycle for the Speaker Encoder. 
# It manages the GE2E (Generalized End-to-End) loss computation, stochastic 
# gradient descent via Adam, and provides rich diagnostic telemetry through 
# Visdom and UMAP projections. It ensures that the model learns a robust 
# identity manifold for zero-shot speaker adaptation.
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

from pathlib import Path
import torch

# --- PROJECT CORE MODULES ---
from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.model import SpeakerEncoder
from encoder.params_model import *
from encoder.visualizations import Visualizations
from utils.profiler import Profiler

def sync(device: torch.device):
    """Ensures GPU operations are completed before profiling ticks."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def train(run_id: str, clean_data_root: Path, models_dir: Path, umap_every: int, save_every: int,
          backup_every: int, vis_every: int, force_restart: bool, visdom_server: str,
          no_visdom: bool):
    """
    Main Orchestrator:
    1. Dataset & DataLoader Initialization (Categorical Batching)
    2. Architecture Construction (LSTM Backbone)
    3. Checkpoint Resumption (Resilient Training)
    4. Optimization Loop (GE2E Loss + UMAP Telemetry)
    """
    # Categorical Data Pipeline
    dataset = SpeakerVerificationDataset(clean_data_root)
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=4,
    )

    # Hardware Orchestration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # GE2E Loss Calculation is often mathematically stable on CPU
    loss_device = torch.device("cpu")

    # Neural & Optimization Setup
    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    init_step = 1

    # Storage Architecture
    model_dir = models_dir / run_id
    model_dir.mkdir(exist_ok=True, parents=True)
    state_fpath = model_dir / "encoder.pt"

    # Checkpoint Management
    if not force_restart:
        if state_fpath.exists():
            print("🤝🏻 Resuming Training Session: Found existing model \"%s\"" % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("🚀 Initiating New Session: Model \"%s\" not found." % run_id)
    else:
        print("📁 Force Restart: Re-initializing weights from scratch.")
    model.train()

    # Telemetry System (Visdom)
    vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    vis.log_dataset(dataset)
    vis.log_params()
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    vis.log_implementation({"Device": device_name})

    # High-Performance Training Cycle
    profiler = Profiler(summarize_every=10, disabled=False)
    for step, speaker_batch in enumerate(loader, init_step):
        profiler.tick("Blocking - Queue Ingestion")

        # 1. Forward Pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)
        profiler.tick("H2D Transfer")
        
        embeds = model(inputs)
        sync(device)
        profiler.tick("LSTM Backbone Inference")
        
        # 2. Geometric Similarity & Loss
        embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)
        sync(loss_device)
        profiler.tick("GE2E Loss Computation")

        # 3. Stochastic Gradient Optimization
        model.zero_grad()
        loss.backward()
        profiler.tick("Backpropagation")
        
        model.do_gradient_ops() # Gradient Clipping & Scaling
        optimizer.step()
        profiler.tick("Parameter Update")

        # 4. Telemetry Update (Smoothing Curve)
        vis.update(loss.item(), eer, step)

        # 5. UMAP Projections (Manifold Visualization)
        if umap_every != 0 and step % umap_every == 0:
            print("\n🌌 Generating Identity Manifold Projection (step %d)" % step)
            projection_fpath = model_dir / f"umap_{step:06d}.png"
            embeds_npy = embeds.detach().cpu().numpy()
            vis.draw_projections(embeds_npy, utterances_per_speaker, step, projection_fpath)
            vis.save()

        # 6. Weight Persistence (Checkpointing)
        if save_every != 0 and step % save_every == 0:
            print("\n💾 Persisting Latest Weights (step %d)" % step)
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)

        # 7. Rollng Backup (Immutable Snapshots)
        if backup_every != 0 and step % backup_every == 0:
            print("\n📁 Creating Immutable Snapshot (step %d)" % step)
            backup_fpath = model_dir / f"encoder_{step:06d}.bak"
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, backup_fpath)

        profiler.tick("Housekeeping (Telemetry & Storage)")
