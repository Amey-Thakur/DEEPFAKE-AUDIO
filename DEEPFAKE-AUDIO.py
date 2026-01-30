# ==================================================================================================
# DEEPFAKE-AUDIO - DEEPFAKE-AUDIO.py (Neural Voice Cloning Research Script)
# ==================================================================================================
#
# 📝 DESCRIPTION
# This script is a comprehensive, production-ready Python implementation of the SV2TTS
# (Speaker Verification to Text-to-Speech) voice cloning pipeline. It is a direct
# conversion of the 'DEEPFAKE-AUDIO.ipynb' Jupyter notebook, designed to run in both
# local and cloud (Colab/Kaggle) environments.
#
# The pipeline enables cloning a voice from as little as 5 seconds of audio. It consists
# of three distinct neural networks:
#   1. Speaker Encoder: Extracts a fixed-dimensional embedding (fingerprint) from
#      the reference audio.
#   2. Synthesizer (Tacotron 2): Generates a Mel-Spectrogram from text, conditioned
#      on the speaker embedding.
#   3. Vocoder (WaveRNN): Converts the Mel-Spectrogram into a raw time-domain waveform
#      (audible speech).
#
# KEY FEATURES:
# - Cross-Platform: Runs seamlessly on Windows, Linux, Google Colab, and Kaggle.
# - Robust Fallbacks: Implements a multi-source data retrieval strategy (Local ->
#   Kaggle -> Hugging Face) to guarantee model availability.
# - Interactive Mode: Supports preset samples, local file uploads, and microphone
#   recording (Colab only).
# - Analysis & Visualization: Generates waveform comparisons, Mel-Spectrograms, and
#   speaker embedding heatmaps.
#
# 👤 AUTHORS
# - Amey Thakur
#   - GitHub: https://github.com/Amey-Thakur
#   - ORCID: https://orcid.org/0000-0001-5644-1575
#   - Google Scholar: https://scholar.google.ca/citations?user=0inooPgAAAAJ
# - Mega Satish
#   - GitHub: https://github.com/msatmod
#   - ORCID: https://orcid.org/0000-0002-1844-9557
#   - Google Scholar: https://scholar.google.ca/citations?user=7Ajrr6EAAAAJ
#
# 🤝🏻 CREDITS
# This project builds upon the foundational work of the Real-Time Voice Cloning project.
# Original Repository: https://github.com/CorentinJ/Real-Time-Voice-Cloning
# Pre-trained Models: https://huggingface.co/CorentinJ/SV2TTS
#
# 🔗 PROJECT LINKS
# Repository: https://github.com/Amey-Thakur/DEEPFAKE-AUDIO
# Live Demo: https://huggingface.co/spaces/ameythakur/Deepfake-Audio
# Video Demo: https://youtu.be/i3wnBcbHDbs
# Kaggle Dataset: https://www.kaggle.com/datasets/ameythakur20/deepfakeaudio
#
# 📜 LICENSE
# Released under the MIT License
# Release Date: 2021-02-06
# ==================================================================================================

"""
DEEPFAKE-AUDIO.py: A complete, standalone script for neural voice cloning.

This script provides a command-line interface for the SV2TTS voice cloning pipeline.
It can be run directly or imported as a module.

Usage:
    python DEEPFAKE-AUDIO.py [--preset <name>] [--input <path>] [--text <text>]

Example:
    python DEEPFAKE-AUDIO.py --preset "Steve Jobs.wav" --text "Hello, this is a cloned voice."
"""

# ==================================================================================================
# SECTION 1: IMPORTS & GLOBAL CONFIGURATION
# ==================================================================================================
# This section imports all necessary libraries and sets up global variables. It handles
# environment detection and defines paths for data and model discovery.

import os
import sys
import shutil
import glob
import argparse
from pathlib import Path

# --- Cross-Platform Compatibility ---
# Ensure the 'Source Code' directory is on Python's path for module imports.
# This is critical for accessing the encoder, synthesizer, and vocoder submodules.
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_CODE_PATH = SCRIPT_DIR / "Source Code"
if str(SOURCE_CODE_PATH) not in sys.path:
    sys.path.insert(0, str(SOURCE_CODE_PATH))

# --- Cloud Environment Detection ---
# These flags help adapt the script's behavior based on the runtime environment.
IS_COLAB = "google.colab" in sys.modules
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ


# ==================================================================================================
# SECTION 2: CLOUD ENVIRONMENT SETUP (COLAB/KAGGLE)
# ==================================================================================================
# This function is designed to run exclusively in cloud notebook environments.
# It handles repository cloning, dependency installation, and data retrieval with
# multiple fallback sources.

def setup_cloud_environment():
    """
    Orchestrates the cloud environment setup for Google Colab and Kaggle.

    This function performs the following steps:
    1. Clones the DEEPFAKE-AUDIO repository from GitHub.
    2. Falls back to Hugging Face if GitHub cloning fails.
    3. Installs necessary system (libsndfile1) and Python dependencies.
    4. Attempts to pull large files via Git LFS.
    5. Falls back to Kagglehub for data if LFS budget is exceeded.
    """
    try:
        # Attempt to get the IPython shell for executing system commands.
        shell = get_ipython()
    except NameError:
        # If NameError, we are not in an interactive IPython environment.
        print("🏠 Running in local or custom environment. Skipping cloud setup.")
        return

    repo_name = "DEEPFAKE-AUDIO"

    # --- Google Colab Specific Setup ---
    if IS_COLAB:
        print("💻 Detected Google Colab Environment. Initiating setup...")
        colab_working_dir = f"/content/{repo_name}"

        # Step 1: Clone the Repository
        if not os.path.exists(colab_working_dir):
            print("⬇️ Cloning DEEPFAKE-AUDIO repository from GitHub...")
            shell.system(f"git clone https://github.com/Amey-Thakur/{repo_name}")

            # Fallback to Hugging Face if GitHub clone fails
            if not os.path.exists(colab_working_dir) or not os.listdir(colab_working_dir):
                print("⚠️ GitHub Clone Failed. Attempting Fallback: Hugging Face Space...")
                if os.path.exists(colab_working_dir):
                    shutil.rmtree(colab_working_dir)
                shell.system(f"git clone https://huggingface.co/spaces/ameythakur/Deepfake-Audio {repo_name}")
                print("✅ Cloned from Hugging Face Space.")

        os.chdir(colab_working_dir)

        # Step 2: Install Dependencies
        print("🔧 Installing system and Python dependencies...")
        shell.system("apt-get install -y libsndfile1")
        deps = "librosa==0.9.2 unidecode webrtcvad inflect umap-learn scikit-learn>=1.3 tqdm scipy 'matplotlib>=3.7,<3.9' Pillow>=10.2 soundfile huggingface_hub"
        shell.system(f"pip install {deps}")

        # Step 3: Attempt Git LFS Pull
        print("📦 Attempting Git LFS pull for large model files...")
        shell.system("git lfs install")
        lfs_status = shell.system("git lfs pull")

        # Step 4: Kaggle Fallback if LFS Fails
        # Check for a valid sample file to determine LFS success.
        sample_trigger = "Dataset/samples/Steve Jobs.wav"
        is_lfs_failed = (
            lfs_status != 0 or
            not os.path.exists(sample_trigger) or
            os.path.getsize(sample_trigger) < 1000  # LFS pointers are < 1KB
        )

        if is_lfs_failed:
            print("⚠️ GitHub LFS failed or exceeded budget. Using Kaggle Fallback...")
            shell.system("pip install kagglehub")
            import kagglehub
            print("🚀 Downloading assets from Kagglehub (ameythakur20/deepfakeaudio)...")
            kaggle_path = kagglehub.dataset_download("ameythakur20/deepfakeaudio")
            _link_kaggle_assets(kaggle_path)

    # --- Kaggle Specific Setup ---
    elif IS_KAGGLE:
        print("💻 Detected Kaggle Environment. Initiating setup...")
        kaggle_working_dir = f"/kaggle/working/{repo_name}"

        # Step 1: Clone the Repository
        if not os.path.exists(kaggle_working_dir):
            print("⬇️ Cloning DEEPFAKE-AUDIO repository from GitHub...")
            os.chdir("/kaggle/working")
            shell.system(f"git clone https://github.com/Amey-Thakur/{repo_name}")

        os.chdir(kaggle_working_dir)

        # Step 2: Link Kaggle Dataset (Highest Priority on Kaggle)
        kaggle_input_path = "/kaggle/input/deepfakeaudio"
        if os.path.exists(kaggle_input_path):
            print(f"✅ Kaggle Dataset Detected at {kaggle_input_path}. Linking assets...")
            _link_kaggle_assets(kaggle_input_path)
        else:
            print("⚠️ Kaggle Input not found. Attempting Git LFS pull...")
            shell.system("git lfs install")
            shell.system("git lfs pull")

        # Step 3: Install Dependencies
        print("🔧 Installing dependencies...")
        shell.system("apt-get install -y libsndfile1")
        deps = "librosa==0.9.2 unidecode webrtcvad inflect umap-learn scikit-learn>=1.3 tqdm scipy 'matplotlib>=3.7,<3.9' Pillow>=10.2 soundfile huggingface_hub"
        shell.system(f"pip install {deps}")
        print("✅ Environment setup complete. Ready for cloning.")


def _link_kaggle_assets(source_path):
    """
    Helper function to create symbolic links from Kaggle data to the expected 'Dataset/' folder.

    Args:
        source_path: The root path of the downloaded/linked Kaggle data.
    """
    target_dir = "Dataset"
    os.makedirs(target_dir, exist_ok=True)

    # Link samples folder
    k_samples = os.path.join(source_path, "samples")
    target_samples = os.path.join(target_dir, "samples")
    if os.path.exists(k_samples):
        if os.path.exists(target_samples):
            shutil.rmtree(target_samples)
        os.symlink(k_samples, target_samples)
        print("✅ Samples linked from Kaggle.")

    # Link model files
    for model_name in ["encoder.pt", "synthesizer.pt", "vocoder.pt"]:
        src = os.path.join(source_path, model_name)
        dst = os.path.join(target_dir, model_name)
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
    print("✅ Models linked from Kaggle.")


# ==================================================================================================
# SECTION 3: MODEL & DATA INITIALIZATION
# ==================================================================================================
# This section handles the discovery and validation of model checkpoints (encoder,
# synthesizer, vocoder). It implements a priority-based search across local paths,
# Kaggle datasets, and Hugging Face.

def is_valid_checkpoint(filepath):
    """
    Validates a file path to ensure it's a real model checkpoint and not an
    LFS pointer file.

    Args:
        filepath: Path object or string to the file.

    Returns:
        True if the file exists and is larger than 1KB, False otherwise.
    """
    path = Path(filepath)
    return path.exists() and path.stat().st_size > 1000


def resolve_checkpoint_path(component_name: str, legacy_suffix: str) -> Path:
    """
    Resolves the path to a model checkpoint using a priority-based search strategy.

    Search Order:
    1. Local 'Dataset/' folder (part of the repository).
    2. Kaggle Input directory (`/kaggle/input/deepfakeaudio/`).
    3. Auto-downloaded 'pretrained_models/default/' folder.
    4. Legacy/manual paths for advanced users.

    Args:
        component_name: The name of the component ('Encoder', 'Synthesizer', 'Vocoder').
        legacy_suffix: A fallback path suffix for older project structures.

    Returns:
        A Path object to the checkpoint, or None if not found.
    """
    model_filename = f"{component_name.lower()}.pt"

    # Priority 1: Local Repository 'Dataset/' folder
    local_path = Path("Dataset") / model_filename
    if is_valid_checkpoint(local_path):
        print(f"🟢 Loading {component_name} from Repository: {local_path}")
        return local_path

    # Priority 2: Kaggle Environment
    kaggle_path = Path("/kaggle/input/deepfakeaudio") / model_filename
    if is_valid_checkpoint(kaggle_path):
        print(f"🟢 Loading {component_name} from Kaggle: {kaggle_path}")
        return kaggle_path

    # Priority 3: Auto-Downloaded Fallback ('pretrained_models/default/')
    default_path = Path("pretrained_models/default") / model_filename
    if is_valid_checkpoint(default_path):
        print(f"🟢 Loading {component_name} from Fallback: {default_path}")
        return default_path

    # Priority 4: Legacy/Manual Paths
    legacy_path = Path("pretrained_models") / legacy_suffix
    if legacy_path.exists():
        if legacy_path.is_dir():
            pts = [f for f in legacy_path.glob("*.pt") if is_valid_checkpoint(f)]
            if pts:
                return pts[0]
        elif is_valid_checkpoint(legacy_path):
            return legacy_path

    print(f"⚠️ Warning: {component_name} checkpoint not found!")
    return None


def download_models_from_huggingface():
    """
    Attempts to download model checkpoints from Hugging Face if they are not
    found locally.

    Tries the following sources in order:
    1. Personal Hugging Face Space: `ameythakur/Deepfake-Audio`
    2. External Fallback: Uses the `utils.default_models` script.
    """
    from huggingface_hub import hf_hub_download

    core_models = ["encoder.pt", "synthesizer.pt", "vocoder.pt"]
    target_dir = Path("pretrained_models")
    target_dir.mkdir(exist_ok=True)

    print("🚀 Attempting download from Hugging Face Space (ameythakur/Deepfake-Audio)...")
    try:
        for model in core_models:
            try:
                # Try nested path first (Dataset/model.pt)
                fpath = hf_hub_download(
                    repo_id="ameythakur/Deepfake-Audio",
                    filename=f"Dataset/{model}",
                    repo_type="space",
                    local_dir=str(target_dir)
                )
            except Exception:
                # Try root path (model.pt)
                fpath = hf_hub_download(
                    repo_id="ameythakur/Deepfake-Audio",
                    filename=model,
                    repo_type="space",
                    local_dir=str(target_dir)
                )
            target_file = target_dir / model
            if Path(fpath) != target_file and Path(fpath).exists():
                shutil.move(fpath, target_file)

        nested_folder = target_dir / "Dataset"
        if nested_folder.exists():
            shutil.rmtree(nested_folder)
        print("✅ Models successfully acquired via Personal Hugging Face fallback.")
        return True
    except Exception as e:
        print(f"⚠️ Personal HF download failed: {e}. Trying external fallback...")

    # External Fallback: Use the utility script from the project
    try:
        from utils.default_models import ensure_default_models
        ensure_default_models(target_dir)
        print("✅ Models successfully acquired via External HuggingFace fallback.")
        return True
    except Exception as e:
        print(f"❌ Critical: Could not auto-download models. Error: {e}")
        return False


def verify_and_load_pipeline():
    """
    Verifies the availability of model checkpoints and loads the SV2TTS pipeline.

    This function first checks for locally available models. If models are missing,
    it attempts to download them from Hugging Face. Finally, it loads the encoder,
    synthesizer, and vocoder into memory.

    Returns:
        A tuple containing (encoder_module, synthesizer_instance, vocoder_module),
        or None if loading fails.
    """
    import torch
    from encoder import inference as encoder
    from synthesizer.inference import Synthesizer
    from vocoder import inference as vocoder

    print(f"🎯 Computation Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("⬇️ Verifying Model Availability...")

    core_models = ["encoder.pt", "synthesizer.pt", "vocoder.pt"]
    dataset_models_present = all(
        is_valid_checkpoint(Path("Dataset") / m) for m in core_models
    )

    if dataset_models_present:
        print("✅ Found high-priority local models in 'Dataset/'. Verified.")
    else:
        # Attempt to download if not present
        kaggle_path = Path("/kaggle/input/deepfakeaudio")
        kaggle_models_present = all(
            is_valid_checkpoint(kaggle_path / m) for m in core_models
        )
        if not kaggle_models_present:
            print("⚠️ Models not found locally. Attempting fallback download...")
            download_models_from_huggingface()

    print("⏳ Loading Neural Networks (SV2TTS Pipeline)...")
    try:
        # 1. Encoder: Extract speaker embedding
        encoder_path = resolve_checkpoint_path("Encoder", "encoder/saved_models")
        encoder.load_model(encoder_path)

        # 2. Synthesizer: Generates spectrograms from text
        synth_path = resolve_checkpoint_path(
            "Synthesizer", "synthesizer/saved_models/logs-pretrained/taco_pretrained"
        )
        synthesizer_instance = Synthesizer(synth_path)

        # 3. Vocoder: Converts spectrograms to audio waveforms
        vocoder_path = resolve_checkpoint_path("Vocoder", "vocoder/saved_models/pretrained")
        vocoder.load_model(vocoder_path)

        print("✅ Pipeline operational. All components loaded correctly.")
        return encoder, synthesizer_instance, vocoder
    except Exception as e:
        print(f"❌ Architecture Error: {e}")
        return None


# ==================================================================================================
# SECTION 4: AUDIO SAMPLE DISCOVERY
# ==================================================================================================
# This section provides utilities for finding reference audio samples (e.g., celebrity
# presets) across various locations in the file system.

def find_samples_directory():
    """
    Locates the directory containing reference audio samples.

    Searches through a prioritized list of paths and returns the first one
    containing valid audio files (>.wav or .mp3 > 1KB).

    Returns:
        A tuple (samples_dir_path, list_of_filenames) or (None, []) if not found.
    """
    priority_paths = [
        "Source Code/samples",
        "Dataset/samples",
        "D:/GitHub/DEEPFAKE-AUDIO/Source Code/samples",
        "D:/GitHub/DEEPFAKE-AUDIO/Dataset/samples",
        "/content/DEEPFAKE-AUDIO/Source Code/samples",
        "/kaggle/input/deepfakeaudio/samples",
        "/kaggle/input/deepfakeaudio",
    ]

    def get_valid_audio_files(directory):
        """Filters a directory for real audio files (not LFS pointers)."""
        if not os.path.exists(directory):
            return []
        return [
            f for f in os.listdir(directory)
            if f.lower().endswith((".wav", ".mp3")) and
               os.path.getsize(os.path.join(directory, f)) > 1024
        ]

    for path in priority_paths:
        files = get_valid_audio_files(path)
        if files:
            print(f"✅ Samples located at: {os.path.abspath(path)}")
            return path, files

    # Fallback: Use glob to search recursively
    print("🔍 Searching for audio samples via glob...")
    potential_matches = (
        glob.glob("**/samples/*.wav", recursive=True) +
        glob.glob("**/samples/*.mp3", recursive=True)
    )
    valid_matches = [m for m in potential_matches if os.path.getsize(m) > 1024]

    if valid_matches:
        root_dir = os.path.dirname(valid_matches[0])
        files = [os.path.basename(f) for f in valid_matches]
        print(f"✨ Located samples via glob at: {os.path.abspath(root_dir)}")
        return root_dir, list(set(files))

    return None, []


# ==================================================================================================
# SECTION 5: INFERENCE & VISUALIZATION
# ==================================================================================================
# This section contains the core voice cloning logic and functions for visualizing
# the results (waveforms, spectrograms, embeddings).

def clone_voice(
    encoder_module,
    synthesizer_instance,
    vocoder_module,
    input_audio_path: str,
    text_to_synthesize: str
):
    """
    Performs the voice cloning inference.

    This is the main function that orchestrates the three-stage pipeline:
    encode -> synthesize -> vocalize.

    Args:
        encoder_module: The loaded encoder module.
        synthesizer_instance: The loaded Synthesizer instance.
        vocoder_module: The loaded vocoder module.
        input_audio_path: Path to the reference audio file.
        text_to_synthesize: The text string to be spoken by the cloned voice.

    Returns:
        A tuple (generated_waveform, mel_spectrogram, speaker_embedding, original_waveform).
    """
    import numpy as np
    import librosa

    print(f"🎙️ Reference Audio: {input_audio_path}")
    print(f"📝 Text to Clone: \"{text_to_synthesize[:80]}...\"" if len(text_to_synthesize) > 80 else f"📝 Text to Clone: \"{text_to_synthesize}\"")

    # --- Step 1: Encode ---
    # Load the reference audio and preprocess it for the encoder.
    print("⏳ Step 1/3: Encoding speaker identity...")
    original_wav, sampling_rate = librosa.load(input_audio_path)
    preprocessed_wav = encoder_module.preprocess_wav(original_wav, sampling_rate)
    speaker_embedding = encoder_module.embed_utterance(preprocessed_wav)
    print(f"   -> Embedding shape: {speaker_embedding.shape}")

    # --- Step 2: Synthesize ---
    # Generate a Mel-Spectrogram from the text, conditioned on the speaker embedding.
    print("⏳ Step 2/3: Synthesizing speech (Mel-Spectrogram)...")
    specs = synthesizer_instance.synthesize_spectrograms([text_to_synthesize], [speaker_embedding])
    mel_spectrogram = specs[0]
    print(f"   -> Spectrogram shape: {mel_spectrogram.shape}")

    # --- Step 3: Vocalize ---
    # Convert the Mel-Spectrogram into a raw audio waveform.
    print("⏳ Step 3/3: Generating waveform (WaveRNN)...")
    generated_wav = vocoder_module.infer_waveform(mel_spectrogram)
    print(f"   -> Waveform length: {len(generated_wav)} samples")

    print("🎉 Synthesis Complete!")
    return generated_wav, mel_spectrogram, speaker_embedding, original_wav


def visualize_results(original_wav, generated_wav, mel_spectrogram, speaker_embedding):
    """
    Generates a multi-panel visualization of the cloning results.

    Creates a figure with three subplots:
    1. Waveform comparison: Original vs. Cloned audio.
    2. Generated Mel-Spectrogram.
    3. Speaker Embedding heatmap.

    Args:
        original_wav: The original waveform as a NumPy array.
        generated_wav: The generated waveform as a NumPy array.
        mel_spectrogram: The Mel-Spectrogram as a NumPy array.
        speaker_embedding: The 256-D speaker embedding vector.
    """
    import matplotlib.pyplot as plt
    import librosa.display

    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 14))

        # --- Panel 1: Waveform Comparison ---
        axes[0].set_title("Input Voice vs. Cloned Voice (Waveform)", fontsize=14)
        try:
            librosa.display.waveshow(original_wav, alpha=0.6, ax=axes[0], label="Original", color='blue')
            librosa.display.waveshow(generated_wav, alpha=0.6, ax=axes[0], label="Cloned", color='red')
        except Exception:
            axes[0].plot(original_wav, alpha=0.6, label="Original", color='blue')
            axes[0].plot(generated_wav, alpha=0.6, label="Cloned", color='red')
        axes[0].legend(loc='upper right')
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Amplitude")

        # --- Panel 2: Mel-Spectrogram ---
        axes[1].set_title("Generated Mel Spectrogram", fontsize=14)
        img = axes[1].imshow(mel_spectrogram, aspect="auto", origin="lower", interpolation="none", cmap='magma')
        fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
        axes[1].set_xlabel("Time Frames")
        axes[1].set_ylabel("Mel Frequency Bins")

        # --- Panel 3: Speaker Embedding ---
        axes[2].set_title("Speaker Embedding (256-D Identity Fingerprint)", fontsize=14)
        if len(speaker_embedding) == 256:
            axes[2].imshow(speaker_embedding.reshape(16, 16), aspect='auto', cmap='viridis')
        else:
            axes[2].plot(speaker_embedding)
        axes[2].set_xlabel("Dimension")
        axes[2].set_ylabel("Value")

        plt.tight_layout()
        plt.savefig("cloning_results.png", dpi=150)
        print("📊 Analysis saved to 'cloning_results.png'.")
        plt.show()
    except Exception as e:
        print(f"⚠️ Visualization partially failed: {e}. Audio cloning was successful.")


# ==================================================================================================
# SECTION 6: MAIN EXECUTION
# ==================================================================================================
# This is the entry point for the script when run from the command line.

def main():
    """
    Main entry point for the DEEPFAKE-AUDIO script.

    Parses command-line arguments and runs the voice cloning pipeline.
    """
    parser = argparse.ArgumentParser(
        description="DEEPFAKE-AUDIO: Neural Voice Cloning with SV2TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python DEEPFAKE-AUDIO.py --preset "Steve Jobs.wav" --text "Hello world!"
  python DEEPFAKE-AUDIO.py --input "my_voice.wav" --text "This is a cloned message."
        """
    )
    parser.add_argument(
        "--preset", type=str, default=None,
        help="Name of a preset sample file (e.g., 'Steve Jobs.wav')."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to a custom audio file to use as the voice reference."
    )
    parser.add_argument(
        "--text", type=str,
        default="Hello, I'm a cloned voice. Welcome to Deepfake Audio by Amey Thakur and Mega Satish.",
        help="The text to be synthesized by the cloned voice."
    )
    parser.add_argument(
        "--output", type=str, default="cloned_output.wav",
        help="Path to save the generated audio file."
    )

    args = parser.parse_args()

    # --- Step 1: Cloud Setup (if applicable) ---
    setup_cloud_environment()

    # --- Step 2: Verify and Load Pipeline ---
    pipeline = verify_and_load_pipeline()
    if pipeline is None:
        print("❌ Failed to load the SV2TTS pipeline. Exiting.")
        sys.exit(1)
    encoder_mod, synthesizer_inst, vocoder_mod = pipeline

    # --- Step 3: Determine Input Audio ---
    input_audio_path = None
    if args.input:
        input_audio_path = args.input
    elif args.preset:
        samples_dir, _ = find_samples_directory()
        if samples_dir:
            input_audio_path = os.path.join(samples_dir, args.preset)
        else:
            print(f"❌ Could not find samples directory for preset '{args.preset}'.")
            sys.exit(1)
    else:
        # Default to first available preset
        samples_dir, preset_files = find_samples_directory()
        if samples_dir and preset_files:
            input_audio_path = os.path.join(samples_dir, preset_files[0])
            print(f"ℹ️ No input specified. Using default preset: {preset_files[0]}")
        else:
            print("❌ No input audio provided and no presets found. Exiting.")
            sys.exit(1)

    if not os.path.exists(input_audio_path):
        print(f"❌ Audio file not found: {input_audio_path}")
        sys.exit(1)

    # --- Step 4: Run Inference ---
    generated_wav, mel_spec, embedding, original_wav = clone_voice(
        encoder_mod, synthesizer_inst, vocoder_mod,
        input_audio_path, args.text
    )

    # --- Step 5: Save Output ---
    import soundfile as sf
    sf.write(args.output, generated_wav, synthesizer_inst.sample_rate)
    print(f"💾 Cloned audio saved to: {args.output}")

    # --- Step 6: Visualize Results ---
    visualize_results(original_wav, generated_wav, mel_spec, embedding)


if __name__ == "__main__":
    main()
