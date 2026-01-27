"""
Deepfake Audio - CLI Demonstration Script
-----------------------------------------
A demonstration script for the Deepfake Audio project, implementing a
command-line interface (CLI) for the Real-Time Voice Cloning framework.

This script integrates the comprehensive pipeline:
1.  **Encoder**: Derives a numerical embedding from a reference voice.
2.  **Synthesizer**: Generates a mel spectrogram from text, conditioned on the embedding.
3.  **Vocoder**: Infers a time-domain waveform from the spectrogram.

Authors:
    - Amey Thakur (https://github.com/Amey-Thakur)
    - Mega Satish (https://github.com/msatmod)

Repository:
    - https://github.com/Amey-Thakur/DEEPFAKE-AUDIO

Release Date:
    - February 06, 2021

License:
    - MIT License

Description:
    This module serves as the primary entry point for headless (non-GUI) interaction
    with the Deepfake Audio system. It demonstrates the Transfer Learning from
    Speaker Verification to Multispeaker Text-To-Speech Synthesis (SV2TTS) architecture.
    The script handles model loading, device configuration (CPU/GPU), and the
    complete inference loop (Embedding -> Spectrogram -> Waveform).
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List, Union

import numpy as np
import soundfile as sf
import librosa
import torch

# Standard Library Imports
from utils.argutils import print_args
from encoder.params_model import model_embedding_size as speaker_embedding_size
from io import StringIO

# Model Interfaces
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

# Optional Audio Playback
try:
    import sounddevice as sd
except ImportError:
    sd = None


def main() -> None:
    """
    Main execution routine for the Deepfake Audio CLI demo.
    
    Orchestrates the loading of models, configuration of hardware acceleration,
    verification of the pipeline integrity, and loops through the interactive
    synthesis process.
    """
    
    # =========================================================================
    # 1. Argument Parsing & Configuration
    # =========================================================================
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the Deepfake Audio CLI demonstration."
    )
    
    parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                        default="encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder model checkpoint.")
    
    parser.add_argument("-s", "--syn_model_dir", type=Path, 
                        default="synthesizer/saved_models/logs-pretrained/",
                        help="Directory containing the synthesizer model checkpoints.")
    
    parser.add_argument("-v", "--voc_model_fpath", type=Path, 
                        default="vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder model checkpoint.")
    
    parser.add_argument("--low_mem", action="store_true", help=\
        "If True, frees synthesizer memory after each inference step. "
        "Useful for GPUs with limited VRAM (e.g., < 4GB).")
    
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, suppresses audio playback via sounddevice.")
        
    parser.add_argument("--cpu", action="store_true", help=\
        "Force execution on CPU, even if CUDA is available.")
        
    args = parser.parse_args()
    print_args(args, parser)

    # Validate sounddevice availability
    if not args.no_sound and sd is None:
        print("[WARNING] sounddevice not found. Audio playback will be disabled.", file=sys.stderr)
        args.no_sound = True

    # =========================================================================
    # 2. Environment & Hardware Verification
    # =========================================================================
    print("Running a test of your configuration...\n")
    
    if args.cpu:
        print("Using CPU for inference.")
    elif torch.cuda.is_available():
        # Retrieve GPU details provided by PyTorch's CUDA interface
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        
        print(f"Found {torch.cuda.device_count()} GPUs available. "
              f"Using GPU {device_id} ({gpu_properties.name}) "
              f"of compute capability {gpu_properties.major}.{gpu_properties.minor} "
              f"with {gpu_properties.total_memory / 1e9:.1f}Gb total memory.\n")
    else:
        print("Your PyTorch installation is not configured for CUDA. "
              "If you have a GPU, please check your drivers and CUDA version.", file=sys.stderr)
        print("Falling back to CPU. Pass --cpu to suppress this warning in the future.", file=sys.stderr)

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    print("Preparing the encoder, the synthesizer and the vocoder...")
    
    # Encoder: Loads the pretrained speaker verification model
    encoder.load_model(args.enc_model_fpath)
    
    # Synthesizer: Loads the Tacotron 2 text-to-spectrogram model
    # Note: We look for 'taco_pretrained' inside the specified directory
    synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem)
    
    # Vocoder: Loads the WaveRNN spectrogram-to-waveform model
    vocoder.load_model(args.voc_model_fpath)

    # =========================================================================
    # 4. Pipeline Verification (Sanity Check)
    # =========================================================================
    print("Testing your configuration with small inputs.")
    
    # 4.1 Encoder Test
    # Forward a 1-second silence vector to verify encoder throughput.
    # The encoder expects input at a specific sampling rate (default 16kHz).
    print("\tTesting the encoder...")
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))
    
    # 4.2 Synthesizer Test
    # Generate a dummy embedding vector for testing.
    # Real embeddings are derived from the encoder, but random noise suffices for
    # verifying the synthesizer's graph execution.
    embed = np.random.rand(speaker_embedding_size)
    embed /= np.linalg.norm(embed) # L2-normalize the embedding (critical for cosine similarity)
    
    embeds = [embed, np.zeros(speaker_embedding_size)] # Batch of 2
    texts = ["test 1", "test 2"]
    
    print("\tTesting the synthesizer... (this may trigger JIT compilation)")
    mels = synthesizer.synthesize_spectrograms(texts, embeds)
    
    # 4.3 Vocoder Test
    # Concatenate generated mel spectrograms to test vocoder processing.
    mel = np.concatenate(mels, axis=1)
    
    # Define a no-op callback to suppress progress bars during this test
    no_action = lambda *args: None
    
    print("\tTesting the vocoder...")
    # Infer waveform with a small target length for speed
    vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
    
    print("All tests passed! You can now synthesize speech.\n\n")

    # =========================================================================
    # 5. Interactive Inference Loop
    # =========================================================================
    print("This is a CLI wrapper for the SV2TTS framework.")
    print("Interactive generation loop initiated.")
    
    num_generated = 0
    
    while True:
        try:
            # -----------------------------------------------------------------
            # 5.1 Reference Audio Input
            # -----------------------------------------------------------------
            message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, wav, m4a, flac, ...):\n"
            in_fpath_str = input(message).replace("\"", "").replace("\'", "").strip()
            in_fpath = Path(in_fpath_str)

            if notin_fpath.exists():
                print(f"Error: File '{in_fpath}' not found.")
                continue

            # -----------------------------------------------------------------
            # 5.2 Embedding Derivation
            # -----------------------------------------------------------------
            # Preprocess the wav: normalize volume, trim silence, resample to encoder rate
            preprocessed_wav = encoder.preprocess_wav(in_fpath)
            print("Loaded file successfully")
            
            # Compute the embedding vector (d-vector)
            embed = encoder.embed_utterance(preprocessed_wav)
            print("Created the embedding")
            
            # -----------------------------------------------------------------
            # 5.3 Text Synthesis (Spectrogram Generation)
            # -----------------------------------------------------------------
            text = input("Write a sentence (+-20 words) to be synthesized:\n")
            
            # Synthesize mel spectrograms (batch mode supported, here batch=1)
            # Returns a list of numpy arrays
            texts = [text]
            embeds = [embed]
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")
            
            # -----------------------------------------------------------------
            # 5.4 Audio Inference (Vocoding)
            # -----------------------------------------------------------------
            print("Synthesizing the waveform:")
            # Convert mel spectrogram to time-domain waveform using WaveRNN
            generated_wav = vocoder.infer_waveform(spec)
            
            # -----------------------------------------------------------------
            # 5.5 Post-Processing & Output
            # -----------------------------------------------------------------
            # Pad silence to compensate for potential sounddevice clipping
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
            
            # Normalize audio to float32 range [-1, 1] for playback/saving
            generated_wav = generated_wav.astype(np.float32)

            # Playback
            if not args.no_sound:
                try:
                    sd.stop()
                    sd.play(generated_wav, synthesizer.sample_rate)
                except Exception as e:
                    print(f"Audio playback error: {e}")

            # Save to disk
            filename = f"demo_output_{num_generated:02d}.wav"
            sf.write(filename, generated_wav, synthesizer.sample_rate)
            
            num_generated += 1
            print(f"\nSaved output as {filename}\n\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Caught exception: {repr(e)}")
            print("Restarting loop...\n")


if __name__ == '__main__':
    main()
