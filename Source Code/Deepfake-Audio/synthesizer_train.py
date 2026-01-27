"""
Deepfake Audio - Synthesizer Training Script
--------------------------------------------
This script manages the training of the Tacotron 2 Synthesizer model that generates
mel spectrograms from input text, conditioned on speaker embeddings.

The training process optimizes the model to produce spectrograms that closely match
the ground truth human speech, while respecting the speaker characteristics captured
by the embeddings.

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
    The training loop includes:
    1.  **Data Feeder**: Asynchronously loads and batches (text, embedding, spectrogram) tuples.
    2.  **Sequence-to-Sequence Modeling**: Trains the Tacotron 2 encoder-decoder architecture with attention.
    3.  **Loss Optimization**: minimizing L1/L2 reconstruction loss on mel spectrograms.
    4.  **Teacher Forcing**: Uses ground truth frames during training to stabilize convergence.
    5.  **Logging**: Tracks training progress via TensorBoard/Visdom.
"""

import argparse
import os
from typing import Tuple, Any

# Standard Library Imports
from utils.argutils import print_args

# Internal Modules
from synthesizer.hparams import hparams
from synthesizer.train import tacotron_train
from synthesizer import infolog


def prepare_run(args: argparse.Namespace) -> Tuple[str, Any]:
    """
    Configures the training run environment.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        log_dir: Path to the logging directory.
        modified_hp: Hyperparameters object with any CLI overrides applied.
    """
    modified_hp = hparams.parse(args.hparams)
    
    # Configure TensorFlow Logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(args.tf_log_level)
    
    run_name = args.name
    log_dir = os.path.join(args.models_dir, "logs-{}".format(run_name))
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize the project-specific logger
    infolog.init(os.path.join(log_dir, "Terminal_train_log"), run_name, args.slack_url)
    
    return log_dir, modified_hp


def main() -> None:
    """
    Main execution routine for synthesizer training.
    """
    
    parser = argparse.ArgumentParser(
        description="Trains the Tacotron 2 Synthesizer model. "
                    "Requires preprocessed spectrograms and embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # -------------------------------------------------------------------------
    # Positional Arguments
    # -------------------------------------------------------------------------
    parser.add_argument("name", help=\
        "Unique name for this training run (used for logging and checkpoints).")
        
    parser.add_argument("synthesizer_root", type=str, help=\
        "Path to the synthesizer training data directory. "
        "Typically: <datasets_root>/SV2TTS/synthesizer/.")

    # -------------------------------------------------------------------------
    # Optional Arguments (Configuration & Hyperparameters)
    # -------------------------------------------------------------------------
    parser.add_argument("-m", "--models_dir", type=str, default="synthesizer/saved_models/", help=\
        "Directory where model checkpoints, logs, and artifacts will be stored.")
        
    parser.add_argument("--mode", default="synthesis",
                        help="Operating mode for Tacotron synthesis (training vs inference).")
                        
    parser.add_argument("--GTA", default="True",
                        help="Ground Truth Aligned (GTA) synthesis. "
                             "If True, uses teacher forcing to align synthesis with ground truth.")
                             
    parser.add_argument("--restore", type=bool, default=True,
                        help="If True, attempts to restore training from the last checkpoint. "
                             "Set False to start fresh.")
    
    parser.add_argument("--summary_interval", type=int, default=2500,
                        help="Step interval for logging training summaries.")
                        
    parser.add_argument("--embedding_interval", type=int, default=10000,
                        help="Step interval for updating embedding visualizations.")
                        
    parser.add_argument("--checkpoint_interval", type=int, default=2000, # Was 5000
                        help="Step interval for saving model checkpoints to disk.")
                        
    parser.add_argument("--eval_interval", type=int, default=100000, # Was 10000
                        help="Step interval for running evaluation on test data.")
                        
    parser.add_argument("--tacotron_train_steps", type=int, default=2000000, # Was 100000
                        help="Total number of training steps before stopping.")
                        
    parser.add_argument("--tf_log_level", type=int, default=1, help="Tensorflow C++ log level.")
    
    parser.add_argument("--slack_url", default=None,
                        help="Slack webhook URL for training notifications.")
                        
    parser.add_argument("--hparams", default="",
                        help="Hyperparameter overrides as a comma-separated list of name=value pairs.")
                        
    args = parser.parse_args()
    print_args(args, parser)
    
    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------
    log_dir, hparams_config = prepare_run(args)
    
    # Start Training
    tacotron_train(args, log_dir, hparams_config)


if __name__ == "__main__":
    main()
