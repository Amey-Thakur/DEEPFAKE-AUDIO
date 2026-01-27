"""
Deepfake Audio - Encoder Model Parameters
-----------------------------------------
This configuration file defines the hyperparameters for the Speaker Encoder model architecture
and the training process. These parameters drive the construction of the LSTM network and
the Generalized End-to-End (GE2E) loss optimization.

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

# -----------------------------------------------------------------------------
# Model Architecture Parameters
# -----------------------------------------------------------------------------
# Number of neurons in each LSTM layer
model_hidden_size = 256

# Dimension of the final speaker embedding (d-vector)
model_embedding_size = 256

# Number of stacked LSTM layers
model_num_layers = 3


# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
# Initial learning rate for the Adam optimizer
learning_rate_init = 1e-4

# Batch size configuration for GE2E loss
# Effectively, batch_size = speakers_per_batch * utterances_per_speaker
speakers_per_batch = 64
utterances_per_speaker = 10
