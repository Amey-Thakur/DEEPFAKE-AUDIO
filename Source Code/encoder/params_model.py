# ==================================================================================================
# DEEPFAKE AUDIO - encoder/params_model.py (Neural Hyperparameters)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module defines the architectural and optimization hyperparameters for 
# the Speaker Encoder. These values determine the depth of the LSTM backbone, 
# the dimensionality of the identity manifold (d-vector space), and the data 
# orchestrations (batch size) required for training stability.
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

# --- ARCHITECTURUAL DIMENSIONS ---
model_hidden_size = 256        # LSTM hidden state capacity
model_embedding_size = 256     # Final identity vector (d-vector) dimensionality
model_num_layers = 3           # Depth of the recurrent stack

# --- OPTIMIZATION ORCHESTRATION ---
learning_rate_init = 1e-4      # Initial stochastic gradient descent scaling
speakers_per_batch = 64        # Categorical diversity per optimization step
utterances_per_speaker = 10    # Sample diversity per categorical identity
