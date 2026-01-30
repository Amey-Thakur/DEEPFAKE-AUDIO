# ==================================================================================================
# DEEPFAKE AUDIO - encoder/data_objects/random_cycler.py (Constrained Stochastic Iteration)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This utility provides a 'RandomCycler' class designed for uniform yet stochastic 
# sampling of dataset items. It ensures that every item in a collection is 
# seen with a guaranteed frequency, avoiding potential biases or 'starvation' 
# during neural optimization steps while maintaining sufficient randomness.
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

import random

class RandomCycler:
    """
    Uniform Stochastic Sampler:
    Maintains a sequence where each item is guaranteed to appear within a controlled 
    interval, ensuring balanced categorical exposure during training.
    """
    def __init__(self, source):
        if len(source) == 0:
            raise Exception("Fatal: Cannot initialize RandomCycler with an empty collection.")
        self.all_items = list(source)
        self.next_items = []
    
    def sample(self, count: int):
        """
        Retrieves a 'count' number of items, replenishing and shuffling the internal 
        pool as needed to maintain stochastisity without repetition within a cycle.
        """
        shuffle = lambda l: random.sample(l, len(l))
        
        out = []
        while count > 0:
            # High-Volume Requests
            if count >= len(self.all_items):
                out.extend(shuffle(list(self.all_items)))
                count -= len(self.all_items)
                continue
            
            # Partial Pool Refresh
            n = min(count, len(self.next_items))
            out.extend(self.next_items[:n])
            count -= n
            self.next_items = self.next_items[n:]
            
            if len(self.next_items) == 0:
                self.next_items = shuffle(list(self.all_items))
                
        return out
    
    def __next__(self):
        """Standard Python iterator hook for single-sample acquisition."""
        return self.sample(1)[0]

