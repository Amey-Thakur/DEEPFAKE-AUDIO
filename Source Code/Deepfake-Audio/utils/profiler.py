"""
Deepfake Audio - Code Profiler
------------------------------
Simple profiler class to measure and log execution time of code blocks.

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

from time import perf_counter as timer
from collections import OrderedDict
from typing import Dict, List, Optional
import numpy as np


class Profiler:
    """
    A simple profiler to measure execution time of code blocks.
    Logs execution times and prints summaries periodically.
    """
    
    def __init__(self, summarize_every: int = 5, disabled: bool = False):
        """
        Initializes the profiler.

        Args:
            summarize_every: Number of ticks before printing a summary.
            disabled: If True, profiling is disabled.
        """
        self.last_tick = timer()
        self.logs: Dict[str, List[float]] = OrderedDict()
        self.summarize_every = summarize_every
        self.disabled = disabled
    
    def tick(self, name: str) -> None:
        """
        Records the time elapsed since the last tick or initialization.

        Args:
            name: Name of the code block being measured.
        """
        if self.disabled:
            return
        
        # Log the time needed to execute that function
        if not name in self.logs:
            self.logs[name] = []
        if len(self.logs[name]) >= self.summarize_every:
            self.summarize()
            self.purge_logs()
        self.logs[name].append(timer() - self.last_tick)
        
        self.reset_timer()
        
    def purge_logs(self) -> None:
        """Clears the recorded logs."""
        for name in self.logs:
            self.logs[name].clear()
    
    def reset_timer(self) -> None:
        """Resets the internal timer."""
        self.last_tick = timer()
    
    def summarize(self) -> None:
        """Prints a summary of average execution times."""
        if not self.logs:
            return
            
        n = max(map(len, self.logs.values()))
        # assert n == self.summarize_every # This assertion can fail if logs are uneven, removing for safety
        print("\nAverage execution time over %d steps:" % n)

        name_msgs = ["%s (%d/%d):" % (name, len(deltas), n) for name, deltas in self.logs.items()]
        pad = max(map(len, name_msgs))
        for name_msg, deltas in zip(name_msgs, self.logs.values()):
            if deltas:
                print("  %s  mean: %4.0fms   std: %4.0fms" % 
                      (name_msg.ljust(pad), np.mean(deltas) * 1000, np.std(deltas) * 1000))
        print("", flush=True)
    
        