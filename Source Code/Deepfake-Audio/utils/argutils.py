"""
Deepfake Audio - Argument Parser Utilities
------------------------------------------
Helper functions for handling and printing command-line arguments.

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
from typing import List, Any, Optional, Union
import argparse
import numpy as np


# Priority of types for sorting arguments
_type_priorities: List[type] = [
    Path,
    str,
    int,
    float,
    bool,
]


def _priority(o: Any) -> int:
    """
    Determines the priority of an object based on its type.
    Used for sorting arguments by type.
    """
    p = next((i for i, t in enumerate(_type_priorities) if type(o) is t), None) 
    if p is not None:
        return p
    p = next((i for i, t in enumerate(_type_priorities) if isinstance(o, t)), None) 
    if p is not None:
        return p
    return len(_type_priorities)


def print_args(args: Union[argparse.Namespace, dict], parser: Optional[argparse.ArgumentParser] = None) -> None:
    """
    Prints the arguments in a structured and sorted format.

    Args:
        args: Parsed arguments (Namespace or dict).
        parser: Optional ArgumentParser instance to determine grouping order.
    """
    args_dict = vars(args) if isinstance(args, argparse.Namespace) else args
    
    if parser is None:
        priorities = list(map(_priority, args_dict.values()))
    else:
        all_params = [a.dest for g in parser._action_groups for a in g._group_actions ]
        priority = lambda p: all_params.index(p) if p in all_params else len(all_params)
        priorities = list(map(priority, args_dict.keys()))
    
    pad = max(map(len, args_dict.keys())) + 3
    indices = np.lexsort((list(args_dict.keys()), priorities))
    items = list(args_dict.items())
    
    print("Arguments:")
    for i in indices:
        param, value = items[i]
        print("    {0}:{1}{2}".format(param, ' ' * (pad - len(param)), value))
    print("")
    