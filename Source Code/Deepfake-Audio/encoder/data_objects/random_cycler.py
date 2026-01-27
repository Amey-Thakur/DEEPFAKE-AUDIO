"""
Deepfake Audio - Random Cycler
------------------------------
A utility class for randomized but guaranteed sampling of items from a collection.
It ensures that all items are visited roughly equally often over many samples.

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

import random
from typing import List, Any, Sequence

class RandomCycler:
    """
    Creates an internal copy of a sequence and allows access to its items in a constrained random 
    order.
    
    Guarantees:
        - Each item will be returned between m // n and ((m - 1) // n) + 1 times.
        - Between two appearances of the same item, there may be at most 2 * (n - 1) other items.
    """
    
    def __init__(self, source: Sequence[Any]):
        """
        Args:
            source: The source collection to sample from.
        """
        if len(source) == 0:
            raise ValueError("Can't create RandomCycler from an empty collection")
        self.all_items = list(source)
        self.next_items: List[Any] = []
    
    def sample(self, count: int) -> List[Any]:
        """
        Returns a list of <count> items from the collection, ensuring coverage constraints.
        """
        shuffle = lambda l: random.sample(l, len(l))
        
        out = []
        while count > 0:
            if count >= len(self.all_items):
                out.extend(shuffle(list(self.all_items)))
                count -= len(self.all_items)
                continue
            
            n = min(count, len(self.next_items))
            out.extend(self.next_items[:n])
            count -= n
            self.next_items = self.next_items[n:]
            
            if len(self.next_items) == 0:
                self.next_items = shuffle(list(self.all_items))
                
        return out
    
    def __next__(self) -> Any:
        return self.sample(1)[0]

