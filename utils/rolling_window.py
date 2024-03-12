#!/usr/bin/env python3
import numpy as np

def rolling_window(window_size: int, *arrays):
    if len(arrays) == 0:
        raise ValueError("Requires at least one ndarray input")

    num_elements = len(arrays[0])

    # Check if the length of input ndarray is the same
    if not all(len(arr) == num_elements for arr in arrays):
        raise ValueError("All input ndarrays must be of the same length")

    subsets = []

    # rolling window
    for i in range(num_elements - window_size + 1):
        subset = np.array([arr[i:i+window_size] for arr in arrays])
        subsets.append(subset)

    return tuple(subsets)