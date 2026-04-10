# Anish's stuff, feel free to change this

import numpy as np

def time_scale_psola(segment: np.ndarray, sample_rate: int, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be positive")
    return segment.copy()