# Anish's stuff, feel free to change this

import numpy as np

def time_scale_psola(segment, sample_rate, scale):
    if scale <= 0:
        raise ValueError("scale must be positive")
    return 0.3 * segment