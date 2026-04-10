import numpy as np

def stitch_segments(prefix: np.ndarray, processed_segment: np.ndarray, suffix: np.ndarray) -> np.ndarray:
    return np.concatenate([prefix, processed_segment, suffix])