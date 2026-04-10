import numpy as np

def seconds_to_sample_index(t: float, sample_rate: int, n_samples: int) -> int:
    if t < 0:
        raise ValueError("time values must be nonnegative")
    idx = int(round(t * sample_rate))
    return max(0, min(idx, n_samples))

def extract_segments(samples: np.ndarray, sample_rate: int, t1: float, t2: float):
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1")

    n_samples = len(samples)
    start = seconds_to_sample_index(t1, sample_rate, n_samples)
    end = seconds_to_sample_index(t2, sample_rate, n_samples)

    if end <= start:
        raise ValueError("selected segment is empty")

    prefix = samples[:start]
    segment = samples[start:end]
    suffix = samples[end:]

    return prefix, segment, suffix