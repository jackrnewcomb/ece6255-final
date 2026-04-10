import numpy as np


def seconds_to_sample_index(t: float, sample_rate: int, n_samples: int) -> int:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if n_samples < 0:
        raise ValueError("n_samples must be nonnegative")
    if t < 0:
        raise ValueError("time values must be nonnegative")

    idx = int(round(t * sample_rate))
    return max(0, min(idx, n_samples))


def extract_segments(samples: np.ndarray, sample_rate: int, t1: float, t2: float):
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError("samples must be a 1-D array")
    if samples.size == 0:
        raise ValueError("samples cannot be empty")
    if t1 < 0 or t2 < 0:
        raise ValueError("time values must be nonnegative")
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1")

    n_samples = len(samples)
    duration = n_samples / sample_rate
    if t1 >= duration:
        raise ValueError(f"t1={t1} exceeds or equals audio duration of {duration:.6f} s")
    if t2 > duration:
        raise ValueError(f"t2={t2} exceeds audio duration of {duration:.6f} s")

    start = seconds_to_sample_index(t1, sample_rate, n_samples)
    end = seconds_to_sample_index(t2, sample_rate, n_samples)

    if end <= start:
        raise ValueError("Selected segment is empty after converting times to sample indices")

    prefix = samples[:start]
    segment = samples[start:end]
    suffix = samples[end:]

    return prefix, segment, suffix