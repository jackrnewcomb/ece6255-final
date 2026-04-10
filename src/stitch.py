import numpy as np


def stitch_segments(prefix: np.ndarray, processed_segment: np.ndarray, suffix: np.ndarray) -> np.ndarray:
    prefix = np.asarray(prefix)
    processed_segment = np.asarray(processed_segment)
    suffix = np.asarray(suffix)

    for name, arr in (
        ("prefix", prefix),
        ("processed_segment", processed_segment),
        ("suffix", suffix),
    ):
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")

    if processed_segment.size == 0:
        raise ValueError("processed_segment cannot be empty")

    output = np.concatenate([prefix, processed_segment, suffix])
    if output.size == 0:
        raise ValueError("Final output is empty")

    return output.astype(np.float32, copy=False)