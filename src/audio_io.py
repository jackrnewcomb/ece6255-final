from pathlib import Path
import numpy as np
from scipy.io import wavfile

def load_wav(path: str):
    sample_rate, data = wavfile.read(path)

    if data.ndim != 1:
        raise ValueError("Only supports mono WAV files")

    # convert to float32 internally
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    return sample_rate, data

def save_wav(path: str, sample_rate: int, data: np.ndarray):
    data = np.asarray(data, dtype=np.float32)
    data = np.clip(data, -1.0, 1.0)

    # convert back to int16 for output
    out = (data * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, out)