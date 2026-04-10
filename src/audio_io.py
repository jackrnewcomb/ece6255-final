from pathlib import Path
import numpy as np
from scipy.io import wavfile


def _normalize_integer_audio(data: np.ndarray) -> np.ndarray:
    info = np.iinfo(data.dtype)
    scale = max(abs(info.min), abs(info.max))
    return data.astype(np.float32) / float(scale)


def load_wav(path: str):
    wav_path = Path(path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Input file not found: {wav_path}")
    if not wav_path.is_file():
        raise ValueError(f"Input path is not a file: {wav_path}")

    try:
        sample_rate, data = wavfile.read(wav_path)
    except Exception as exc:
        raise ValueError(f"Could not read WAV file: {wav_path}") from exc

    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate in WAV file: {sample_rate}")
    if data.ndim != 1:
        raise ValueError("Phase 1 supports mono WAV files only")
    if data.size == 0:
        raise ValueError("Loaded audio is empty")

    if np.issubdtype(data.dtype, np.integer):
        data = _normalize_integer_audio(data)
    elif np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV sample dtype: {data.dtype}")

    if not np.all(np.isfinite(data)):
        raise ValueError("Audio contains non-finite values")

    return sample_rate, data


def save_wav(path: str, sample_rate: int, data: np.ndarray):
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 1:
        raise ValueError("Output audio must be a 1-D array")
    if data.size == 0:
        raise ValueError("Output audio is empty")
    if not np.all(np.isfinite(data)):
        raise ValueError("Output audio contains non-finite values")

    data = np.clip(data, -1.0, 1.0)
    out = (data * 32767).astype(np.int16)
    wavfile.write(out_path, sample_rate, out)