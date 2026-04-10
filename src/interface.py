from src.audio_io import load_wav, save_wav
from src.segment import extract_segments
from src.psola import time_scale_psola
from src.stitch import stitch_segments


def _validate_processed_segment(processed_segment):
    if processed_segment is None:
        raise ValueError("time_scale_psola returned None")

    try:
        import numpy as np
        processed_segment = np.asarray(processed_segment, dtype=np.float32)
    except Exception as exc:
        raise ValueError("time_scale_psola must return audio data convertible to a numpy array") from exc

    if processed_segment.ndim != 1:
        raise ValueError("time_scale_psola must return a 1-D array")
    if processed_segment.size == 0:
        raise ValueError("Processed segment is empty")
    if not np.all(np.isfinite(processed_segment)):
        raise ValueError("Processed segment contains non-finite values")

    return processed_segment


def process_file(input_path: str, output_path: str, t1: float, t2: float, scale: float):
    if scale <= 0:
        raise ValueError("scale must be positive")

    sample_rate, samples = load_wav(input_path)
    prefix, segment, suffix = extract_segments(samples, sample_rate, t1, t2)
    processed_segment = time_scale_psola(segment, sample_rate, scale)
    processed_segment = _validate_processed_segment(processed_segment)
    output = stitch_segments(prefix, processed_segment, suffix)
    save_wav(output_path, sample_rate, output)