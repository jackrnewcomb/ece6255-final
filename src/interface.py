import numpy as np

from src.audio_io import load_wav, save_wav
from src.segment import extract_segments
from src.psola import time_scale_psola
from src.stitch import stitch_segments
from src.utils import load_edit_spec

def _resolve_scale(t1: float, t2: float, scale: float | None, target_duration: float | None) -> float:
    original_duration = t2 - t1

    if original_duration <= 0:
        raise ValueError("t2 must be greater than t1")

    if scale is not None and target_duration is not None:
        raise ValueError("Provide either scale or target_duration, not both")

    if scale is None and target_duration is None:
        raise ValueError("Must provide either scale or target_duration")

    if scale is not None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        return scale

    if target_duration <= 0:
        raise ValueError("target_duration must be positive")

    return target_duration / original_duration


def _validate_processed_segment(processed_segment):
    if processed_segment is None:
        raise ValueError("time_scale_psola returned None")

    try:
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


def _apply_single_edit(
    samples: np.ndarray,
    sample_rate: int,
    t1: float,
    t2: float,
    scale: float | None = None,
    target_duration: float | None = None,
) -> np.ndarray:
    resolved_scale = _resolve_scale(t1, t2, scale, target_duration)

    prefix, segment, suffix = extract_segments(samples, sample_rate, t1, t2)
    processed_segment = time_scale_psola(segment, sample_rate, resolved_scale)
    processed_segment = _validate_processed_segment(processed_segment)

    return stitch_segments(prefix, processed_segment, suffix)


def process_file(
    input_path: str,
    output_path: str,
    t1: float,
    t2: float,
    scale: float | None = None,
    target_duration: float | None = None,
):
    sample_rate, samples = load_wav(input_path)
    output = _apply_single_edit(
        samples,
        sample_rate,
        t1,
        t2,
        scale=scale,
        target_duration=target_duration,
    )
    save_wav(output_path, sample_rate, output)


def process_file_batch(input_path: str, output_path: str, csv_path: str):
    sample_rate, samples = load_wav(input_path)
    edits = load_edit_spec(csv_path)

    # Treat CSV times as referring to the ORIGINAL input timeline.
    edits = sorted(edits, key=lambda e: e["t1"])

    for prev, curr in zip(edits, edits[1:]):
        if curr["t1"] < prev["t2"]:
            raise ValueError(
                f"Overlapping batch edits are not supported: "
                f"({prev['t1']}, {prev['t2']}) overlaps ({curr['t1']}, {curr['t2']})"
            )

    current = samples
    time_offset = 0.0  # cumulative shift between original timeline and current signal

    for edit in edits:
        original_t1 = edit["t1"]
        original_t2 = edit["t2"]

        resolved_scale = _resolve_scale(
            original_t1,
            original_t2,
            edit.get("scale"),
            edit.get("target_duration"),
        )

        original_duration = original_t2 - original_t1
        new_duration = original_duration * resolved_scale

        adjusted_t1 = original_t1 + time_offset
        adjusted_t2 = original_t2 + time_offset

        current = _apply_single_edit(
            current,
            sample_rate,
            t1=adjusted_t1,
            t2=adjusted_t2,
            scale=resolved_scale,
        )

        time_offset += new_duration - original_duration

    save_wav(output_path, sample_rate, current)
