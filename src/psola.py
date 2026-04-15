import numpy as np
from scipy.signal import correlate


def time_scale_psola(segment: np.ndarray, sample_rate: int, scale: float) -> np.ndarray:
    """
    Time-scale speech using TD-PSOLA without altering pitch.

    Args:
        segment: Mono audio samples (float, typically -1 to 1)
        sample_rate: Sample rate in Hz
        scale: Duration multiplier (>1 = longer/slower, <1 = shorter/faster)

    Returns:
        Time-scaled audio as np.ndarray
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    if segment.size == 0:
        return segment.copy()
    if np.isclose(scale, 1.0, atol=1e-6):
        return segment.copy()

    signal = np.asarray(segment, dtype=np.float64)

    # Detect pitch marks (epoch-like locations)
    pitch_marks = _detect_pitch_marks(signal, sample_rate)

    if len(pitch_marks) < 2:
        # Fallback for very short / poorly voiced segments
        return _resample_linear(signal, scale).astype(segment.dtype, copy=False)

    output = _psola_synthesize(signal, pitch_marks, scale)

    # Mild peak protection so overlap-add does not overshoot badly
    peak = np.max(np.abs(output)) if output.size > 0 else 0.0
    if peak > 1.0:
        output = output / peak

    return output.astype(segment.dtype, copy=False)


def _refine_mark_to_peak(signal: np.ndarray, mark: int, search_radius: int) -> int:
    """
    Refine a rough pitch mark to a nearby strong local waveform extremum.
    """
    left = max(0, mark - search_radius)
    right = min(len(signal), mark + search_radius + 1)
    if right <= left:
        return int(np.clip(mark, 0, len(signal) - 1))

    local = signal[left:right]
    refined = left + np.argmax(np.abs(local))
    return int(refined)


def _detect_pitch_marks(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Detect approximately pitch-synchronous marks using normalized autocorrelation,
    then refine each mark to a nearby waveform peak.
    """
    min_f0, max_f0 = 60, 400  # slightly narrower typical voiced speech band
    min_period = max(1, sample_rate // max_f0)
    max_period = max(min_period + 1, sample_rate // min_f0)

    frame_length = max_period * 2
    if len(signal) < frame_length:
        return np.array([], dtype=np.int64)

    pitch_marks = []
    position = 0

    while position + frame_length < len(signal):
        frame = signal[position:position + frame_length]
        frame = frame * np.hanning(len(frame))

        autocorr = correlate(frame, frame, mode="full")
        autocorr = autocorr[len(frame) - 1:]

        if autocorr[0] > 1e-10:
            autocorr = autocorr / autocorr[0]

        search_region = autocorr[min_period:max_period + 1]

        if len(search_region) == 0:
            break

        peak_value = float(np.max(search_region))
        best_period = min_period + int(np.argmax(search_region))

        # If confidence is weak, do not let period jump wildly.
        if peak_value < 0.30:
            if pitch_marks:
                prev_period = pitch_marks[-1] - pitch_marks[-2] if len(pitch_marks) >= 2 else best_period
                period = int(np.clip(prev_period, min_period, max_period))
            else:
                period = int(np.clip(sample_rate // 140, min_period, max_period))
        else:
            period = best_period

        if not pitch_marks:
            rough_mark = position + frame_length // 2
            refined_mark = _refine_mark_to_peak(signal, rough_mark, max(1, period // 3))
            pitch_marks.append(refined_mark)
        else:
            rough_mark = pitch_marks[-1] + period
            if rough_mark >= len(signal):
                break
            refined_mark = _refine_mark_to_peak(signal, rough_mark, max(1, period // 3))

            # Guard against duplicates / backward movement after refinement
            if refined_mark <= pitch_marks[-1]:
                refined_mark = min(len(signal) - 1, pitch_marks[-1] + max(1, period // 2))

            if refined_mark < len(signal):
                pitch_marks.append(refined_mark)

        position += period

    return np.array(pitch_marks, dtype=np.int64)


def _local_periods(pitch_marks: np.ndarray) -> np.ndarray:
    """
    Estimate a local pitch period at each mark.
    """
    if len(pitch_marks) < 2:
        return np.array([160], dtype=np.int64)

    diffs = np.diff(pitch_marks)
    diffs = np.maximum(diffs, 1)

    periods = np.empty(len(pitch_marks), dtype=np.int64)
    periods[0] = diffs[0]
    periods[-1] = diffs[-1]

    for i in range(1, len(pitch_marks) - 1):
        periods[i] = max(1, (diffs[i - 1] + diffs[i]) // 2)

    return periods


def _build_output_mark_positions(
    pitch_marks: np.ndarray,
    periods: np.ndarray,
    output_length: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build output pitch-mark positions using period accumulation rather than
    evenly spaced linspace interpolation.

    Returns:
        source_indices: which input pitch mark to use for each output grain
        output_positions: where to place each output grain center
    """
    if len(pitch_marks) < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    source_indices = []
    output_positions = []

    if scale >= 1.0:
        # Stretching: repeat source marks as needed by advancing output position
        # with a scaled local period, while cycling source marks more gently.
        out_pos = int(round(pitch_marks[0] * scale))
        src_idx = 0

        while out_pos < output_length and src_idx < len(pitch_marks):
            source_indices.append(src_idx)
            output_positions.append(out_pos)

            local_period = max(1, periods[src_idx])
            out_pos += int(round(local_period * scale))

            # Advance source index more slowly during stretching
            src_idx += 1

        # If needed, keep reusing the final few marks to fill tail
        tail_idx = max(0, len(pitch_marks) - 2)
        while out_pos < output_length:
            source_indices.append(tail_idx)
            output_positions.append(out_pos)
            local_period = max(1, periods[tail_idx])
            out_pos += int(round(local_period * scale))
    else:
        # Compression: place grains more closely and skip through source marks faster
        out_pos = int(round(pitch_marks[0] * scale))
        src_float = 0.0
        src_step = 1.0 / scale  # > 1, so we skip marks when compressing

        while out_pos < output_length:
            src_idx = int(round(src_float))
            if src_idx >= len(pitch_marks):
                break

            source_indices.append(src_idx)
            output_positions.append(out_pos)

            local_period = max(1, periods[src_idx])
            out_pos += max(1, int(round(local_period * scale)))
            src_float += src_step

    return np.array(source_indices, dtype=np.int64), np.array(output_positions, dtype=np.int64)


def _extract_grain(signal: np.ndarray, pitch_marks: np.ndarray, idx: int) -> tuple[np.ndarray, int]:
    """
    Extract a grain centered on pitch_marks[idx], with boundaries based on
    neighboring pitch marks.

    Returns:
        grain: windowed waveform grain
        center_offset: index inside grain corresponding to the pitch mark center
    """
    current = pitch_marks[idx]

    if idx == 0:
        prev_mark = max(0, current - (pitch_marks[idx + 1] - current))
    else:
        prev_mark = pitch_marks[idx - 1]

    if idx == len(pitch_marks) - 1:
        next_mark = min(len(signal) - 1, current + (current - pitch_marks[idx - 1]))
    else:
        next_mark = pitch_marks[idx + 1]

    left_len = max(1, current - prev_mark)
    right_len = max(1, next_mark - current)

    grain_start = max(0, current - left_len)
    grain_end = min(len(signal), current + right_len)

    grain = signal[grain_start:grain_end].copy()
    if len(grain) < 4:
        return np.array([], dtype=np.float64), 0

    window = np.hanning(len(grain))
    grain *= window

    center_offset = current - grain_start
    return grain, center_offset


def _psola_synthesize(signal: np.ndarray, pitch_marks: np.ndarray, scale: float) -> np.ndarray:
    """
    Overlap-add synthesis with time-scaled pitch mark positions.
    """
    output_length = max(1, int(round(len(signal) * scale)))
    periods = _local_periods(pitch_marks)

    source_indices, output_positions = _build_output_mark_positions(
        pitch_marks, periods, output_length, scale
    )

    if len(source_indices) == 0:
        return _resample_linear(signal, scale)

    # Padding protects final grains near the output boundary
    pad = int(max(periods) * 4) if len(periods) > 0 else 1024
    output = np.zeros(output_length + pad, dtype=np.float64)
    weight = np.zeros_like(output)

    for src_idx, out_pos in zip(source_indices, output_positions):
        grain, center_offset = _extract_grain(signal, pitch_marks, src_idx)
        if len(grain) == 0:
            continue

        out_start = out_pos - center_offset
        out_end = out_start + len(grain)

        grain_start = 0
        grain_end = len(grain)

        if out_start < 0:
            grain_start = -out_start
            out_start = 0

        if out_end > len(output):
            trim = out_end - len(output)
            grain_end -= trim
            out_end = len(output)

        if grain_end <= grain_start or out_end <= out_start:
            continue

        grain_slice = grain[grain_start:grain_end]

        # Use a matching synthesis weight for normalization
        synth_window = np.hanning(len(grain_slice))
        if len(synth_window) != len(grain_slice):
            continue

        output[out_start:out_end] += grain_slice
        weight[out_start:out_end] += synth_window

    nonzero = weight > 1e-8
    output[nonzero] /= weight[nonzero]

    output = output[:output_length]

    # Remove tiny DC bias that can sometimes build up
    if output.size > 0:
        output = output - np.mean(output)

    return output


def _resample_linear(signal: np.ndarray, scale: float) -> np.ndarray:
    """
    Simple linear interpolation fallback for edge cases.
    """
    output_length = int(round(len(signal) * scale))
    if output_length <= 0:
        return np.array([], dtype=signal.dtype)

    indices = np.linspace(0, len(signal) - 1, output_length)
    return np.interp(indices, np.arange(len(signal)), signal)