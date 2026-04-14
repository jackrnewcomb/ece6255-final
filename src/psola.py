
import numpy as np


def time_scale_psola(segment: np.ndarray, sample_rate: int, scale: float) -> np.ndarray:
    """
    Time-scale a 1D audio segment using a simple TD-PSOLA-style method.

    Parameters
    ----------
    segment : np.ndarray
        1D mono audio segment.
    sample_rate : int
        Audio sample rate in Hz.
    scale : float
        Duration scale factor.
        scale > 1.0 => longer
        scale < 1.0 => shorter

    Returns
    -------
    np.ndarray
        Time-scaled segment as a 1D array.

    Notes
    -----
    This is a simplified PSOLA implementation intended for voiced speech segments.
    If pitch estimation fails or the segment is too short, it falls back to linear
    interpolation resampling so the pipeline remains robust.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")

    segment = np.asarray(segment)
    if segment.ndim != 1:
        raise ValueError("segment must be a 1D numpy array")

    if len(segment) == 0:
        return segment.copy()

    original_dtype = segment.dtype
    x = segment.astype(np.float64, copy=False)

    target_length = max(1, int(round(len(x) * scale)))

    # Fast path for identity scaling
    if np.isclose(scale, 1.0, atol=1e-6):
        return segment.copy()

    # Too short for reliable PSOLA -> fallback
    if len(x) < max(32, sample_rate // 100):  # about 10 ms or minimum small size
        y = _linear_resample(x, target_length)
        return _cast_like_input(y, original_dtype)

    # Estimate pitch period
    pitch_period = _estimate_pitch_period_autocorr(x, sample_rate)

    # If pitch estimation fails, fallback so the system still works
    if pitch_period is None:
        y = _linear_resample(x, target_length)
        return _cast_like_input(y, original_dtype)

    # Build analysis marks
    analysis_marks = _make_analysis_marks(len(x), pitch_period, x)

    # Need enough marks to do meaningful overlap-add
    if len(analysis_marks) < 2:
        y = _linear_resample(x, target_length)
        return _cast_like_input(y, original_dtype)

    # Synthesis spacing controls new duration
    synthesis_period = pitch_period
    synthesis_marks = _make_synthesis_marks(target_length, pitch_period, synthesis_period)

    if len(synthesis_marks) == 0:
        y = _linear_resample(x, target_length)
        return _cast_like_input(y, original_dtype)

    y = _overlap_add_psola(
        x=x,
        analysis_marks=analysis_marks,
        synthesis_marks=synthesis_marks,
        pitch_period=pitch_period,
        target_length=target_length,
    )

    if len(y) != target_length:
        y = _fix_length(y, target_length)

    # Avoid clipping if overlap-add pushes amplitude above nominal range
    peak = np.max(np.abs(y)) if len(y) > 0 else 0.0
    if peak > 1.0:
        y = y / peak

    return _cast_like_input(y, original_dtype)


def _estimate_pitch_period_autocorr(
    x: np.ndarray,
    sample_rate: int,
    f0_min: float = 70.0,
    f0_max: float = 300.0,
) -> int | None:
    """
    Estimate a single global pitch period using autocorrelation.
    Returns pitch period in samples, or None if no reliable estimate is found.
    """
    if len(x) < 3:
        return None

    x = x - np.mean(x)
    energy = np.sum(x * x)
    if energy <= 1e-12:
        return None

    min_lag = max(1, int(sample_rate / f0_max))
    max_lag = min(len(x) - 1, int(sample_rate / f0_min))

    if min_lag >= max_lag:
        return None

    autocorr = np.correlate(x, x, mode="full")
    autocorr = autocorr[len(x) - 1:]  # nonnegative lags only

    search_region = autocorr[min_lag : max_lag + 1]
    if len(search_region) == 0:
        return None

    best_offset = int(np.argmax(search_region))
    best_lag = min_lag + best_offset

    # Simple reliability check using normalized autocorrelation peak
    normalized_peak = autocorr[best_lag] / (autocorr[0] + 1e-12)
    if normalized_peak < 0.2:
        return None

    return best_lag


def _make_analysis_marks(length: int, pitch_period: int, x: np.ndarray) -> np.ndarray:
    """
    Create approximately pitch-spaced analysis marks and snap them
    to nearby positive peaks for better phase consistency.
    """
    if pitch_period <= 0 or length <= 0:
        return np.array([], dtype=int)

    start = pitch_period
    stop = max(start, length - pitch_period)
    if start >= stop:
        return np.array([], dtype=int)

    predicted_marks = np.arange(start, stop, pitch_period, dtype=int)
    search_radius = max(1, pitch_period // 4)

    snapped_marks = []
    last_mark = -1

    for mark in predicted_marks:
        left = max(0, mark - search_radius)
        right = min(length, mark + search_radius + 1)
        if right <= left:
            continue

        local = x[left:right]

        # Prefer the strongest positive peak for phase consistency
        local_idx = int(np.argmax(local))
        snapped = left + local_idx

        if snapped <= last_mark:
            continue

        if snapped - pitch_period < 0 or snapped + pitch_period > length:
            continue

        snapped_marks.append(snapped)
        last_mark = snapped

    return np.array(snapped_marks, dtype=int)


def _make_synthesis_marks(
    target_length: int,
    analysis_period: int,
    synthesis_period: int,
) -> np.ndarray:
    """
    Create synthesis marks for the output timeline.
    """
    if target_length <= 0 or analysis_period <= 0 or synthesis_period <= 0:
        return np.array([], dtype=int)

    start = analysis_period
    stop = max(start, target_length - analysis_period)
    if start >= stop:
        return np.array([], dtype=int)

    return np.arange(start, stop, synthesis_period, dtype=int)


def _overlap_add_psola(
    x: np.ndarray,
    analysis_marks: np.ndarray,
    synthesis_marks: np.ndarray,
    pitch_period: int,
    target_length: int,
) -> np.ndarray:
    """
    Window grains centered at analysis marks and overlap-add them at synthesis marks.
    """
    grain_half_width = pitch_period
    grain_length = 2 * grain_half_width

    if grain_length < 2:
        return _linear_resample(x, target_length)

    window = np.hanning(grain_length)
    y = np.zeros(target_length + 2 * grain_half_width, dtype=np.float64)
    weight = np.zeros_like(y)

    num_analysis = len(analysis_marks)
    num_synthesis = len(synthesis_marks)

    used_grains = 0
    skipped_grains = 0

    for j, synth_mark in enumerate(synthesis_marks):
        # Map synthesis grain index to analysis grain index proportionally
        if num_synthesis == 1:
            analysis_idx = 0
        else:
            analysis_idx = int(round(j * (num_analysis - 1) / (num_synthesis - 1)))

        analysis_mark = int(analysis_marks[analysis_idx])

        src_start = analysis_mark - grain_half_width
        src_end = analysis_mark + grain_half_width
        dst_start = int(synth_mark - grain_half_width)
        dst_end = int(synth_mark + grain_half_width)

        if src_start < 0 or src_end > len(x):
            skipped_grains += 1
            continue
        if dst_start < 0 or dst_end > len(y):
            skipped_grains += 1
            continue

        grain = x[src_start:src_end]
        if len(grain) != grain_length:
            skipped_grains += 1
            continue

        grain_windowed = grain * window
        y[dst_start:dst_end] += grain_windowed
        weight[dst_start:dst_end] += window
        used_grains += 1

    nonzero = weight > 1e-8
    y[nonzero] /= weight[nonzero]

    print(f"[PSOLA] used_grains={used_grains}, skipped_grains={skipped_grains}")

    y = y[:target_length]
    return y


def _linear_resample(x: np.ndarray, target_length: int) -> np.ndarray:
    """
    Simple linear interpolation fallback.
    """
    if target_length <= 0:
        return np.array([], dtype=np.float64)

    if len(x) == 0:
        return np.zeros(target_length, dtype=np.float64)

    if len(x) == 1:
        return np.full(target_length, x[0], dtype=np.float64)

    old_positions = np.linspace(0.0, 1.0, num=len(x))
    new_positions = np.linspace(0.0, 1.0, num=target_length)
    return np.interp(new_positions, old_positions, x)


def _fix_length(x: np.ndarray, target_length: int) -> np.ndarray:
    """
    Trim or zero-pad to exact length.
    """
    if len(x) == target_length:
        return x
    if len(x) > target_length:
        return x[:target_length]

    y = np.zeros(target_length, dtype=x.dtype)
    y[: len(x)] = x
    return y


def _cast_like_input(x: np.ndarray, original_dtype: np.dtype) -> np.ndarray:
    """
    Cast output back to the general dtype family of the input.
    """
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        x = np.clip(x, info.min, info.max)
        return np.rint(x).astype(original_dtype)

    return x.astype(original_dtype, copy=False)