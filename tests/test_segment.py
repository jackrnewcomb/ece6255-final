import numpy as np
import pytest

from src.segment import extract_segments, seconds_to_sample_index


def test_extract_segments_basic():
    samples = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)
    sample_rate = 2

    prefix, segment, suffix = extract_segments(samples, sample_rate, 1.0, 2.0)

    assert np.array_equal(prefix, np.array([0, 1], dtype=np.float32))
    assert np.array_equal(segment, np.array([2, 3], dtype=np.float32))
    assert np.array_equal(suffix, np.array([4, 5], dtype=np.float32))


def test_extract_segments_rejects_out_of_bounds_t2():
    samples = np.arange(6, dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds audio duration"):
        extract_segments(samples, 2, 1.0, 4.0)


def test_extract_segments_rejects_empty_after_rounding():
    samples = np.arange(8, dtype=np.float32)
    with pytest.raises(ValueError, match="empty"):
        extract_segments(samples, 4, 0.10, 0.11)


def test_seconds_to_sample_index_rejects_negative_time():
    with pytest.raises(ValueError, match="nonnegative"):
        seconds_to_sample_index(-0.1, 16000, 100)