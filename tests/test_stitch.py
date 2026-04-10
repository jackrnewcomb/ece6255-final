import numpy as np
import pytest

from src.stitch import stitch_segments


def test_stitch_segments_basic():
    prefix = np.array([0, 1], dtype=np.float32)
    segment = np.array([9, 9], dtype=np.float32)
    suffix = np.array([4, 5], dtype=np.float32)

    result = stitch_segments(prefix, segment, suffix)

    assert np.array_equal(result, np.array([0, 1, 9, 9, 4, 5], dtype=np.float32))


def test_stitch_segments_rejects_empty_processed_segment():
    with pytest.raises(ValueError, match="cannot be empty"):
        stitch_segments(np.array([1], dtype=np.float32), np.array([], dtype=np.float32), np.array([2], dtype=np.float32))