from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from src.interface import process_file


def _write_test_wav(path: Path, sample_rate: int = 8000, seconds: float = 1.0):
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wavfile.write(path, sample_rate, (data * 32767).astype(np.int16))


def test_process_file_happy_path(tmp_path: Path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    _write_test_wav(input_path)

    process_file(str(input_path), str(output_path), t1=0.2, t2=0.4, scale=1.0)

    assert output_path.exists()
    sample_rate, output = wavfile.read(output_path)
    assert sample_rate == 8000
    assert output.ndim == 1
    assert output.size > 0


def test_process_file_rejects_nonpositive_scale(tmp_path: Path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    _write_test_wav(input_path)

    with pytest.raises(ValueError, match="scale must be positive"):
        process_file(str(input_path), str(output_path), t1=0.2, t2=0.4, scale=0)


def test_process_file_rejects_missing_input(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        process_file(str(tmp_path / "missing.wav"), str(tmp_path / "out.wav"), t1=0.2, t2=0.4, scale=1.0)