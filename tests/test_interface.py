import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

import src.interface as interface_module
from src.interface import process_file, process_file_batch


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


def test_process_file_batch_happy_path(tmp_path: Path, monkeypatch):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    csv_path = tmp_path / "edits.csv"

    _write_test_wav(input_path, sample_rate=8000, seconds=1.0)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t1", "t2", "scale"])
        writer.writeheader()
        writer.writerow({"t1": 0.2, "t2": 0.4, "scale": 1.5})
        writer.writerow({"t1": 0.6, "t2": 0.8, "scale": 0.8})

    def fake_time_scale_psola(segment, sample_rate, scale):
        old_positions = np.linspace(0, len(segment) - 1, num=len(segment))
        new_length = max(1, int(round(len(segment) * scale)))
        new_positions = np.linspace(0, len(segment) - 1, num=new_length)
        return np.interp(new_positions, old_positions, segment).astype(np.float32)

    monkeypatch.setattr(interface_module, "time_scale_psola", fake_time_scale_psola)

    process_file_batch(str(input_path), str(output_path), str(csv_path))

    assert output_path.exists()
    output_sr, output_data = wavfile.read(output_path)
    assert output_sr == 8000
    assert output_data.ndim == 1
    assert output_data.size > 0

def test_process_file_target_duration_happy_path(tmp_path: Path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    _write_test_wav(input_path)

    process_file(
        str(input_path),
        str(output_path),
        t1=0.2,
        t2=0.4,
        target_duration=0.5,
    )

    assert output_path.exists()
    sample_rate, output = wavfile.read(output_path)
    assert sample_rate == 8000
    assert output.ndim == 1
    assert output.size > 0

def test_process_file_rejects_scale_and_target_duration_together(tmp_path: Path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    _write_test_wav(input_path)

    with pytest.raises(ValueError, match="either scale or target_duration"):
        process_file(
            str(input_path),
            str(output_path),
            t1=0.2,
            t2=0.4,
            scale=1.5,
            target_duration=0.5,
        )

def test_process_file_rejects_missing_scale_and_target_duration(tmp_path: Path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    _write_test_wav(input_path)

    with pytest.raises(ValueError, match="Must provide either scale or target_duration"):
        process_file(
            str(input_path),
            str(output_path),
            t1=0.2,
            t2=0.4,
        )

def test_process_file_rejects_nonpositive_target_duration(tmp_path: Path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    _write_test_wav(input_path)

    with pytest.raises(ValueError, match="target_duration must be positive"):
        process_file(
            str(input_path),
            str(output_path),
            t1=0.2,
            t2=0.4,
            target_duration=0.0,
        )

def test_process_file_batch_target_duration_happy_path(tmp_path: Path, monkeypatch):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    csv_path = tmp_path / "edits.csv"

    _write_test_wav(input_path, sample_rate=8000, seconds=1.0)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t1", "t2", "scale", "target_duration"])
        writer.writeheader()
        writer.writerow({"t1": 0.2, "t2": 0.4, "scale": "", "target_duration": 0.5})

    def fake_time_scale_psola(segment, sample_rate, scale):
        old_positions = np.linspace(0, len(segment) - 1, num=len(segment))
        new_length = max(1, int(round(len(segment) * scale)))
        new_positions = np.linspace(0, len(segment) - 1, num=new_length)
        return np.interp(new_positions, old_positions, segment).astype(np.float32)

    monkeypatch.setattr(interface_module, "time_scale_psola", fake_time_scale_psola)

    process_file_batch(str(input_path), str(output_path), str(csv_path))

    assert output_path.exists()
    output_sr, output_data = wavfile.read(output_path)
    assert output_sr == 8000
    assert output_data.ndim == 1
    assert output_data.size > 0

def test_process_file_batch_adjusts_later_edit_times_after_duration_change(tmp_path: Path, monkeypatch):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    csv_path = tmp_path / "edits.csv"

    _write_test_wav(input_path, sample_rate=8000, seconds=2.0)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t1", "t2", "scale"])
        writer.writeheader()
        writer.writerow({"t1": 0.2, "t2": 0.4, "scale": 2.0})   # grows by +0.2 s
        writer.writerow({"t1": 0.6, "t2": 0.8, "scale": 1.0})   # should be shifted to 0.8 - 1.0

    calls = []

    def fake_apply_single_edit(samples, sample_rate, t1, t2, scale=None, target_duration=None):
        calls.append(
            {
                "t1": t1,
                "t2": t2,
                "scale": scale,
                "target_duration": target_duration,
                "input_length": len(samples),
            }
        )

        resolved_scale = interface_module._resolve_scale(t1, t2, scale, target_duration)
        original_len = len(samples)
        segment_len = int(round((t2 - t1) * sample_rate))
        new_segment_len = int(round(segment_len * resolved_scale))
        new_total_len = original_len - segment_len + new_segment_len

        return np.zeros(new_total_len, dtype=np.float32)

    monkeypatch.setattr(interface_module, "_apply_single_edit", fake_apply_single_edit)

    process_file_batch(str(input_path), str(output_path), str(csv_path))

    assert len(calls) == 2

    # First edit uses original times unchanged
    assert calls[0]["t1"] == pytest.approx(0.2)
    assert calls[0]["t2"] == pytest.approx(0.4)

    # First edit expands 0.2 s -> 0.4 s, so later edits shift by +0.2 s
    assert calls[1]["t1"] == pytest.approx(0.8)
    assert calls[1]["t2"] == pytest.approx(1.0)

def test_process_file_batch_rejects_overlapping_original_time_edits(tmp_path: Path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    csv_path = tmp_path / "overlap.csv"

    _write_test_wav(input_path, sample_rate=8000, seconds=2.0)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t1", "t2", "scale"])
        writer.writeheader()
        writer.writerow({"t1": 0.2, "t2": 0.5, "scale": 1.2})
        writer.writerow({"t1": 0.4, "t2": 0.7, "scale": 0.8})  # overlaps previous edit

    with pytest.raises(ValueError, match="Overlapping batch edits are not supported"):
        process_file_batch(str(input_path), str(output_path), str(csv_path))

def test_process_file_batch_adjusts_later_edit_times_with_target_duration(tmp_path: Path, monkeypatch):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    csv_path = tmp_path / "edits.csv"

    _write_test_wav(input_path, sample_rate=8000, seconds=2.0)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t1", "t2", "scale", "target_duration"])
        writer.writeheader()
        writer.writerow({"t1": 0.2, "t2": 0.4, "scale": "", "target_duration": 0.5})  # +0.3 s shift
        writer.writerow({"t1": 0.6, "t2": 0.8, "scale": 1.0, "target_duration": ""})

    calls = []

    def fake_apply_single_edit(samples, sample_rate, t1, t2, scale=None, target_duration=None):
        calls.append(
            {
                "t1": t1,
                "t2": t2,
                "scale": scale,
                "target_duration": target_duration,
                "input_length": len(samples),
            }
        )

        resolved_scale = interface_module._resolve_scale(t1, t2, scale, target_duration)
        original_len = len(samples)
        segment_len = int(round((t2 - t1) * sample_rate))
        new_segment_len = int(round(segment_len * resolved_scale))
        new_total_len = original_len - segment_len + new_segment_len

        return np.zeros(new_total_len, dtype=np.float32)

    monkeypatch.setattr(interface_module, "_apply_single_edit", fake_apply_single_edit)

    process_file_batch(str(input_path), str(output_path), str(csv_path))

    assert len(calls) == 2
    assert calls[1]["t1"] == pytest.approx(0.9)
    assert calls[1]["t2"] == pytest.approx(1.1)