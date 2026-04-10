import csv
import pytest
from src.utils import load_edit_spec

def test_load_edit_spec_rejects_both_scale_and_target_duration(tmp_path):
    csv_path = tmp_path / "edits.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t1", "t2", "scale", "target_duration"])
        writer.writeheader()
        writer.writerow({"t1": 0.2, "t2": 0.4, "scale": 1.5, "target_duration": 0.5})

    with pytest.raises(ValueError, match="either scale or target_duration, not both"):
        load_edit_spec(str(csv_path))

def test_load_edit_spec_rejects_missing_scale_and_target_duration(tmp_path):
    csv_path = tmp_path / "edits.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t1", "t2", "scale", "target_duration"])
        writer.writeheader()
        writer.writerow({"t1": 0.2, "t2": 0.4, "scale": "", "target_duration": ""})

    with pytest.raises(ValueError, match="must specify one of scale or target_duration"):
        load_edit_spec(str(csv_path))