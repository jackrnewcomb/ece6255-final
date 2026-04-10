import csv
from pathlib import Path


def load_edit_spec(csv_path: str):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV spec file not found: {path}")
    if not path.is_file():
        raise ValueError(f"CSV spec path is not a file: {path}")

    edits = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {"t1", "t2", "scale"}
        if reader.fieldnames is None:
            raise ValueError("CSV spec file is empty or missing a header row")

        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV spec missing required columns: {sorted(missing)}")

        for i, row in enumerate(reader, start=2):  # line 1 = header
            try:
                t1 = float(row["t1"])
                t2 = float(row["t2"])
                scale = float(row["scale"])
            except Exception as exc:
                raise ValueError(f"Invalid numeric value in CSV row {i}: {row}") from exc

            edits.append({
                "t1": t1,
                "t2": t2,
                "scale": scale,
            })

    if not edits:
        raise ValueError("CSV spec file contains no edit rows")

    return edits