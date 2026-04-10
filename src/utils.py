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

        if reader.fieldnames is None:
            raise ValueError("CSV spec file is empty or missing a header row")

        fieldnames = set(reader.fieldnames)

        required_columns = {"t1", "t2"}
        missing = required_columns - fieldnames
        if missing:
            raise ValueError(f"CSV spec missing required columns: {sorted(missing)}")

        if "scale" not in fieldnames and "target_duration" not in fieldnames:
            raise ValueError("CSV spec must include at least one of: scale, target_duration")

        for i, row in enumerate(reader, start=2):  # line 1 = header
            try:
                t1 = float(row["t1"])
                t2 = float(row["t2"])
            except Exception as exc:
                raise ValueError(f"Invalid t1/t2 value in CSV row {i}: {row}") from exc

            scale_raw = row.get("scale", "")
            target_raw = row.get("target_duration", "")

            scale = None
            target_duration = None

            if scale_raw is not None and scale_raw.strip() != "":
                try:
                    scale = float(scale_raw)
                except Exception as exc:
                    raise ValueError(f"Invalid scale value in CSV row {i}: {row}") from exc

            if target_raw is not None and target_raw.strip() != "":
                try:
                    target_duration = float(target_raw)
                except Exception as exc:
                    raise ValueError(f"Invalid target_duration value in CSV row {i}: {row}") from exc

            if scale is not None and target_duration is not None:
                raise ValueError(f"CSV row {i} must specify either scale or target_duration, not both")

            if scale is None and target_duration is None:
                raise ValueError(f"CSV row {i} must specify one of scale or target_duration")

            edits.append({
                "t1": t1,
                "t2": t2,
                "scale": scale,
                "target_duration": target_duration,
            })

    if not edits:
        raise ValueError("CSV spec file contains no edit rows")

    return edits