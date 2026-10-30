#!/usr/bin/env python3
"""Keep shoulders, hips, knees and ankles without changing rows or values."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "Main Data/Combined_penalty_pose_keypoints_normalized_20frames.csv"
DEFAULT_OUTPUT = ROOT / "Main Data/Combined_penalty_pose_keypoints_normalized_20frames_8joints.csv"
JOINTS = (5, 6, 11, 12, 13, 14, 15, 16)
KEYPOINT_COLUMN = re.compile(r"^kp_(\d+)_")


def extract_8_joints(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT):
    """Retain all coordinate representations of eight joints and all metadata.

    Values are read as strings to preserve coordinate precision and labels.
    This is only a column selection: no resampling, interpolation, sorting,
    renormalization, or label transformation is performed.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must differ")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path}")
    original_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    required = {f"kp_{joint}_{axis}n" for joint in JOINTS for axis in ("x", "y")}
    if not required.issubset(fields):
        raise ValueError(f"Missing normalized coordinates: {sorted(required-set(fields))}")
    if len(fields) != len(set(fields)) or any(None in row or None in row.values() for row in rows):
        raise ValueError("Malformed CSV")

    # Remove only the keypoint columns for joints outside the selected eight.
    # Keep pixel x/y, bbox-relative xc/yc, normalized xn/yn for selected joints,
    # and every non-keypoint column (including diagnostics and sampling IDs).
    retained = [column for column in fields
                if not (match := KEYPOINT_COLUMN.match(column))
                or int(match.group(1)) in JOINTS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=retained, lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row[column] for column in retained} for row in rows)

    # Read the saved artifact back: every retained cell and row order must match.
    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == retained
        saved = list(reader)
    assert saved == [{column: row[column] for column in retained} for row in rows]
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == original_hash
    present_joints = {int(match.group(1)) for column in retained
                      if (match := KEYPOINT_COLUMN.match(column))}
    assert present_joints == set(JOINTS)
    counts = Counter((row["dataset"], row["kick_id"]) for row in saved)
    assert all(count == 20 for count in counts.values())
    invalid_normalized = 0
    for row in saved:
        for column in required:
            try:
                valid = math.isfinite(float(row[column]))
            except (TypeError, ValueError):
                valid = False
            invalid_normalized += not valid
    summary = {
        "output": str(output_path), "joints": list(JOINTS),
        "rows": len(saved), "kicks": len(counts), "columns": len(retained),
        "normalized_coordinate_columns": len(required),
        "invalid_normalized_coordinates": invalid_normalized,
        "kicks_by_dataset": dict(Counter(dataset for dataset, kick in counts)),
        "removed_columns": [column for column in fields if column not in retained],
        "row_order_and_retained_values_unchanged": True,
        "input_file_unchanged": True,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    extract_8_joints(args.input, args.output)
