#!/usr/bin/env python3
"""Merge the three current Main Data CSVs without modifying their sources."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Main Data"
DATASETS = ("EPL", "CLT", "Women")
OUTPUT = DATA / "Combined_penalty_pose_keypoints.csv"
REPORT = DATA / "Combined_penalty_pose_keypoints_summary.json"
FIELDS = ["dataset", "kick_id", "frame", "fps", "kicker", "game_or_training",
          "last_touch", "kicker_track_id_main", "track_id", "camera_direction",
          "r_or_l", "kick_direction", "goal", "bbox_cx", "bbox_cy", "bbox_w", "bbox_h"]
FIELDS += [f"kp_{j}_{axis}" for j in range(5, 17) for axis in ("x", "y")]
FIELDS += [f"kp_{j}_{axis}" for j in range(5, 17) for axis in ("xc", "yc")]
INTEGER_FIELDS = {"frame", "kicker", "last_touch", "kicker_track_id_main", "track_id", "goal"}
LABEL_FIELDS = {"camera_direction", "r_or_l", "kick_direction"}
COORD_FIELDS = {c for c in FIELDS if c.startswith(("kp_", "bbox_"))}
SELECTED = (5, 6, 11, 12, 13, 14, 15, 16)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert all(None not in r and None not in r.values() for r in rows)
        return reader.fieldnames, rows


def normalize_value(column, value):
    text = (value or "").strip()
    if not text or text.lower() in ("nan", "na", "null", "none"):
        return ""
    if column in INTEGER_FIELDS:
        number = Decimal(text)
        assert number.is_finite() and number == number.to_integral_value(), (column, value)
        return str(int(number))
    if column in LABEL_FIELDS:
        text = text.upper()
        assert text in ({"L", "R"} if column == "r_or_l" else {"L", "R", "C"}), (column, value)
    if column in COORD_FIELDS or column == "fps":
        assert Decimal(text).is_finite(), (column, value)
    if column == "fps":
        assert Decimal(text) > 0
    if column in ("last_touch", "goal"):
        assert text in ("0", "1")
    return text


def sources():
    return [(dataset, DATA / f"{dataset}_penalty_pose_keypoints.csv") for dataset in DATASETS]


def verify():
    summary = json.loads(REPORT.read_text())
    assert digest(OUTPUT) == summary["output_sha256"]
    fields, merged = read(OUTPUT)
    assert fields == FIELDS and fields[0] == "dataset"
    assert len(fields) == len(set(fields)) == 65
    assert len(merged) == summary["total_rows"]
    with OUTPUT.open(newline="", encoding="utf-8") as stream:
        assert all(len(row) == len(FIELDS) for row in csv.reader(stream))
    cursor = 0
    counts = Counter()
    verified_cells = 0
    for dataset, path in sources():
        assert digest(path) == summary["sources"][dataset]["sha256"], f"Source changed: {dataset}"
        old_fields, old_rows = read(path)
        for old in old_rows:
            new = merged[cursor]
            assert new["dataset"] == dataset
            original_columns = {"goal" if c == "Goal" else c for c in old_fields}
            for old_col, old_value in old.items():
                column = "goal" if old_col == "Goal" else old_col
                value = (old_value or "").strip()
                if not value or value.lower() in ("nan", "na", "null", "none"):
                    assert new[column] == ""
                elif column in INTEGER_FIELDS:
                    assert Decimal(value) == Decimal(new[column])
                    assert str(int(Decimal(new[column]))) == new[column]
                elif column in LABEL_FIELDS:
                    assert new[column] == value.upper()
                else:
                    assert new[column] == value
                verified_cells += 1
            assert all(new[c] == "" for c in set(FIELDS)-original_columns-{"dataset"})
            counts[dataset] += 1
            cursor += 1
    assert cursor == len(merged)
    assert dict(counts) == summary["rows_by_dataset"]
    selected_missing = sum(any(not row[f"kp_{j}_{suffix}"] for j in SELECTED
                               for suffix in ("x", "y", "xc", "yc")) for row in merged)
    assert selected_missing == 0
    result = {"rows": len(merged), "columns": len(fields), "rows_by_dataset": dict(counts),
              "verified_source_cells": verified_cells, "source_files_unchanged": True,
              "row_order_preserved": True, "all_source_columns_preserved": True,
              "selected8_blank_missing_rows": selected_missing}
    print(json.dumps(result, indent=2))
    return result


def merge():
    assert not OUTPUT.exists() and not REPORT.exists(), "Output already exists; use --verify to inspect it"
    merged = []
    info = {}
    conversions = Counter()
    for dataset, path in sources():
        before_hash = digest(path)
        fields, rows = read(path)
        standard_fields = ["goal" if c == "Goal" else c for c in fields]
        assert len(standard_fields) == len(set(standard_fields))
        assert set(standard_fields) <= set(FIELDS), "Unmapped source columns"
        for row in rows:
            output_row = dict.fromkeys(FIELDS, "")
            output_row["dataset"] = dataset
            for original_column, value in row.items():
                column = "goal" if original_column == "Goal" else original_column
                normalized = normalize_value(column, value)
                output_row[column] = normalized
                if normalized != value:
                    conversions[f"{dataset}:{column}"] += 1
            merged.append(output_row)
        info[dataset] = {"path": str(path.relative_to(ROOT)), "sha256": before_hash,
                         "rows": len(rows), "columns": len(fields),
                         "absent_columns_left_blank": [c for c in FIELDS[1:] if c not in standard_fields]}
        assert digest(path) == before_hash
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=OUTPUT.name+".", suffix=".tmp", dir=OUTPUT.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(merged)
        os.replace(temp_name, OUTPUT)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    summary = {"output": str(OUTPUT.relative_to(ROOT)), "output_sha256": digest(OUTPUT),
               "total_rows": len(merged), "columns": FIELDS,
               "rows_by_dataset": dict(Counter(row["dataset"] for row in merged)),
               "sources": info, "column_aliases": {"Goal": "goal"},
               "integer_columns": sorted(INTEGER_FIELDS),
               "value_format_changes": dict(conversions),
               "missing_values": dict(Counter({c: sum(not r[c] for r in merged) for c in FIELDS})),
               "rules": ["All source rows retained in EPL, CLT, Women order",
                         "Source kick_id strings and coordinate precision preserved",
                         "No deduplication or label/coordinate transformation during merge",
                         "Source-absent columns and whitespace-only cells remain blank",
                         "No inference of fps, camera direction, goal, or missing metadata",
                         "Use dataset with kick_id to distinguish different datasets"]}
    REPORT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify() if args.verify else merge()
