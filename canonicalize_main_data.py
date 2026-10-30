#!/usr/bin/env python3
"""Exclude CLT 026 and make Main Data poses right-foot canonical.

Reflection is around each row's bbox center, not the full video canvas:
    x_new[j] = 2*bbox_cx - x_old[left_right_partner(j)]
    xc_new[j] = -xc_old[left_right_partner(j)]
The paired y/yc values are exchanged without changing their signs. This is
equivalent to the existing hip-centered training reflection after recentering.
Raw canonical coordinates are model inputs, not overlays on the original video.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = ("EPL", "Women", "CLT")
JOINT_PAIRS = ((5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
SELECTED = (5, 6, 11, 12, 13, 14, 15, 16)
SWAP = {a: b for pair in JOINT_PAIRS for a, b in (pair, pair[::-1])}
DIRECTION_SWAP = {"L": "R", "R": "L", "C": "C"}
OUTPUT = ROOT / "runs/main_data_right_canonical"
EXCLUDED = ROOT / "Main Data/excluded/CLT_kick_026-C-L.csv"


def missing(value):
    return value is None or value.strip().lower() in ("", "nan", "na", "null", "none")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read(data):
    lines = data.decode("utf-8").splitlines(keepends=True)
    reader = csv.DictReader(lines)
    rows = list(reader)
    assert len(rows)+1 == len(lines), "Multiline CSV fields are not supported"
    assert all(None not in row and None not in row.values() for row in rows)
    return lines, reader.fieldnames, rows


def serialize(row, fields, newline):
    stream = io.StringIO(newline="")
    csv.DictWriter(stream, fieldnames=fields, lineterminator=newline).writerow(row)
    return stream.getvalue()


def canonicalize_row(row):
    foot = row["r_or_l"].strip().upper()
    assert foot in ("L", "R"), f"Unknown kicking foot: {foot!r}"
    if foot == "R":
        return dict(row)
    assert not missing(row["bbox_cx"]), "A reflection axis is required for raw coordinates"
    center = Decimal(row["bbox_cx"])
    assert center.is_finite()
    result = dict(row)
    for joint, partner in SWAP.items():
        for suffix in ("x", "y", "xc", "yc"):
            source = row[f"kp_{partner}_{suffix}"]
            dest = f"kp_{joint}_{suffix}"
            if missing(source) or suffix in ("y", "yc"):
                result[dest] = source
            elif suffix == "x":
                value = Decimal(source)
                assert value.is_finite()
                result[dest] = format(2*center-value, "f")
            else:
                value = Decimal(source)
                assert value.is_finite()
                result[dest] = format(-value, "f")
    label = row["kick_direction"].strip().upper()
    assert label in DIRECTION_SWAP, f"Unexpected direction: {label!r}"
    result["kick_direction"] = DIRECTION_SWAP[label]
    result["r_or_l"] = "R"
    return result


def prepare(dataset, before):
    lines, fields, rows = read(before)
    newline = "\r\n" if lines[0].endswith("\r\n") else "\n"
    new_lines = [lines[0]]
    new_rows = []
    excluded_lines = [lines[0]]
    excluded_rows = []
    audit = []
    left_before = sum(r["r_or_l"].strip().upper() == "L" for r in rows)
    selected_ids = set()
    changed_points = 0
    for index, row in enumerate(rows):
        if dataset == "CLT" and row["kick_id"] == "026-C-L.mp4":
            excluded_lines.append(lines[index+1])
            excluded_rows.append(row)
            continue
        new_row = canonicalize_row(row)
        flipped = row["r_or_l"].strip().upper() == "L"
        new_rows.append(new_row)
        new_lines.append(serialize(new_row, fields, newline) if flipped else lines[index+1])
        if flipped:
            selected_ids.add(row["kick_id"])
            audit.append({"dataset": dataset, "input_csv_line": index+2,
                          "output_csv_line": len(new_rows)+1, "kick_id": row["kick_id"],
                          "frame": row["frame"], "track_id": row.get("track_id", ""),
                          "kicker_track_id_main": row.get("kicker_track_id_main", ""),
                          "original_r_or_l": row["r_or_l"], "canonical_r_or_l": "R",
                          "original_kick_direction": row["kick_direction"],
                          "canonical_kick_direction": new_row["kick_direction"]})
            changed_points += sum(not missing(row[f"kp_{j}_x"]) for j in SWAP)
    after = "".join(new_lines).encode("utf-8")
    excluded = "".join(excluded_lines).encode("utf-8") if excluded_rows else b""
    report = {"dataset": dataset, "rows_before": len(rows), "rows_after": len(new_rows),
              "excluded_rows": len(excluded_rows), "left_rows_before": left_before,
              "mirrored_rows": len(audit), "mirrored_kicks": len(selected_ids),
              "mirrored_kick_ids": sorted(selected_ids),
              "valid_joint_points_in_mirrored_rows": changed_points,
              "direction_counts_before": dict(Counter(r["kick_direction"] for r in rows)),
              "direction_counts_after": dict(Counter(r["kick_direction"] for r in new_rows)),
              "foot_counts_after": dict(Counter(r["r_or_l"] for r in new_rows)),
              "sha256_before": sha256(before), "sha256_after": sha256(after)}
    return after, excluded, audit, report


def validate(before, after, dataset):
    old_lines, old_fields, old_rows = read(before)
    lines, fields, rows = read(after)
    assert fields == old_fields
    expected = [r for r in old_rows
                if not (dataset == "CLT" and r["kick_id"] == "026-C-L.mp4")]
    expected_lines = [old_lines[i+1] for i, r in enumerate(old_rows)
                      if not (dataset == "CLT" and r["kick_id"] == "026-C-L.mp4")]
    assert len(expected) == len(rows)
    selected_missing = 0
    max_formula_error = 0.0
    right_rows_preserved = 0
    raw_nonpositive = 0
    nonpositive_rows = 0
    before_nonpositive = sum(
        not missing(row[f"kp_{j}_{axis}"]) and float(row[f"kp_{j}_{axis}"]) <= 0
        for row in expected for j in SELECTED for axis in ("x", "y"))
    altered_allowed = {"r_or_l", "kick_direction"} | {
        f"kp_{j}_{suffix}" for j in SWAP for suffix in ("x", "y", "xc", "yc")}
    for i, (old, new) in enumerate(zip(expected, rows)):
        assert new["r_or_l"] == "R"
        assert not {c for c in fields if old[c] != new[c]} - altered_allowed
        assert new == canonicalize_row(old)
        if old["r_or_l"].strip().upper() == "R":
            assert lines[i+1] == expected_lines[i], "An original right-foot row changed"
            right_rows_preserved += 1
        else:
            # A second geometric reflection (with semantic swaps) must restore
            # the original numeric observations and all missingness locations.
            temp = dict(new, r_or_l="L")
            restored = canonicalize_row(temp)
            for j in SWAP:
                for suffix in ("x", "y", "xc", "yc"):
                    c = f"kp_{j}_{suffix}"
                    assert missing(restored[c]) == missing(old[c])
                    if not missing(old[c]):
                        assert Decimal(restored[c]) == Decimal(old[c])
            assert restored["kick_direction"] == old["kick_direction"]
        if any(missing(new[f"kp_{j}_{suffix}"]) for j in SELECTED for suffix in ("x", "y", "xc", "yc")):
            selected_missing += 1
        if any(not missing(new[f"kp_{j}_{axis}"]) and float(new[f"kp_{j}_{axis}"]) <= 0
               for j in SELECTED for axis in ("x", "y")):
            nonpositive_rows += 1
        for j in SWAP:
            for axis, center in (("x", "bbox_cx"), ("y", "bbox_cy")):
                raw, centered = new[f"kp_{j}_{axis}"], new[f"kp_{j}_{axis}c"]
                assert missing(raw) == missing(centered)
                if not missing(raw):
                    assert math.isfinite(float(raw)) and math.isfinite(float(centered))
                    if j in SELECTED:
                        raw_nonpositive += float(raw) <= 0
                    if not missing(new[center]):
                        max_formula_error = max(max_formula_error,
                            abs(float(raw)-float(new[center])-float(centered)))
    assert selected_missing == 0, f"Missing selected joints in {dataset}"
    assert raw_nonpositive == before_nonpositive, f"New nonpositive raw coordinates in {dataset}"
    assert max_formula_error <= 1e-5
    # Foot metadata makes the normalization idempotent.
    assert all(canonicalize_row(r) == r for r in rows)
    return {"dataset": dataset, "rows": len(rows), "columns": len(fields),
            "selected8_missing_rows": selected_missing,
            "right_foot_rows_preserved_byte_for_byte": right_rows_preserved,
            "existing_nonpositive_raw_coordinate_values": before_nonpositive,
            "nonpositive_raw_coordinate_rows": nonpositive_rows,
            "new_nonpositive_coordinate_values": 0,
            "geometry_roundtrip_exact": True, "idempotent": True,
            "max_centering_formula_error": max_formula_error}


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as f:
            f.write(data)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def execute(apply=False, verify=False):
    manifest_path = OUTPUT / "summary.json"
    if verify:
        manifest = json.loads(manifest_path.read_text())
        verification = []
        for report in manifest["datasets"]:
            ds = report["dataset"]
            name = f"{ds}_penalty_pose_keypoints.csv"
            before = (OUTPUT / "backups" / name).read_bytes()
            after = (ROOT / "Main Data" / name).read_bytes()
            assert sha256(before) == report["sha256_before"]
            assert sha256(after) == report["sha256_after"]
            verification.append(validate(before, after, ds))
        assert sha256(EXCLUDED.read_bytes()) == manifest["excluded_clt26_sha256"]
        atomic_write(OUTPUT / "verification.json", json.dumps(verification, indent=2).encode())
        print(json.dumps(verification, indent=2))
        return
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for report in manifest["datasets"]:
            path = ROOT / "Main Data" / f'{report["dataset"]}_penalty_pose_keypoints.csv'
            assert sha256(path.read_bytes()) == report["sha256_after"], (
                "This transformation was already applied, but a source CSV changed later; "
                "existing backups will not be overwritten.")
        print(json.dumps({"status": "already_applied", "summary": str(manifest_path)}))
        return
    assert not EXCLUDED.exists(), "CLT 026 backup already exists"
    plans = []
    audit = []
    verifications = []
    excluded = b""
    for dataset in DATASETS:
        path = ROOT / "Main Data" / f"{dataset}_penalty_pose_keypoints.csv"
        before = path.read_bytes()
        after, removed, row_audit, report = prepare(dataset, before)
        verifications.append(validate(before, after, dataset))
        if removed:
            assert dataset == "CLT"
            excluded = removed
        audit.extend(row_audit)
        plans.append((path, before, after, report))
    assert plans[2][3]["excluded_rows"] == 116
    summary = {"canonical_foot": "R", "reflection_axis": "per-row bbox_cx",
               "raw_x_formula": "2*bbox_cx - opposite_joint_x",
               "centered_x_formula": "-opposite_joint_xc",
               "y_formula": "opposite_joint_y (same for yc)",
               "joint_pairs": JOINT_PAIRS, "direction_mapping_for_left_foot": DIRECTION_SWAP,
               "camera_direction": "original source camera metadata preserved",
               "excluded_kick": "CLT:026-C-L.mp4", "excluded_clt26_sha256": sha256(excluded),
               "datasets": [p[3] for p in plans],
               "originals": "backups/<dataset>_penalty_pose_keypoints.csv",
               "warning": "Canonical raw coordinates use a bbox-centered reflection and must not be overlaid on the unmirrored source video."}
    print(json.dumps(summary, indent=2))
    if not apply:
        return
    # Preflight all transformations and validations before any data mutation.
    backup_dir = OUTPUT / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path, before, _, _ in plans:
        backup = backup_dir / path.name
        assert not backup.exists(), f"Backup already exists: {backup}"
        assert path.read_bytes() == before, f"Concurrent source edit: {path}"
    for path, before, _, _ in plans:
        backup = backup_dir / path.name
        shutil.copy2(path, backup)
        assert backup.read_bytes() == before
    atomic_write(EXCLUDED, excluded)
    audit_stream = io.StringIO(newline="")
    writer = csv.DictWriter(audit_stream, fieldnames=list(audit[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(audit)
    atomic_write(OUTPUT / "mirrored_rows.csv", audit_stream.getvalue().encode())
    for path, before, after, _ in plans:
        if before != after:
            atomic_write(path, after)
    atomic_write(manifest_path, json.dumps(summary, indent=2).encode())
    atomic_write(OUTPUT / "verification.json", json.dumps(verifications, indent=2).encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    execute(args.apply, args.verify)
