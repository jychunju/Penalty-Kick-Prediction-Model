#!/usr/bin/env python3
"""Full-sequence pelvis centering with one median torso scale per kick.

This stage assumes interpolation and right-foot canonicalization are finished.
It never selects frames, sorts rows, rotates skeletons, swaps joints, changes
labels, or normalizes x/y by separate bounding-box dimensions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np


JOINT_IDS = tuple(range(5, 17))
INPUT_COORD_COLUMNS = tuple(f"kp_{j}_{axis}" for j in JOINT_IDS for axis in ("xc", "yc"))
NORMALIZED_COLUMNS = tuple(f"kp_{j}_{axis}" for j in JOINT_IDS for axis in ("xn", "yn"))
DIAGNOSTIC_COLUMNS = ("pelvis_xc", "pelvis_yc", "shoulder_xc", "shoulder_yc",
                      "torso_length", "clip_torso_scale")
ADDED_COLUMNS = NORMALIZED_COLUMNS + DIAGNOSTIC_COLUMNS
DEFAULT_INPUTS = ("CLT_penalty_pose_keypoints.csv", "EPL_penalty_pose_keypoints.csv",
                  "Women_penalty_pose_keypoints.csv", "Combined_penalty_pose_keypoints.csv")


def file_sha256(path):
    """Record provenance and detect a source file changed during processing."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pose_csv(path):
    """Keep every original field as a string, including whitespace/ID formatting."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"Missing or duplicate CSV headers: {path}")
        required = {"kick_id", *INPUT_COORD_COLUMNS}
        absent = sorted(required-set(fields))
        if absent:
            raise ValueError(f"Missing required columns in {path}: {absent}")
        conflicts = sorted(set(ADDED_COLUMNS) & set(fields))
        if conflicts:
            raise ValueError(f"Already-normalized input would overwrite columns: {conflicts}")
        rows = list(reader)
    for line_number, row in enumerate(rows, start=2):
        if None in row or None in row.values():
            raise ValueError(f"Malformed CSV row at line {line_number}: {path}")
    return fields, rows


def parse_coordinate(value):
    """Non-numeric, NaN and infinite coordinates become NaN only in new arrays.

    The original CSV cell is never edited. A centered coordinate of zero is
    legitimate and is not interpreted as missing.
    """
    try:
        parsed = float(value.strip())
    except (AttributeError, TypeError, ValueError):
        return np.nan
    return parsed if np.isfinite(parsed) else np.nan


def read_coordinate_array(rows):
    """Return (all rows, all 12 joints, x/y), without using the frame column."""
    values = [[parse_coordinate(row[c]) for c in INPUT_COORD_COLUMNS] for row in rows]
    return np.asarray(values, dtype=np.float64).reshape(len(rows), len(JOINT_IDS), 2)


def group_row_indices(rows, fields, group_columns=None):
    """Group by kick ID, with dataset protection for a merged input.

    Indices refer to the existing row order. We never sort frames or emit rows
    grouped by ID: interleaved sequences retain exactly their original order.
    Track IDs and camera directions do not subdivide a kick's scale because the
    requested rule is one scale per entire kick_id.
    """
    if group_columns is None:
        group_columns = ("dataset", "kick_id") if "dataset" in fields else ("kick_id",)
    group_columns = tuple(group_columns)
    if not group_columns or "kick_id" not in group_columns:
        raise ValueError("Grouping must include kick_id")
    if not set(group_columns) <= set(fields):
        raise ValueError(f"Unknown group columns: {group_columns}")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        key = tuple(row[c] for c in group_columns)
        if any(not value.strip() for value in key):
            raise ValueError(f"Empty group identifier at CSV line {index+2}: {key}")
        groups[key].append(index)
    return group_columns, dict(groups)


def compute_frame_geometry(coordinates):
    """Compute per-frame hip/shoulder midpoints and Euclidean torso length."""
    with np.errstate(invalid="ignore", over="ignore"):
        # Multiplying each operand by 0.5 avoids overflow in the midpoint sum.
        pelvis = coordinates[:, 11-5, :]*0.5 + coordinates[:, 12-5, :]*0.5
        shoulders = coordinates[:, 5-5, :]*0.5 + coordinates[:, 6-5, :]*0.5
        delta = shoulders-pelvis
        # hypot is the stable equivalent of sqrt(dx**2 + dy**2).
        torso = np.hypot(delta[:, 0], delta[:, 1])
    return pelvis, shoulders, torso


def compute_clip_scales(torso, groups, group_columns):
    """Use ALL valid frame lengths for one fixed scale per kick.

    Invalid lengths (NaN, infinity, zero or negative) do not enter the median.
    An entirely invalid kick is retained with NaN scale, not a bbox fallback.
    """
    row_scales = np.full(len(torso), np.nan, dtype=np.float64)
    clip_info = []
    failed = []
    for key, index_list in groups.items():
        indices = np.asarray(index_list, dtype=int)
        lengths = torso[indices]
        valid = lengths[np.isfinite(lengths) & (lengths > 0)]
        identifiers = dict(zip(group_columns, key))
        # Half-sum median is stable even if two positive middle values are huge.
        if len(valid):
            ordered = np.sort(valid)
            middle = len(ordered)//2
            scale = (float(ordered[middle]) if len(ordered) % 2 else
                     float(0.5*ordered[middle-1] + 0.5*ordered[middle]))
            row_scales[indices] = scale
        else:
            scale = np.nan
            failed.append(identifiers)
            warnings.warn(
                f"No valid positive torso length for {identifiers}; all rows retained "
                "with NaN scale/normalized coordinates. No bbox fallback used.",
                RuntimeWarning, stacklevel=2,
            )
        clip_info.append({**identifiers, "number_of_frames": len(indices),
                          "valid_torso_frames": len(valid),
                          "invalid_torso_frames": len(indices)-len(valid),
                          "clip_torso_scale": scale})
    return row_scales, clip_info, failed


def normalize_coordinates(coordinates, pelvis, row_scales):
    """Translate per frame, then divide both axes by the SAME clip-level D."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        normalized = (coordinates-pelvis[:, None, :])/row_scales[:, None, None]
    # Failed kicks or unusable input cells stay NaN; infinity is never exported
    # as if it were a usable normalized coordinate.
    normalized[~np.isfinite(normalized)] = np.nan
    return normalized


def validate_normalization(normalized, pelvis, torso, row_scales, groups):
    """Check pelvis origin, scale constancy and the full-sequence median rule."""
    with np.errstate(invalid="ignore"):
        hip_center = normalized[:, 11-5, :]*0.5 + normalized[:, 12-5, :]*0.5
    finite_errors = np.abs(hip_center[np.isfinite(hip_center)])
    max_pelvis_error = float(finite_errors.max()) if finite_errors.size else None
    if max_pelvis_error is not None and max_pelvis_error > 1e-10:
        raise AssertionError(f"Normalized pelvis center error too large: {max_pelvis_error}")
    scales_constant = True
    for indices in groups.values():
        values = row_scales[indices]
        if np.isnan(values).all():
            continue
        if not np.isfinite(values).all() or not np.all(values == values[0]):
            scales_constant = False
            break
        valid = torso[indices]
        valid = valid[np.isfinite(valid) & (valid > 0)]
        if not len(valid) or not np.isclose(values[0], np.median(valid), rtol=1e-14, atol=0):
            raise AssertionError("Scale was not computed from the entire valid sequence")
    if not scales_constant:
        raise AssertionError("More than one scale occurred within a kick")
    return {"max_normalized_pelvis_absolute_error": max_pelvis_error,
            "all_kicks_have_one_fixed_scale": scales_constant,
            "failed_kicks_use_nan_scale_for_all_rows": True,
            "normalized_nan_count": int(np.isnan(normalized).sum()),
            "normalized_inf_count": int(np.isinf(normalized).sum()),
            "frames_with_unverifiable_pelvis": int((~np.isfinite(hip_center).all(axis=1)).sum())}


def random_clip_examples(rows, normalized, row_scales, groups, group_columns,
                         sample_count=5, random_seed=42):
    """Sample only for printing diagnostics; every data row still goes to CSV."""
    keys = list(groups)
    rng = np.random.default_rng(random_seed)
    choices = rng.choice(len(keys), size=min(sample_count, len(keys)), replace=False)
    examples = []
    for choice in choices:
        key = keys[int(choice)]
        indices = groups[key]
        first = indices[0]  # First ORIGINAL row, never sorted by frame number.
        pose = normalized[first]
        examples.append({**dict(zip(group_columns, key)),
                         "number_of_frames": len(indices),
                         "clip_torso_scale": float(row_scales[first]),
                         "first_row_frame": rows[first].get("frame"),
                         "first_frame_pelvis_normalized":
                             ((pose[11-5]+pose[12-5])*0.5).tolist(),
                         "first_frame_left_shoulder_normalized": pose[5-5].tolist(),
                         "first_frame_right_shoulder_normalized": pose[6-5].tolist(),
                         "first_frame_left_ankle_normalized": pose[15-5].tolist(),
                         "first_frame_right_ankle_normalized": pose[16-5].tolist()})
    return examples


def json_safe(value):
    """Use JSON null for unavailable statistics instead of nonstandard NaN."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_normalized_csv(path, fields, rows, normalized, pelvis, shoulders,
                         torso, row_scales):
    """Append new columns, emitting each original row in its original position."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    added_values = np.column_stack((normalized.reshape(len(rows), 24), pelvis,
                                   shoulders, torso, row_scales))
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields+list(ADDED_COLUMNS), lineterminator="\n")
            writer.writeheader()
            for index, original in enumerate(rows):
                output = dict(original)
                for column, value in zip(ADDED_COLUMNS, added_values[index]):
                    # An empty new cell represents NaN for standard CSV readers.
                    output[column] = format(float(value), ".17g") if np.isfinite(value) else ""
                writer.writerow(output)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def verify_saved_csv(path, original_fields, original_rows, normalized, row_scales):
    """Re-read the actual output and verify EVERY original cell and row order."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != original_fields+list(ADDED_COLUMNS):
            raise AssertionError("Output column order changed")
        count = 0
        for index, saved in enumerate(reader):
            if index >= len(original_rows):
                raise AssertionError("Unexpected output rows")
            if {c: saved[c] for c in original_fields} != original_rows[index]:
                raise AssertionError(f"An original field/row changed at CSV line {index+2}")
            values = np.asarray([parse_coordinate(saved[c]) for c in NORMALIZED_COLUMNS])
            if not np.array_equal(values, normalized[index].reshape(24), equal_nan=True):
                raise AssertionError("Saved normalized coordinate precision changed")
            saved_scale = parse_coordinate(saved["clip_torso_scale"])
            expected_scale = row_scales[index]
            if not (saved_scale == expected_scale or np.isnan(saved_scale) and np.isnan(expected_scale)):
                raise AssertionError("Saved clip scale changed")
            count += 1
    if count != len(original_rows):
        raise AssertionError("Output lost rows")


def print_validation_report(report):
    """Print requested file statistics and deterministic random kick examples."""
    stats = report["validation"]
    print(f"\nFile: {report['output_path']}")
    print(f"Total rows: {report['total_rows']}; kick_ids: {report['unique_kick_id_count']}; "
          f"clip groups: {report['clip_group_count']}")
    print(f"Grouping: {report['group_columns']}")
    print(f"Failed kick count: {len(report['failed_kicks'])}; list: {report['failed_kicks']}")
    print(f"Clip scale min / median / max: {report['clip_scale_statistics']}")
    print(f"Normalized pelvis max absolute error: {stats['max_normalized_pelvis_absolute_error']}")
    print(f"Normalized NaNs: {stats['normalized_nan_count']}; infs: {stats['normalized_inf_count']}")
    print(f"One fixed scale per kick: {stats['all_kicks_have_one_fixed_scale']}")
    print("Original columns and row order preserved: True")
    print("Random kick examples (first row in original order; null means unavailable):")
    for example in report["random_examples"]:
        print(json.dumps(json_safe(example), ensure_ascii=False, allow_nan=False))


def normalize_pose_csv(input_path, output_path, *, group_columns=None,
                       sample_count=5, random_seed=42, overwrite=False):
    """Normalize one CSV and write a JSON validation report alongside it.

    Single-dataset CSV: group by kick_id. Combined CSV with a dataset column:
    group by (dataset, kick_id), preventing different datasets' IDs from mixing.
    Failed kicks are recorded in report['failed_kicks'] and retained unchanged
    in original columns, with unavailable new values left blank (CSV NaN).
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must be separate from the source CSV")
    report_path = output_path.with_name(output_path.stem+"_normalization_report.json")
    if not overwrite and (output_path.exists() or report_path.exists()):
        raise FileExistsError(f"Output/report already exists: {output_path}; use overwrite=True explicitly")
    if sample_count < 0:
        raise ValueError("sample_count must be nonnegative")
    before_hash = file_sha256(input_path)
    fields, rows = read_pose_csv(input_path)
    coordinates = read_coordinate_array(rows)
    grouping, groups = group_row_indices(rows, fields, group_columns)
    pelvis, shoulders, torso = compute_frame_geometry(coordinates)
    scales, clips, failures = compute_clip_scales(torso, groups, grouping)
    normalized = normalize_coordinates(coordinates, pelvis, scales)
    validation = validate_normalization(normalized, pelvis, torso, scales, groups)
    valid_clip_scales = np.asarray([c["clip_torso_scale"] for c in clips
                                   if np.isfinite(c["clip_torso_scale"])])
    scale_stats = {"min": float(valid_clip_scales.min()) if valid_clip_scales.size else None,
                   "median": float(np.median(valid_clip_scales)) if valid_clip_scales.size else None,
                   "max": float(valid_clip_scales.max()) if valid_clip_scales.size else None}
    report = {"input_path": str(input_path), "output_path": str(output_path),
              "source_sha256": before_hash, "total_rows": len(rows),
              "unique_kick_id_count": len({row["kick_id"] for row in rows}),
              "clip_group_count": len(groups), "group_columns": list(grouping),
              "original_column_count": len(fields), "added_columns": list(ADDED_COLUMNS),
              "invalid_input_coordinate_count": int(np.isnan(coordinates).sum()),
              "clip_scale_statistics": scale_stats, "failed_kicks": failures,
              "clips": clips, "validation": validation,
              "random_examples": random_clip_examples(rows, normalized, scales, groups, grouping,
                                                       sample_count, random_seed),
              "method": "per-frame pelvis centering / full-sequence median positive finite torso length",
              "no_frame_selection": True, "no_rotation": True,
              "no_bbox_scaling": True, "no_flip_or_label_changes": True}
    if file_sha256(input_path) != before_hash:
        raise RuntimeError("Input changed during normalization")
    write_normalized_csv(output_path, fields, rows, normalized, pelvis, shoulders, torso, scales)
    verify_saved_csv(output_path, fields, rows, normalized, scales)
    if file_sha256(input_path) != before_hash:
        raise RuntimeError("Source CSV was modified")
    report["output_sha256"] = file_sha256(output_path)
    report["validation"].update(original_columns_unchanged=True, row_order_unchanged=True,
                                source_file_unchanged=True, saved_values_verified=True)
    report = json_safe(report)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print_validation_report(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path,
                        help="Defaults to the three Main Data source CSVs and the Combined CSV")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    inputs = args.inputs or [root/"Main Data"/name for name in DEFAULT_INPUTS]
    for input_path in inputs:
        directory = args.output_dir if args.output_dir is not None else input_path.parent
        output_path = directory/(input_path.stem+"_normalized.csv")
        normalize_pose_csv(input_path, output_path, sample_count=args.samples,
                           random_seed=args.seed, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
