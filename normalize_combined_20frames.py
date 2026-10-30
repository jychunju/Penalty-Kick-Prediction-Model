#!/usr/bin/env python3
"""Restore source bbox fields and create right-foot-canonical normalized features."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

JOINTS = (11, 12, 13, 14, 15, 16)
COORDS = tuple(f"kp_{j}_{axis}" for j in JOINTS for axis in ("x", "y"))
SWAP = (1, 0, 3, 2, 5, 4)
FPS = {"CLT": 30.0, "EPL": 25.0, "Women": 22.12}


def read(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def number(value):
    try:
        result = float((value or "").strip())
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def frame(row):
    value = number(row.get("frame"))
    return None if value is None else int(round(value))


def row_distance(combined, source):
    errors = []
    for key in COORDS:
        a, b = number(combined.get(key)), number(source.get(key))
        if a is not None and b is not None:
            errors.append(abs(a - b))
    return (float(np.mean(errors)), len(errors)) if errors else (float("inf"), 0)


def candidate_cost(combined_rows, source_rows):
    by_frame = defaultdict(list)
    for row in source_rows:
        if frame(row) is not None:
            by_frame[frame(row)].append(row)
    distances = []
    for row in combined_rows:
        if frame(row) is None:
            continue
        scored = [row_distance(row, candidate) for candidate in by_frame.get(frame(row), [])]
        scored = [score for score in scored if score[1] >= 4]
        if scored:
            distances.append(min(score[0] for score in scored))
    if not distances:
        return float("inf"), 0
    return float(np.mean(distances)), len(distances)


def source_groups(dataset, combined_id, source_rows):
    grouped = defaultdict(list)
    if dataset == "CLT":
        source_id = combined_id.removeprefix("clt_")
        for row in source_rows:
            if row.get("kick_id") == source_id:
                grouped[(source_id, row.get("kicker_track_id_main") or row.get("track_id") or "")].append(row)
    elif dataset == "EPL":
        source_id = combined_id.removeprefix("epl_")
        for row in source_rows:
            if row.get("kick_id") == source_id:
                # A single camera sequence may cross tracker-ID boundaries.
                grouped[(source_id, row.get("camera_direction", ""))].append(row)
    else:
        for row in source_rows:
            grouped[(row.get("kick_id", ""),)].append(row)
    return grouped


def select_source_group(dataset, combined_id, combined_rows, source_rows):
    candidates = []
    for key, rows in source_groups(dataset, combined_id, source_rows).items():
        cost, coverage = candidate_cost(combined_rows, rows)
        if coverage:
            candidates.append((cost, -coverage, key, rows))
    if not candidates:
        raise ValueError(f"No source group matched {dataset}:{combined_id}")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best = candidates[0]
    gap = float("inf") if len(candidates) == 1 else candidates[1][0] - best[0]
    return best[2], best[3], best[0], gap, -best[1]


def match_source_rows(combined_rows, source_rows):
    by_frame = defaultdict(list)
    for row in source_rows:
        if frame(row) is not None:
            by_frame[frame(row)].append(row)
    matched = []
    for row in combined_rows:
        candidates = by_frame.get(frame(row), []) if frame(row) is not None else source_rows
        scored = [(row_distance(row, source), source) for source in candidates]
        scored = [item for item in scored if item[0][1] >= 4]
        if not scored:
            # Bbox values are frame-level even when the source pose has too
            # many missing joints to compare. A unique same-frame row is safe.
            if frame(row) is not None and len(candidates) == 1:
                matched.append(candidates[0]); continue
            raise ValueError(f"Could not match source row at frame {row.get('frame')!r}")
        scored.sort(key=lambda item: item[0][0])
        matched.append(scored[0][1])
    return matched


def restore_bbox(matched, source_group):
    names = ("bbox_cx", "bbox_cy", "bbox_w", "bbox_h")
    targets = np.asarray([frame(row) for row in matched], dtype=float)
    result = np.asarray([[number(row.get(name)) for name in names] for row in matched], dtype=float)
    interpolated = ~np.isfinite(result).all(axis=1)
    for axis, name in enumerate(names):
        known = [(frame(row), number(row.get(name))) for row in source_group]
        known = [(f, value) for f, value in known if f is not None and value is not None]
        if len(known) < 2:
            raise ValueError(f"Insufficient valid {name} values for bbox interpolation")
        # Collapse duplicate frames deterministically before interpolation.
        by_frame = defaultdict(list)
        for f, value in known:
            by_frame[f].append(value)
        xp = np.asarray(sorted(by_frame), dtype=float)
        fp = np.asarray([np.median(by_frame[int(f)]) for f in xp], dtype=float)
        missing = ~np.isfinite(result[:, axis])
        result[missing, axis] = np.interp(targets[missing], xp, fp)
    return result, interpolated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("penalty_kicks_20frames_combined_with_foot.csv"))
    p.add_argument("--output", type=Path, default=Path("penalty_kicks_20frames_combined_normalized.csv"))
    p.add_argument("--clt", type=Path, default=Path("merged_full_data_001_030_trimmed.csv"))
    p.add_argument("--epl", type=Path, default=Path("kicker_pose_keypoints.csv"))
    p.add_argument("--women", type=Path, default=Path("penalty_pose_keypoints.csv"))
    args = p.parse_args()
    rows, fields = read(args.input)
    sources = {"CLT": read(args.clt)[0], "EPL": read(args.epl)[0], "Women": read(args.women)[0]}
    sequences = defaultdict(list)
    for row in rows:
        sequences[(row["dataset"], row["kick_id"])].append(row)

    output_rows = []
    exclusions = []
    for (dataset, kick_id), sequence in sequences.items():
        sequence.sort(key=lambda row: int(row["seq_20_idx"]))
        key, source_group, score, gap, coverage = select_source_group(
            dataset, kick_id, sequence, sources[dataset])
        try:
            matched = match_source_rows(sequence, source_group)
        except ValueError as error:
            raise ValueError(f"{dataset}:{kick_id}: {error}") from error
        ordered = sorted(zip(sequence, matched), key=lambda pair: frame(pair[1]))
        sequence = [pair[0] for pair in ordered]
        matched = [pair[1] for pair in ordered]
        resolved_frames = np.asarray([frame(source) for source in matched], dtype=float)
        bbox, bbox_interpolated = restore_bbox(matched, source_group)
        if not np.isfinite(bbox).all() or np.any(bbox[:, 3] <= 0):
            raise ValueError(f"Invalid bbox in {dataset}:{kick_id}")
        scale = float(np.median(bbox[:, 3]))
        positions = np.empty((len(sequence), len(JOINTS), 2), dtype=float)
        for t, row in enumerate(sequence):
            for j, joint in enumerate(JOINTS):
                positions[t, j] = [number(row[f"kp_{joint}_xc"]), number(row[f"kp_{joint}_yc"])]
        if not np.isfinite(positions).all():
            raise ValueError(f"Invalid centered pose in {dataset}:{kick_id}")
        positions /= scale
        mirrored = sequence[0]["r_or_l"].strip().upper() == "L"
        if mirrored:
            positions[..., 0] *= -1
            positions = positions[:, SWAP, :]
        velocity = np.zeros_like(positions)
        dt = np.diff(resolved_frames) / FPS[dataset]
        if np.any(dt <= 0):
            exclusions.append({"dataset": dataset, "kick_id": kick_id,
                               "reason": "duplicate_or_non_increasing_source_frame",
                               "resolved_source_frames": resolved_frames.astype(int).tolist()})
            continue
        velocity[1:] = np.diff(positions, axis=0) / dt[:, None, None]
        actual_direction = sequence[0]["kick_direction"].strip().upper()
        canonical_direction = ({"L": "R", "R": "L"}.get(actual_direction, actual_direction)
                               if mirrored else actual_direction)
        for t, (row, source) in enumerate(zip(sequence, matched)):
            out = dict(row)
            out["original_seq_20_idx"] = row["seq_20_idx"]
            out["seq_20_idx"] = t + 1
            out.update({
                "fps": FPS[dataset], "source_kick_id": key[0],
                "resolved_source_frame": int(resolved_frames[t]),
                "bbox_cx": bbox[t, 0], "bbox_cy": bbox[t, 1],
                "bbox_w": bbox[t, 2], "bbox_h": bbox[t, 3],
                "bbox_interpolated": int(bbox_interpolated[t]),
                "normalization_scale": scale,
                "mirrored_to_right_foot": int(mirrored),
                "kick_direction_canonical": canonical_direction,
                "source_match_mean_abs_error": score,
                "source_match_next_best_gap": gap,
                "source_match_coverage_frames": coverage,
            })
            for j, joint in enumerate(JOINTS):
                out[f"kp_{joint}_x_norm"] = positions[t, j, 0]
                out[f"kp_{joint}_y_norm"] = positions[t, j, 1]
                out[f"kp_{joint}_vx_per_sec"] = velocity[t, j, 0]
                out[f"kp_{joint}_vy_per_sec"] = velocity[t, j, 1]
            output_rows.append(out)

    added = ["original_seq_20_idx", "fps", "source_kick_id", "resolved_source_frame", "bbox_cx", "bbox_cy", "bbox_w", "bbox_h", "bbox_interpolated",
             "normalization_scale", "mirrored_to_right_foot", "kick_direction_canonical",
             "source_match_mean_abs_error", "source_match_next_best_gap", "source_match_coverage_frames"]
    for joint in JOINTS:
        added += [f"kp_{joint}_x_norm", f"kp_{joint}_y_norm", f"kp_{joint}_vx_per_sec", f"kp_{joint}_vy_per_sec"]
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + added)
        writer.writeheader(); writer.writerows(output_rows)
    temporary.replace(args.output)
    exclusion_path = args.output.with_name(args.output.stem + "_exclusions.json")
    exclusion_path.write_text(json.dumps(exclusions, indent=2), encoding="utf-8")
    print(f"wrote {args.output}: {len(output_rows)} rows, {len(output_rows)//20} sequences; excluded {len(exclusions)}")


if __name__ == "__main__":
    main()
