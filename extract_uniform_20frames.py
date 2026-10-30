#!/usr/bin/env python3
"""Select 20 observed frames, including start and last touch, per dataset/kick.

One eligible camera sequence is selected per kick, in original appearance
order. The normalized inputs are already interpolated and canonicalized; no
coordinate or label is modified and no new pose observation is synthesized.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT/"Main Data/Combined_penalty_pose_keypoints_normalized.csv"
DEFAULT_OUTPUT = ROOT/"Main Data/Combined_penalty_pose_keypoints_normalized_20frames.csv"
SAMPLE_SIZE = 20
ADDED_COLUMNS = ("sequence_id", "sample_20_idx")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_number(row):
    try:
        value = float(row["frame"].strip())
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value.is_integer() else None


def is_touch(row):
    try:
        return float(row["last_touch"].strip()) == 1
    except (KeyError, AttributeError, TypeError, ValueError):
        return False


def source_track(row):
    # CLT detection track IDs can be zero/blank/change when detection fails.
    # Its stable designated-kicker ID, like Women's, identifies the sequence.
    column = "track_id" if row["dataset"] == "EPL" else "kicker_track_id_main"
    return row.get(column, "")


def candidate_sequences(rows):
    """Separate cameras and overlapping tracker timelines before sampling.

    A tracker handoff with increasing, nonoverlapping frame ranges is merged
    within the same camera. A second track whose frame range overlaps/reset
    starts a separate candidate; e.g. two shots sharing one Women kick_id.
    """
    cameras = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        if not row["dataset"].strip() or not row["kick_id"].strip():
            raise ValueError(f"Missing dataset/kick_id at CSV line {index+2}")
        key = (row["dataset"], row["kick_id"], row.get("camera_direction", ""))
        cameras[key][source_track(row)].append(index)
    candidates = []
    for key, tracks in cameras.items():
        segments = []
        for track, indices in tracks.items():
            frames = [frame_number(rows[i]) for i in indices]
            valid = all(f is not None for f in frames)
            low, high = (min(frames), max(frames)) if valid else (None, None)
            # Only a forward handoff is merged; overlapping timelines stay
            # separate even when their camera metadata happen to be identical.
            if (segments and valid and segments[-1]["max_frame"] is not None
                    and low > segments[-1]["max_frame"]):
                segment = segments[-1]
                segment["indices"].extend(indices)
                segment["tracks"].append(track)
                segment["max_frame"] = high
            else:
                segments.append({"key": key, "tracks": [track], "indices": list(indices),
                                 "min_frame": low, "max_frame": high})
        candidates.extend(segments)
    # Source-row appearance determines which eligible sequence is preferred.
    candidates.sort(key=lambda c: min(c["indices"]))
    return candidates


def uniform_observed_indices(frames, count=SAMPLE_SIZE):
    """Find unique chronological observations nearest uniform TIME targets.

    The endpoints are fixed. A small dynamic program chooses the interior
    count-2 frames with minimum total absolute error from linspace targets.
    Unlike rounding row indices, this respects gaps in the actual frame numbers.
    It also cannot select the same observation twice when a large gap exists.
    """
    frames = np.asarray(frames, dtype=float)
    n = len(frames)
    if count < 2 or n < count or not np.all(np.diff(frames) > 0):
        raise ValueError("Need enough strictly increasing observed frames")
    targets = np.linspace(frames[0], frames[-1], count)
    previous = np.full(n, np.inf)
    previous[0] = 0
    parents = np.full((count, n), -1, dtype=int)
    for step in range(1, count-1):
        current = np.full(n, np.inf)
        best_cost, best_index = np.inf, -1
        for j in range(1, n-1):
            if previous[j-1] < best_cost:
                best_cost, best_index = previous[j-1], j-1
            if step <= j <= n-count+step and best_index >= 0:
                current[j] = best_cost+abs(frames[j]-targets[step])
                parents[step, j] = best_index
        previous = current
    selected = [n-1]
    cursor = int(np.argmin(previous))
    if not np.isfinite(previous[cursor]):
        raise ValueError("No unique uniform frame assignment exists")
    for step in range(count-2, 0, -1):
        selected.append(cursor)
        cursor = int(parents[step, cursor])
    selected.append(0)
    selected.reverse()
    assert len(selected) == count and len(set(selected)) == count
    assert selected[0] == 0 and selected[-1] == n-1
    return selected, targets


def assess_candidate(candidate, rows):
    """Eligibility is based on unique available frames THROUGH last_touch."""
    dataset, kick, camera = candidate["key"]
    record = {"dataset": dataset, "kick_id": kick, "camera_direction": camera,
              "source_track_ids": candidate["tracks"], "input_rows": len(candidate["indices"]),
              "first_source_csv_line": min(candidate["indices"])+2}
    indices = candidate["indices"]
    if any(frame_number(rows[i]) is None for i in indices):
        return None, dict(record, status="invalid_frame_number")
    touches = [i for i in indices if is_touch(rows[i])]
    touch_frames = {frame_number(rows[i]) for i in touches}
    if not touches:
        return None, dict(record, status="missing_last_touch")
    if len(touch_frames) != 1:
        return None, dict(record, status="ambiguous_last_touch", touch_frames=sorted(touch_frames))
    touch_frame = next(iter(touch_frames))
    # There may be multiple rows for one physical timestamp. Keep the first
    # observed row, but prefer the explicit marker for the endpoint timestamp.
    by_frame = {}
    duplicate_lines = []
    for index in sorted(indices):
        frame = frame_number(rows[index])
        if frame > touch_frame:
            continue
        if frame in by_frame:
            duplicate_lines.append(index+2)
            if frame == touch_frame and is_touch(rows[index]) and not is_touch(rows[by_frame[frame]]):
                by_frame[frame] = index
        else:
            by_frame[frame] = index
    frames = sorted(by_frame)
    record.update(first_frame=frames[0], last_touch_frame=touch_frame,
                  available_unique_frames_through_touch=len(frames),
                  duplicate_timestamp_rows=duplicate_lines,
                  rows_after_last_touch=sum(frame_number(rows[i]) > touch_frame for i in indices))
    # Exactly 20 is excluded too, per the explicit "20 or fewer" rule.
    if len(frames) <= SAMPLE_SIZE:
        return None, dict(record, status="20_or_fewer_frames_through_touch")
    selection, targets = uniform_observed_indices(frames)
    chosen_frames = [frames[i] for i in selection]
    selected_source_indices = [by_frame[frame] for frame in chosen_frames]
    assert is_touch(rows[selected_source_indices[-1]])
    sequence_id = "|".join((dataset, kick, camera or "NA", "+".join(candidate["tracks"])))
    record.update(status="eligible", sequence_id=sequence_id,
                  selected_frames=chosen_frames, target_uniform_frames=targets.tolist(),
                  selected_source_csv_lines=[i+2 for i in selected_source_indices],
                  max_uniform_target_error_frames=max(abs(a-b) for a, b in zip(chosen_frames, targets)))
    return selected_source_indices, record


def write_csv(path, fields, output_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(output_rows)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def validate_output(path, fields, original_rows, selected_records):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == fields+list(ADDED_COLUMNS)
        saved = list(reader)
    assert len(saved) == SAMPLE_SIZE*len(selected_records)
    keys = set()
    for offset, record in enumerate(selected_records):
        key = (record["dataset"], record["kick_id"])
        assert key not in keys
        keys.add(key)
        block = saved[offset*SAMPLE_SIZE:(offset+1)*SAMPLE_SIZE]
        assert all((r["dataset"], r["kick_id"]) == key for r in block)
        times = [frame_number(row) for row in block]
        assert all(b > a for a, b in zip(times, times[1:]))
        assert times == record["selected_frames"]
        assert times[0] == record["first_frame"]
        assert times[-1] == record["last_touch_frame"] and is_touch(block[-1])
        assert sum(is_touch(r) for r in block) == 1
        assert record["available_unique_frames_through_touch"] > SAMPLE_SIZE
        for step, (row, line) in enumerate(zip(block, record["selected_source_csv_lines"])):
            assert row["sample_20_idx"] == str(step)
            assert row["sequence_id"] == record["sequence_id"]
            assert {c: row[c] for c in fields} == original_rows[line-2]
    return {"exactly_20_rows_per_dataset_kick": True, "strict_chronological_order": True,
            "first_and_last_touch_endpoints_included": True,
            "one_sequence_per_dataset_kick": True,
            "selected_original_values_unchanged": True, "duplicate_selected_timestamps": 0}


def extract_uniform_20frames(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT, *, overwrite=False):
    input_path, output_path = Path(input_path), Path(output_path)
    report_path = output_path.with_name(output_path.stem+"_sampling_report.json")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different")
    if not overwrite and (output_path.exists() or report_path.exists()):
        raise FileExistsError("Output already exists; use overwrite=True explicitly")
    before_hash = digest(input_path)
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields, rows = list(reader.fieldnames or []), list(reader)
    if not {"dataset", "kick_id", "frame", "last_touch"} <= set(fields):
        raise ValueError("Missing dataset/kick_id/frame/last_touch columns")
    if set(ADDED_COLUMNS) & set(fields):
        raise ValueError("Input already contains sampling columns")
    if len(fields) != len(set(fields)) or any(None in r or None in r.values() for r in rows):
        raise ValueError("Malformed CSV")
    candidates = candidate_sequences(rows)
    by_kick = defaultdict(list)
    for candidate in candidates:
        by_kick[candidate["key"][:2]].append(candidate)
    outputs, records, all_assessments, excluded = [], [], [], []
    for key, choices in by_kick.items():
        chosen = False
        for candidate in choices:
            source_indices, record = assess_candidate(candidate, rows)
            if source_indices is not None and not chosen:
                record["status"] = "selected"
                records.append(record)
                for step, index in enumerate(source_indices):
                    output_row = dict(rows[index])
                    output_row.update(sequence_id=record["sequence_id"], sample_20_idx=str(step))
                    outputs.append(output_row)
                chosen = True
            elif source_indices is not None:
                record["status"] = "eligible_alternative_not_selected"
            all_assessments.append(record)
        if not chosen:
            reasons = [r["status"] for r in all_assessments
                       if (r["dataset"], r["kick_id"]) == key]
            excluded.append({"dataset": key[0], "kick_id": key[1], "reasons": reasons})
    if digest(input_path) != before_hash:
        raise RuntimeError("Input changed during sampling")
    write_csv(output_path, fields+list(ADDED_COLUMNS), outputs)
    validation = validate_output(output_path, fields, rows, records)
    assert digest(input_path) == before_hash
    report = {"input": str(input_path), "output": str(output_path),
              "input_sha256": before_hash, "output_sha256": digest(output_path),
              "input_rows": len(rows), "input_kicks": len(by_kick),
              "selected_kicks": len(records), "excluded_kicks_count": len(excluded),
              "output_rows": len(outputs),
              "selected_kicks_by_dataset": dict(Counter(r["dataset"] for r in records)),
              "excluded_kicks_by_dataset": dict(Counter(r["dataset"] for r in excluded)),
              "excluded_kicks": excluded, "selected_sequences": records,
              "all_candidate_assessments": all_assessments,
              "validation": dict(validation, input_file_unchanged=True),
              "sampling": "Start + 18 observed frames nearest uniform frame-number targets + last_touch",
              "eligibility": "More than 20 unique observed frames from start through last_touch inclusive",
              "sequence_choice": "First eligible sequence in original source-row appearance order",
              "new_columns": list(ADDED_COLUMNS)}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("output", "input_kicks", "selected_kicks",
                     "excluded_kicks_count", "output_rows", "selected_kicks_by_dataset",
                     "excluded_kicks_by_dataset", "excluded_kicks", "validation")},
                     indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    extract_uniform_20frames(arguments.input, arguments.output, overwrite=arguments.overwrite)
