#!/usr/bin/env python3
"""Create a source-safe, normalized L/R six-joint dataset from three CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


JOINT_IDS = (11, 12, 13, 14, 15, 16)
JOINT_NAMES = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def group_rows(rows: list[dict[str, str]], grouping: tuple[str, ...]):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(column, "") for column in grouping)].append(row)
    return groups


def build_sample(rows: list[dict[str, str]], key: tuple[str, ...], source: str, frames: int,
                 label_mapping: dict[str, int], canonical_foot: str | None = None,
                 require_known_foot: bool = False):
    labels = {row.get("kick_direction", "").upper() for row in rows if row.get("kick_direction", "")}
    if len(labels) != 1 or next(iter(labels), "") not in label_mapping:
        return None, "invalid_or_unsupported_label"
    feet = {row.get("r_or_l", "").upper() for row in rows if row.get("r_or_l", "")}
    foot = next(iter(feet)) if len(feet) == 1 and next(iter(feet)) in {"L", "R"} else None
    if require_known_foot and foot is None:
        return None, "unknown_or_inconsistent_kicking_foot"
    try:
        parsed = [(int(float(row["frame"])), row) for row in rows]
    except (KeyError, ValueError):
        return None, "invalid_frame"
    parsed.sort(key=lambda item: item[0])
    frame_numbers = [frame for frame, _ in parsed]
    if len(frame_numbers) != len(set(frame_numbers)):
        return None, "duplicate_frame"
    touches = [frame for frame, row in parsed if row.get("last_touch") == "1"]
    if len(touches) != 1:
        return None, f"last_touch_count_{len(touches)}"
    touch = touches[0]
    selected = list(range(touch - frames, touch))
    by_frame = dict(parsed)
    if any(frame not in by_frame for frame in selected):
        return None, "incomplete_pre_contact_window"

    positions = np.empty((frames, len(JOINT_IDS), 2), dtype=np.float64)
    scales = np.empty(frames, dtype=np.float64)
    try:
        for t, frame in enumerate(selected):
            row = by_frame[frame]
            scales[t] = float(row["bbox_h"])
            for joint_index, joint_id in enumerate(JOINT_IDS):
                positions[t, joint_index] = (
                    float(row[f"kp_{joint_id}_x"]),
                    float(row[f"kp_{joint_id}_y"]),
                )
    except (KeyError, ValueError):
        return None, "missing_or_invalid_lower_body"
    # Ultralytics commonly emits (0, 0) for an unobserved keypoint. Treat any
    # non-positive raw coordinate as missing rather than as a real image point.
    if not np.isfinite(positions).all() or np.any(positions <= 0):
        return None, "missing_or_zero_lower_body"
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        return None, "invalid_bbox_scale"

    scale = float(np.median(scales))
    hip_center = (positions[:, 0] + positions[:, 1]) / 2
    normalized = (positions - hip_center[:, None, :]) / scale
    mirrored = bool(canonical_foot and foot != canonical_foot)
    if mirrored:
        # Reflect around the hip-centered y-axis, then restore semantic node
        # order so indices still mean left/right hip, knee, and ankle.
        normalized[..., 0] *= -1
        normalized = normalized[:, [1, 0, 3, 2, 5, 4], :]
    velocity = np.zeros_like(normalized)
    velocity[1:] = np.diff(normalized, axis=0)
    features = np.concatenate((normalized, velocity), axis=2).transpose(2, 0, 1)[..., None]
    original_label = next(iter(labels))
    label = ({"L": "R", "R": "L"}.get(original_label, original_label)
             if mirrored else original_label)
    raw_kick_id = key[0]
    return {
        "x": features.astype(np.float32),
        "y": label_mapping[label],
        "label": label,
        "original_label": original_label,
        "kicking_foot": foot,
        "mirrored_to_canonical_foot": mirrored,
        "source": source,
        "raw_kick_id": raw_kick_id,
        "group_id": f"{source}:{raw_kick_id}",
        "sequence_id": f"{source}:" + "|".join(key),
        "selected_frames": selected,
        "last_touch_frame": touch,
        "normalization_scale": scale,
    }, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("integrated_3csv_config.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text())
    label_mapping = {name: index for index, name in enumerate(config["label_names"])}
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)

    samples = []
    manifest = []
    file_info = {}
    for spec in config["sources"]:
        path = root / spec["csv"]
        rows = read_rows(path)
        groups = group_rows(rows, tuple(spec["group_by"]))
        included = 0
        reasons = Counter()
        for key, group in groups.items():
            sample, reason = build_sample(
                group, key, spec["name"], config["frames"], label_mapping,
                config.get("canonical_foot"), config.get("require_known_foot", False))
            record = {
                "source": spec["name"], "raw_kick_id": key[0],
                "sequence_id": f"{spec['name']}:" + "|".join(key),
                "group_id": f"{spec['name']}:{key[0]}",
                "included": int(sample is not None), "reason": reason or "included",
            }
            if sample is not None:
                record.update(kicking_foot=sample["kicking_foot"],
                              mirrored_to_canonical_foot=int(sample["mirrored_to_canonical_foot"]),
                              original_label=sample["original_label"], label=sample["label"])
            manifest.append(record)
            if sample is None:
                reasons[reason] += 1
            else:
                samples.append(sample); included += 1
        file_info[spec["name"]] = {
            "csv": spec["csv"], "sha256": sha256(path), "rows": len(rows),
            "candidate_sequences": len(groups), "included_sequences": included,
            "excluded_reasons": dict(reasons),
        }

    labels_by_group = {}
    for sample in samples:
        previous = labels_by_group.setdefault(sample["group_id"], sample["label"])
        if previous != sample["label"]:
            raise ValueError(f"Conflicting labels for {sample['group_id']}")

    x = np.stack([sample["x"] for sample in samples])
    y = np.asarray([sample["y"] for sample in samples], dtype=np.int64)
    original_y = np.asarray([label_mapping[sample["original_label"]] for sample in samples], dtype=np.int64)
    kicking_foot = np.asarray([sample["kicking_foot"] for sample in samples])
    mirrored = np.asarray([sample["mirrored_to_canonical_foot"] for sample in samples], dtype=np.bool_)
    source = np.asarray([sample["source"] for sample in samples])
    group_id = np.asarray([sample["group_id"] for sample in samples])
    sequence_id = np.asarray([sample["sequence_id"] for sample in samples])
    np.savez_compressed(output / "integrated_6j_16f.npz", X=x, y=y, original_y=original_y,
                        kicking_foot=kicking_foot, mirrored=mirrored, source=source,
                        group_id=group_id, sequence_id=sequence_id)

    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)

    long_fields = ["source", "group_id", "sequence_id", "label", "original_label", "kicking_foot",
                   "mirrored_to_canonical_foot", "timestep", "source_frame",
                   "joint_id", "joint_name", "x", "y", "vx", "vy"]
    with (output / "integrated_6j_16f_long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields); writer.writeheader()
        for sample in samples:
            data = sample["x"][..., 0]
            for t, frame in enumerate(sample["selected_frames"]):
                for joint_index, (joint_id, joint_name) in enumerate(zip(JOINT_IDS, JOINT_NAMES)):
                    writer.writerow({
                        "source": sample["source"], "group_id": sample["group_id"],
                        "sequence_id": sample["sequence_id"], "label": sample["label"],
                        "original_label": sample["original_label"],
                        "kicking_foot": sample["kicking_foot"],
                        "mirrored_to_canonical_foot": int(sample["mirrored_to_canonical_foot"]),
                        "timestep": t, "source_frame": frame, "joint_id": joint_id,
                        "joint_name": joint_name, "x": data[0,t,joint_index],
                        "y": data[1,t,joint_index], "vx": data[2,t,joint_index],
                        "vy": data[3,t,joint_index],
                    })

    per_source_sequences = Counter(sample["source"] for sample in samples)
    per_source_groups = Counter(group.split(":", 1)[0] for group in labels_by_group)
    group_labels = Counter(labels_by_group.values())
    summary = {
        "shape": list(x.shape), "channels": ["x", "y", "vx", "vy"],
        "label_mapping": label_mapping,
        "joint_ids": list(JOINT_IDS), "frames": config["frames"],
        "sequences": len(samples), "unique_source_prefixed_kicks": len(labels_by_group),
        "sequence_labels": dict(Counter(sample["label"] for sample in samples)),
        "kick_labels": dict(group_labels), "sequences_by_source": dict(per_source_sequences),
        "kicks_by_source": dict(per_source_groups), "files": file_info,
        "normalization": "per-frame hip midpoint; median bbox_h over selected 16 frames",
        "velocity": "first difference of normalized coordinates per original frame",
        "missing_rule": "blank, non-finite, or raw x/y <= 0 excludes the sequence",
        "canonical_foot": config.get("canonical_foot"),
        "require_known_foot": config.get("require_known_foot", False),
        "mirrored_sequences": sum(s["mirrored_to_canonical_foot"] for s in samples),
        "canonicalization": "hip-centered x reflection; left/right joint swap; velocity recomputed; L/R target swapped",
        "sampling": "16 consecutive original frames immediately before last_touch",
        "warning": "No cross-source semantic duplicate detection was possible from CSV metadata alone.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
