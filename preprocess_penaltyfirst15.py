#!/usr/bin/env python3
"""Apply the existing Main Data preprocessing to the verified PK pose CSV.

Filename: number-camera_view-kicking_foot-kick_direction.mp4.
Directions are from the kicker's perspective. Raw sources are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from canonicalize_main_data import canonicalize_row, SWAP
from extract_8_joint_pose import extract_8_joints, JOINTS as SELECTED
from extract_uniform_20frames import extract_uniform_20frames
from spatial_normalize_pose import normalize_pose_csv

NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle")
BODY = tuple(range(5, 17))
ANGLES = {"knee_angle_l_n": (11, 13, 15), "knee_angle_r_n": (12, 14, 16),
          "hip_angle_l_n": (5, 11, 13), "hip_angle_r_n": (6, 12, 14)}
PATTERN = re.compile(r"^(\d+)-([LCR])-([LR])-([LCR])\.mp4$")
META = ("dataset", "kick_id", "frame", "fps", "kicker", "game_or_training",
        "last_touch", "kicker_track_id_main", "track_id", "camera_direction",
        "r_or_l", "kick_direction", "goal", "bbox_cx", "bbox_cy", "bbox_w", "bbox_h")
COORDS = tuple(f"kp_{j}_{suffix}" for suffixes in (("x", "y"), ("xc", "yc"))
               for j in BODY for suffix in suffixes)
AUDIT = ("source_frame", "time_sec", "source_clip_frame", "original_r_or_l",
         "original_kick_direction", "mirrored_to_right_foot", "contact_status",
         "source_contact_candidate", "person_conf", "source_detection_index")
QUALITY = tuple(f"kp_{j}_{suffix}" for j in BODY
                for suffix in ("conf", "observed_valid", "interpolated"))
FIELDS = list(META + COORDS + AUDIT + QUALITY)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if any(None in row or None in row.values() for row in rows):
            raise ValueError(f"Malformed CSV: {path}")
        return list(reader.fieldnames or []), rows


def write_csv(path, fields, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def filename_labels(name):
    m = PATTERN.fullmatch(name)
    if not m:
        raise ValueError(f"Expected number-camera-foot-direction.mp4: {name}")
    _, camera, foot, direction = m.groups()
    return camera, foot, direction


def finite_coordinate(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def adapt(source, spec):
    camera, foot, direction = filename_labels(source["video"])
    bbox = [float(source[k]) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
    if not all(math.isfinite(v) for v in bbox):
        raise ValueError(f"Missing kicker bbox: {source['video']} frame {source['source_frame']}")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bbox")
    row = dict.fromkeys(FIELDS, "")
    row.update(dataset="penaltyfirst15", kick_id=source["video"], frame=source["source_frame"],
               fps="25", last_touch=source["is_contact"], kicker_track_id_main="1",
               camera_direction=camera, r_or_l=foot, kick_direction=direction,
               bbox_cx=str((x1+x2)/2), bbox_cy=str((y1+y2)/2),
               bbox_w=str(x2-x1), bbox_h=str(y2-y1),
               source_frame=source["source_frame"], time_sec=source["time_sec"],
               source_clip_frame=source["clip_frame"], original_r_or_l=foot,
               original_kick_direction=direction, mirrored_to_right_foot=str(int(foot == "L")),
               contact_status=spec["contact_status"], source_contact_candidate=source["is_contact_candidate"],
               person_conf=source["person_conf"], source_detection_index=source["detection_index"])
    # ID 1 denotes the already selected kicker within this clip, not a detector track ID.
    # Unknown outcome/player labels and detector track IDs stay empty.
    for j in BODY:
        name = NAMES[j]
        observed = source[f"{name}_valid"] == "1"
        if observed:
            for axis, center in (("x", "bbox_cx"), ("y", "bbox_cy")):
                raw = source[f"{name}_{axis}"]
                if not finite_coordinate(raw):
                    raise ValueError("Valid mask disagrees with source coordinates")
                row[f"kp_{j}_{axis}"] = raw
                row[f"kp_{j}_{axis}c"] = str(float(raw)-float(row[center]))
        row[f"kp_{j}_conf"] = source[f"{name}_conf"]
        row[f"kp_{j}_observed_valid"] = str(int(observed))
        row[f"kp_{j}_interpolated"] = "0"
    assert abs(float(row["time_sec"])-int(row["frame"])/25) < 1e-9
    return row


def interpolate_internal(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["kick_id"]].append(row)
    repairs, unfilled = [], []
    for kick, sequence in groups.items():
        times = np.array([int(r["frame"]) for r in sequence])
        assert np.all(np.diff(times) > 0)
        for j in BODY:
            valid = [t for t, r in enumerate(sequence)
                     if finite_coordinate(r[f"kp_{j}_x"]) and finite_coordinate(r[f"kp_{j}_y"])]
            for t, row in enumerate(sequence):
                if t in valid:
                    continue
                before, after = [i for i in valid if i < t], [i for i in valid if i > t]
                if not before or not after:
                    unfilled.append({"kick_id": kick, "frame": int(times[t]), "joint_id": j,
                                     "reason": "no_bracketing_observations_no_extrapolation"})
                    continue
                lo, hi = before[-1], after[0]
                q = (times[t]-times[lo])/(times[hi]-times[lo])
                for axis, center in (("x", "bbox_cx"), ("y", "bbox_cy")):
                    value = ((1-q)*float(sequence[lo][f"kp_{j}_{axis}"])
                             + q*float(sequence[hi][f"kp_{j}_{axis}"]))
                    row[f"kp_{j}_{axis}"] = str(value)
                    row[f"kp_{j}_{axis}c"] = str(value-float(row[center]))
                row[f"kp_{j}_interpolated"] = "1"
                repairs.append({"kick_id": kick, "frame": int(times[t]), "joint_id": j,
                                "left_frame": int(times[lo]), "right_frame": int(times[hi])})
    return repairs, unfilled


def canonicalize_with_quality(row):
    result = canonicalize_row(row)
    if row["r_or_l"] == "L":
        for j in BODY:
            for suffix in ("conf", "observed_valid", "interpolated"):
                result[f"kp_{j}_{suffix}"] = row[f"kp_{SWAP[j]}_{suffix}"]
        # Real-data roundtrip validates axes, semantic joint swaps and labels.
        restored = canonicalize_row(dict(result, r_or_l="L"))
        for j in BODY:
            for suffix in ("x", "y", "xc", "yc"):
                column = f"kp_{j}_{suffix}"
                if row[column] == "":
                    assert restored[column] == ""
                else:
                    assert math.isclose(float(restored[column]), float(row[column]), abs_tol=1e-9, rel_tol=1e-12)
        assert restored["kick_direction"] == row["kick_direction"]
    else:
        assert result == row
    assert result["camera_direction"] == row["camera_direction"]
    assert result["r_or_l"] == "R"
    return result


def angle_scaled(a, b, c):
    u, v = np.asarray(a)-b, np.asarray(c)-b
    denom = float(np.linalg.norm(u)*np.linalg.norm(v))
    if denom <= 1e-12 or not math.isfinite(denom):
        raise ValueError("Degenerate angle has no valid value")
    return math.acos(float(np.clip(np.dot(u, v)/denom, -1., 1.)))/math.pi


def add_angles(input_path, output_path):
    fields, rows = read_csv(input_path)
    for row in rows:
        for name, triplet in ANGLES.items():
            points = [np.array([float(row[f"kp_{j}_xn"]), float(row[f"kp_{j}_yn"])]) for j in triplet]
            value = angle_scaled(*points)
            assert math.isfinite(value) and 0 <= value <= 1
            row[name] = str(value)
    write_csv(output_path, fields+list(ANGLES), rows)
    return rows


def validate_final(final_path, reference, source_rows, canonical_rows, normalized_path):
    fields, rows = read_csv(final_path)
    ref_fields, ref_rows = read_csv(reference)
    assert set(ref_fields) <= set(fields), "Existing model input columns are missing"
    angle_errors = {}
    for name, triplet in ANGLES.items():
        differences = []
        for row in ref_rows:
            points = [np.array([float(row[f"kp_{j}_xn"]), float(row[f"kp_{j}_yn"])]) for j in triplet]
            differences.append(abs(angle_scaled(*points)-float(row[name])))
        angle_errors[name] = max(differences)
        assert angle_errors[name] < 1e-6, "Angle definition differs from prior dataset"
    assert math.isclose(angle_scaled([1, 0], np.array([0, 0]), [0, 1]), .5)
    assert math.isclose(angle_scaled([-1, 0], np.array([0, 0]), [1, 0]), 1)
    counts = Counter(r["kick_id"] for r in rows)
    assert len(rows) == 280 and len(counts) == 14
    assert set(counts.values()) == {20}
    assert "001-R-L-L.mp4" not in counts and "007-C-R-L.mp4" not in counts
    assert "008-C-L-R.mp4" in counts and "008-C-R-L.mp4" in counts
    full = defaultdict(list)
    for row in canonical_rows:
        full[row["kick_id"]].append(row)
    _, norm_rows = read_csv(normalized_path)
    normalized = {(r["kick_id"], r["frame"]): r for r in norm_rows}
    source = {(r["video"], r["source_frame"]): r for r in source_rows}
    features = [f"kp_{j}_{axis}n" for j in SELECTED for axis in ("x", "y")] + list(ANGLES)
    for kick in counts:
        sample = [r for r in rows if r["kick_id"] == kick]
        assert [int(r["sample_20_idx"]) for r in sample] == list(range(20))
        times = [int(r["frame"]) for r in sample]
        assert all(b > a for a, b in zip(times[:-1], times[1:]))
        assert times[0] == int(full[kick][0]["frame"])
        assert times[-1] == int(full[kick][-1]["frame"])
        assert sample[-1]["last_touch"] == "1"
        torso = []
        for r in full[kick]:
            pts = {j: np.array([float(r[f"kp_{j}_x"]), float(r[f"kp_{j}_y"])]) for j in (5, 6, 11, 12)}
            torso.append(np.linalg.norm((pts[5]+pts[6])/2-(pts[11]+pts[12])/2))
        expected_scale = float(np.median(torso))
        for r in sample:
            assert r["r_or_l"] == "R"
            assert all(math.isfinite(float(r[c])) for c in features)
            assert math.isclose(float(r["clip_torso_scale"]), expected_scale, rel_tol=1e-12)
            assert abs((float(r["kp_11_xn"])+float(r["kp_12_xn"]))/2) < 1e-10
            assert abs((float(r["kp_11_yn"])+float(r["kp_12_yn"]))/2) < 1e-10
            original = source[(kick, r["source_frame"])]
            camera, foot, direction = filename_labels(kick)
            assert r["camera_direction"] == camera
            assert r["original_r_or_l"] == foot and r["original_kick_direction"] == direction
            assert r["kick_direction"] == ({"L": "R", "R": "L", "C": "C"}[direction] if foot == "L" else direction)
            assert r["time_sec"] == original["time_sec"]
            assert r["goal"] == ""
            for j in SELECTED:
                for suffix in ("x", "y", "xc", "yc", "xn", "yn"):
                    assert r[f"kp_{j}_{suffix}"] == normalized[(kick, r["frame"])][f"kp_{j}_{suffix}"]
    return {"passed": True, "rows": len(rows), "kicks": len(counts), "rows_per_kick": 20,
            "model_features": features, "all_model_features_finite": True,
            "all_existing_final_columns_present": True, "original_frame_and_time_preserved": True,
            "source_filename_labels_verified": True, "reflection_roundtrip_verified": True,
            "pelvis_origin_and_full_sequence_torso_median_verified": True,
            "angle_definition_max_error_vs_existing_csv": angle_errors,
            "angles_in_unit_interval": True, "duplicate_008_ids_kept_distinct": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("Main Data/penaltyfirst15_pose_keypoints.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("Main Data"))
    p.add_argument("--report-dir", type=Path, default=Path("runs/penaltyfirst15_preprocessing"))
    a = p.parse_args()
    reference = Path("Main Data/Combined_penalty_pose_keypoints_normalized_20frames_8joints_angles_scaled.csv")
    manifest = Path("runs/penaltyfirst15_pose/segments.json")
    protected = [a.input, manifest, *Path("Main Data").glob("Combined_penalty_pose_keypoints*.csv")]
    hashes = {str(path): digest(path) for path in protected}
    _, source_rows = read_csv(a.input)
    specs = json.loads(manifest.read_text())
    if {r["video"] for r in source_rows} != set(specs):
        raise ValueError("Source videos disagree with verified segment manifest")
    prefix = "penaltyfirst15_pose_keypoints"
    paths = {"canonical": a.output_dir/f"{prefix}_canonical.csv",
             "normalized": a.output_dir/f"{prefix}_normalized.csv",
             "sampled": a.output_dir/f"{prefix}_normalized_20frames.csv",
             "eight_joints": a.output_dir/f"{prefix}_normalized_20frames_8joints.csv",
             "final": a.output_dir/f"{prefix}_normalized_20frames_8joints_angles_scaled.csv"}
    report_files = [a.report_dir / x for x in ("summary.json", "trimmed_leading_frames.csv", "excluded_final_raw_rows.csv")]
    existing = [str(path) for path in [*paths.values(), *report_files] if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    a.report_dir.mkdir(parents=True, exist_ok=True)
    trimmed = [r for r in source_rows if r["video"] == "011-R-L-R.mp4" and int(r["source_frame"]) < 9]
    assert [int(r["source_frame"]) for r in trimmed] == [4, 5, 6, 7, 8]
    kept = [r for r in source_rows if not (r["video"] == "011-R-L-R.mp4" and int(r["source_frame"]) < 9)]
    rows = [adapt(r, specs[r["video"]]) for r in kept]
    repairs, unfilled = interpolate_internal(rows)
    canonical_rows = [canonicalize_with_quality(r) for r in rows]
    write_csv(paths["canonical"], FIELDS, canonical_rows)
    normalize_pose_csv(paths["canonical"], paths["normalized"])
    sampling = extract_uniform_20frames(paths["normalized"], paths["sampled"])
    assert {(x["kick_id"], tuple(x["reasons"])) for x in sampling["excluded_kicks"]} == {
        ("001-R-L-L.mp4", ("missing_last_touch",)),
        ("007-C-R-L.mp4", ("20_or_fewer_frames_through_touch",))}
    extract_8_joints(paths["sampled"], paths["eight_joints"])
    final_rows = add_angles(paths["eight_joints"], paths["final"])
    validation = validate_final(paths["final"], reference, source_rows, canonical_rows, paths["normalized"])
    for path, expected_hash in hashes.items():
        assert digest(path) == expected_hash, f"Source modified: {path}"
    validation["raw_and_prior_datasets_unchanged"] = True
    source_fields, _ = read_csv(a.input)
    write_csv(a.report_dir/"trimmed_leading_frames.csv", source_fields, trimmed)
    excluded_names = {x["kick_id"] for x in sampling["excluded_kicks"]}
    write_csv(a.report_dir/"excluded_final_raw_rows.csv", source_fields,
              [r for r in source_rows if r["video"] in excluded_names])
    per_kick = []
    for name in specs:
        camera, foot, direction = filename_labels(name)
        cf = [r for r in canonical_rows if r["kick_id"] == name]
        output = [r for r in final_rows if r["kick_id"] == name]
        per_kick.append({"kick_id": name, "camera_direction": camera, "original_foot": foot,
                         "original_kick_direction": direction, "canonical_foot": "R",
                         "canonical_kick_direction": cf[0]["kick_direction"],
                         "mirrored": foot == "L", "retained_full_frames": len(cf),
                         "first_frame": int(cf[0]["frame"]), "last_frame": int(cf[-1]["frame"]),
                         "sampled_frames": [int(r["frame"]) for r in output],
                         "included_in_final": bool(output)})
    summary = {"input": str(a.input), "reference": str(reference), "source_hashes": hashes,
               "filename_format_user_confirmed": "number-camera_view-kicker_foot-kick_direction_from_kicker_perspective",
               "input_rows": len(source_rows), "input_videos": len(specs),
               "trimmed_frames_011": [4, 5, 6, 7, 8], "retained_full_rows": len(rows),
               "final_rows": len(final_rows), "final_videos": len(final_rows)//20,
               "interpolation": "linear in source-frame time; internal bracketed gaps only, no extrapolation",
               "interpolated_joint_observations": len(repairs), "repairs": repairs,
               "remaining_unfilled_joints": unfilled, "excluded_final": sampling["excluded_kicks"],
               "canonicalization": "same canonicalize_main_data.canonicalize_row; bbox-centered x reflection and anatomical swaps",
               "normalization": "same spatial_normalize_pose.normalize_pose_csv; per-frame hip center / full-sequence median torso length",
               "sampling": sampling["sampling"], "selected_joints": list(SELECTED),
               "angles": "knee: hip-knee-ankle; hip: shoulder-hip-knee; acos(dot/norms)/pi",
               "outcome": "unknown; goal column left empty",
               "contact_semantics": "source is_contact labels the visually estimated contact endpoint, not a new post-contact event search",
               "metadata": "kicker_track_id_main=1 is selected kicker ID within each clip, not detector track ID",
               "labels_in_final": dict(Counter(r["kick_direction"] for r in final_rows[::20])),
               "outputs": {k: str(path) for k, path in paths.items()},
               "output_sha256": {k: digest(path) for k, path in paths.items()},
               "per_kick": per_kick, "validation": validation}
    (a.report_dir/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps({k: summary[k] for k in ("final_rows", "final_videos", "interpolated_joint_observations",
                     "remaining_unfilled_joints", "excluded_final", "labels_in_final", "validation")}, indent=2))


if __name__ == "__main__":
    main()
