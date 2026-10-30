#!/usr/bin/env python3
"""Inspect replacement-video alignment and propose only missing EPL keypoints.

Inference outputs and provenance go under runs/epl_missing_redetection. CSV
changes are emitted as an apply_patch patch, so existing observations stay intact.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


JOINTS = (5, 6, 11, 12, 13, 14, 15, 16)
ANCHORS = (5, 6, 11, 12, 13, 14)
VIDEOS = {"8": "008-L-C.mp4", "9": "009-R-C.mp4",
          "15": "015-C-C.mp4", "72": "072-C-C.mp4"}
SOURCE = Path("Main Data/EPL_penalty_pose_keypoints.csv")
OUTPUT = Path("runs/epl_missing_redetection")


def missing(value):
    return value is None or value.strip().lower() in ("", "nan", "na", "null", "none")


def missing_joints(row):
    return [j for j in JOINTS
            if missing(row[f"kp_{j}_x"]) or missing(row[f"kp_{j}_y"])]


def prediction(model, frame, imgsz, device):
    result = model.predict(frame, imgsz=imgsz, conf=0.15, device=device,
                           verbose=False)[0]
    people = []
    if result.keypoints is not None and result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        points = result.keypoints.data.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for box, point, conf in zip(boxes, points, confidences):
            people.append({"box": box.tolist(), "points": point.tolist(),
                           "person_conf": float(conf)})
    return people


def distance(row, person):
    errors = []
    for j in ANCHORS:
        if missing(row[f"kp_{j}_x"]) or missing(row[f"kp_{j}_y"]):
            continue
        x, y, confidence = person["points"][j]
        if confidence < 0.25 or x <= 0 or y <= 0:
            continue
        errors.append(float(np.hypot(x-float(row[f"kp_{j}_x"]),
                                     y-float(row[f"kp_{j}_y"]))))
    return float(np.median(errors)) if len(errors) >= 3 else float("inf")


def closest(row, people):
    choices = [(distance(row, p), i) for i, p in enumerate(people)]
    return min(choices) if choices else (float("inf"), -1)


def sample_rows(rows, count=12):
    usable = [r for r in rows if sum(not missing(r[f"kp_{j}_x"])
                                    and not missing(r[f"kp_{j}_y"])
                                    for j in ANCHORS) >= 4]
    indices = np.linspace(0, len(usable)-1, min(count, len(usable)), dtype=int)
    return [usable[i] for i in indices] if usable else []


def align(rows, detections):
    comparisons = []
    for offset in range(-3, 4):
        errors = []
        for row in sample_rows(rows):
            frame = int(float(row["frame"])) + offset
            if 0 <= frame < len(detections):
                error, _ = closest(row, detections[frame])
                errors.append(error)
        finite = [e for e in errors if np.isfinite(e)]
        score = float(np.median(errors)) if errors else float("inf")
        comparisons.append({"offset": offset, "median_anchor_error_px": score,
                            "samples": len(errors), "finite_samples": len(finite),
                            "within_25px": sum(e <= 25 for e in errors)})
    best = min(comparisons, key=lambda c: c["median_anchor_error_px"])
    accepted = (best["median_anchor_error_px"] <= 20
                and best["within_25px"] >= max(3, best["samples"] * 0.75))
    return {"accepted": accepted, "best": best, "offset_comparison": comparisons}


def draw_audit(frame, row, person, recovered, title):
    image = frame.copy()
    if person is not None:
        x1, y1, x2, y2 = map(round, person["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    for j in JOINTS:
        if not missing(row[f"kp_{j}_x"]) and not missing(row[f"kp_{j}_y"]):
            pt = (round(float(row[f"kp_{j}_x"])), round(float(row[f"kp_{j}_y"])))
            cv2.circle(image, pt, 4, (255, 180, 0), -1)
    for j, data in recovered.items():
        pt = (round(data["x"]), round(data["y"]))
        cv2.circle(image, pt, 5, (0, 0, 255), -1)
        cv2.putText(image, f'{j}:{data["confidence"]:.2f}', (pt[0]+5, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(image, title, (15, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def scan():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for row in rows:
        if row["kick_id"] in VIDEOS:
            groups[(row["kick_id"], row["track_id"], row["camera_direction"])].append(row)
    model = YOLO("yolov8m-pose.pt")
    device = "0" if torch.cuda.is_available() else "cpu"
    all_reports = []
    cache = OUTPUT / "full_frame_detections.json"
    cache_data = json.loads(cache.read_text()) if cache.exists() else {}
    video_specs = list(VIDEOS.items())
    if Path("072-C-L.mp4").exists():
        video_specs.append(("72", "072-C-L.mp4"))
    for extra in (Path("072-C-R.mp4"), Path("runs/visualizations/072-C-R.mp4")):
        if extra.exists():
            video_specs.append(("72", str(extra)))
    for kick, video_name in video_specs:
        cap = cv2.VideoCapture(video_name)
        if not cap.isOpened():
            raise FileNotFoundError(video_name)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        video_hash = hashlib.sha256(Path(video_name).read_bytes()).hexdigest()
        cached = cache_data.get(video_name, {})
        if cached.get("video_sha256") == video_hash:
            detections = cached["detections"]
        else:
            detections = []
            for i, frame in enumerate(frames):
                detections.append(prediction(model, frame, 1280, device))
                if i % 15 == 0:
                    print(f"{video_name}: full-frame inference {i+1}/{len(frames)}", flush=True)
            cache_data[video_name] = {"video_sha256": video_hash,
                                     "model": "yolov8m-pose.pt", "imgsz": 1280,
                                     "detections": detections}
            cache.write_text(json.dumps(cache_data))
        for key, group_rows in groups.items():
            if key[0] != kick:
                continue
            alignment = align(group_rows, detections)
            report = {"kick_id": kick, "track_id": key[1], "camera": key[2],
                      "video": video_name, "fps": fps, "video_frames": len(frames),
                      "alignment": alignment, "rows": []}
            print(f"ALIGN {key}: {alignment['best']} accepted={alignment['accepted']}", flush=True)
            tiles = []
            if alignment["accepted"]:
                offset = alignment["best"]["offset"]
                for row in group_rows:
                    joints = missing_joints(row)
                    if not joints:
                        continue
                    video_frame = int(float(row["frame"])) + offset
                    error, pi = closest(row, detections[video_frame])
                    entry = {"csv_frame": row["frame"], "video_frame": video_frame,
                             "anchor_error_px": error, "missing_joints": joints,
                             "recovered": {}}
                    person = detections[video_frame][pi] if pi >= 0 and error <= 30 else None
                    if person is not None:
                        candidates = [("full1280", person)]
                        # Check multiple input scales before accepting that an
                        # occluded ankle cannot be recovered from this view.
                        for size in (640, 960, 1536):
                            extra = prediction(model, frames[video_frame], size, device)
                            ee, ei = closest(row, extra)
                            if ei >= 0 and ee <= 30:
                                candidates.append((f"full{size}", extra[ei]))
                        # Expand the selected person box to keep feet and hands
                        # inside a high-resolution single-person crop.
                        height, width = frames[video_frame].shape[:2]
                        x1, y1, x2, y2 = person["box"]
                        bw, bh = x2-x1, y2-y1
                        xa = max(0, int(x1-0.65*bw)); xb = min(width, int(x2+0.65*bw))
                        ya = max(0, int(y1-0.25*bh)); yb = min(height, int(y2+0.35*bh))
                        crop = frames[video_frame][ya:yb, xa:xb]
                        for size in (640, 960, 1280):
                            cp = prediction(model, crop, size, device)
                            for p in cp:
                                p["box"] = [p["box"][0]+xa, p["box"][1]+ya,
                                            p["box"][2]+xa, p["box"][3]+ya]
                                for point in p["points"]:
                                    point[0] += xa; point[1] += ya
                            ce, ci = closest(row, cp)
                            if ci >= 0 and ce <= 30:
                                candidates.append((f"crop{size}", cp[ci]))
                        entry["candidate_keypoints"] = {
                            method: {j: p["points"][j] for j in joints}
                            for method, p in candidates}
                        for j in joints:
                            choices = []
                            for method, p in candidates:
                                x, y, conf = p["points"][j]
                                if conf >= 0.30 and 0 < x < width and 0 < y < height:
                                    choices.append((conf, x, y, method))
                            if choices:
                                conf, x, y, method = max(choices)
                                entry["recovered"][j] = {"x": x, "y": y,
                                                         "confidence": conf, "method": method}
                    report["rows"].append(entry)
                    annotated = draw_audit(frames[video_frame], row, person,
                                           entry["recovered"],
                                           f"kick {kick}, track {key[1]}, CSV {row['frame']}, video {video_frame}")
                    image_path = OUTPUT / f"{Path(video_name).stem}_track{key[1]}_frame{row['frame']}.jpg"
                    cv2.imwrite(str(image_path), annotated)
                    tiles.append(cv2.resize(annotated, (640, 360)))
                if tiles:
                    while len(tiles) % 2:
                        tiles.append(np.zeros_like(tiles[0]))
                    sheet = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)])
                    cv2.imwrite(str(OUTPUT / f"{Path(video_name).stem}_track{key[1]}_review.jpg"), sheet)
            else:
                row = sample_rows(group_rows)[0]
                fi = int(float(row["frame"]))
                fi = min(max(fi, 0), len(frames)-1)
                annotated = draw_audit(frames[fi], row, None, {},
                                       f"ALIGNMENT REJECTED: kick {kick}, track {key[1]}, CSV {row['frame']}")
                cv2.imwrite(str(OUTPUT / f"{Path(video_name).stem}_track{key[1]}_alignment_rejected.jpg"), annotated)
            all_reports.append(report)
            (OUTPUT / "redetection_report.json").write_text(json.dumps(all_reports, indent=2))
    print(json.dumps([{"kick": r["kick_id"], "track": r["track_id"],
                       "aligned": r["alignment"]["accepted"],
                       "missing_points": sum(len(x["missing_joints"]) for x in r["rows"]),
                       "recovered_points": sum(len(x["recovered"]) for x in r["rows"])}
                      for r in all_reports], indent=2), flush=True)


def propose():
    """Emit a reversible patch after inspection, without writing source CSVs."""
    threshold = 0.50
    reports = json.loads((OUTPUT / "redetection_report.json").read_text())
    original = SOURCE.read_bytes()
    lines = original.decode("utf-8").splitlines(keepends=True)
    reader = csv.DictReader(lines)
    fields = reader.fieldnames
    rows = list(reader)
    assert len(lines) == len(rows)+1
    by_key = {}
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        key = (row["kick_id"], row["track_id"], row["frame"])
        if row["kick_id"] in VIDEOS:
            assert key not in by_key, f"Ambiguous target row: {key}"
            by_key[key] = index
            groups[key[:2]].append(index)
    changes = defaultdict(list)
    estimates = {}
    for report in reports:
        if not report["alignment"]["accepted"]:
            continue
        for entry in report["rows"]:
            key = (report["kick_id"], report["track_id"], entry["csv_frame"])
            index = by_key[key]
            row = rows[index]
            for joint, data in entry["recovered"].items():
                if data["confidence"] < threshold:
                    continue
                j = int(joint)
                if not (missing(row[f"kp_{j}_x"]) and missing(row[f"kp_{j}_y"])):
                    raise ValueError(f"Would overwrite an observed keypoint: {key}, {j}")
                assert not missing(row["bbox_cx"]) and not missing(row["bbox_cy"])
                edits = {f"kp_{j}_x": data["x"], f"kp_{j}_y": data["y"],
                         f"kp_{j}_xc": data["x"]-float(row["bbox_cx"]),
                         f"kp_{j}_yc": data["y"]-float(row["bbox_cy"])}
                for col, value in edits.items():
                    assert missing(row[col]), f"Would overwrite {col} in {key}"
                    row[col] = format(value, ".15g")
                item = dict(data, joint=j, video=report["video"],
                            video_frame=entry["video_frame"],
                            anchor_error_px=entry["anchor_error_px"],
                            source_csv_line=index+2,
                            note="Model estimate; occluded joints are not directly visible measurements")
                changes[index].append(item)
                estimates[(index, j)] = item

    trimmed = set()
    for key, indices in groups.items():
        if not any(missing_joints(rows[i]) for i in indices):
            continue
        indices.sort(key=lambda i: float(rows[i]["frame"]))
        # Delete only an incomplete leading prefix. End gaps are never trimmed
        # implicitly, since they can contain the kick/contact event.
        leading = []
        for index in indices:
            if not missing_joints(rows[index]):
                break
            leading.append(index)
        if leading and len(leading) < len(indices):
            trimmed.update(leading)

    retained_indices = [i for i in range(len(rows)) if i not in trimmed]
    unresolved = [{"kick_id": rows[i]["kick_id"], "track_id": rows[i]["track_id"],
                   "frame": rows[i]["frame"], "joints": missing_joints(rows[i])}
                  for i in retained_indices if missing_joints(rows[i])]
    assert not unresolved, f"Unresolved EPL coordinates: {unresolved}"
    newline = "\r\n" if lines[0].endswith("\r\n") else "\n"

    def serialize(row):
        stream = io.StringIO(newline="")
        csv.DictWriter(stream, fieldnames=fields, lineterminator=newline).writerow(row)
        return stream.getvalue()

    new_lines = [lines[0]]
    for i in retained_indices:
        new_lines.append(serialize(rows[i]) if i in changes else lines[i+1])

    patch = ["*** Begin Patch", "*** Update File: " + str(SOURCE)]
    diff = list(difflib.unified_diff(lines, new_lines, n=2))
    for line in diff[2:]:
        patch.append("@@" if line.startswith("@@") else line.rstrip("\r\n"))
    backup_specs = [
        (OUTPUT / "modified_rows_before.csv", sorted(set(changes)-trimmed)),
        (Path("Main Data/excluded/EPL_leading_frames_trimmed.csv"), sorted(trimmed)),
    ]
    backup_reports = []
    for backup, indices in backup_specs:
        assert not backup.exists(), f"Backup already exists: {backup}"
        backup_lines = [lines[0]]+[lines[i+1] for i in indices]
        patch.append("*** Add File: " + str(backup))
        patch.extend("+"+line.rstrip("\r\n") for line in backup_lines)
        backup_reports.append({"path": str(backup), "rows": len(indices),
                               "sha256": hashlib.sha256("".join(backup_lines).encode()).hexdigest()})
    patch.append("*** End Patch")
    changed_report = []
    for index in sorted(changes):
        if index in trimmed:
            continue
        changed_report.append({"kick_id": rows[index]["kick_id"],
                               "track_id": rows[index]["track_id"],
                               "frame": rows[index]["frame"],
                               "joints": changes[index]})
    trimmed_report = [{"kick_id": rows[i]["kick_id"], "track_id": rows[i]["track_id"],
                       "frame": rows[i]["frame"], "source_csv_line": i+2,
                       "reason": "Leading incomplete frame after multi-scale redetection"}
                      for i in sorted(trimmed)]
    summary = {"confidence_threshold": threshold, "model": "yolov8m-pose.pt",
               "imgsz_attempted": [640, 960, 1280, 1536],
               "source_before_sha256": hashlib.sha256(original).hexdigest(),
               "source_after_sha256": hashlib.sha256("".join(new_lines).encode()).hexdigest(),
               "source_rows_before": len(rows), "source_rows_after": len(retained_indices),
               "filled_joint_points": sum(len(r["joints"]) for r in changed_report),
               "filled_rows": len(changed_report), "trimmed_rows": trimmed_report,
               "modified_rows": changed_report, "backups": backup_reports,
               "remaining_missing_rows": len(unresolved),
               "existing_observations_preserved": True,
               "occlusion_note": "Redetection supplies model estimates, including for occluded joints; confidence does not guarantee true ankle location."}
    (OUTPUT / "applied_changes.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"patch": "\n".join(patch)+"\n", "summary": summary}))


def verify():
    summary = json.loads((OUTPUT / "applied_changes.json").read_text())
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == summary["source_after_sha256"]
    for backup in summary["backups"]:
        assert hashlib.sha256(Path(backup["path"]).read_bytes()).hexdigest() == backup["sha256"]
    with SOURCE.open(newline="") as f:
        source_reader = csv.DictReader(f)
        fields = source_reader.fieldnames
        rows = list(source_reader)
    targets = {(r["kick_id"], r["track_id"], r["frame"]): r
               for r in rows if r["kick_id"] in VIDEOS}
    with (OUTPUT / "modified_rows_before.csv").open(newline="") as f:
        originals = {(r["kick_id"], r["track_id"], r["frame"]): r
                     for r in csv.DictReader(f)}
    audit_sheets = defaultdict(list)
    cache = json.loads((OUTPUT / "full_frame_detections.json").read_text())
    changed_cells = 0
    for change in summary["modified_rows"]:
        key = (change["kick_id"], change["track_id"], change["frame"])
        before, after = originals[key], targets[key]
        expected_columns = {f"kp_{j['joint']}_{suffix}" for j in change["joints"]
                            for suffix in ("x", "y", "xc", "yc")}
        actual_columns = {col for col in fields if before[col] != after[col]}
        assert actual_columns == expected_columns
        assert all(missing(before[c]) for c in actual_columns)
        assert all(not missing(after[c]) for c in actual_columns)
        changed_cells += len(actual_columns)
        for estimate in change["joints"]:
            j = estimate["joint"]
            assert estimate["confidence"] >= summary["confidence_threshold"]
            for axis, center in (("x", "bbox_cx"), ("y", "bbox_cy")):
                assert abs(float(after[f"kp_{j}_{axis}"])-float(after[center])
                           -float(after[f"kp_{j}_{axis}c"])) < 1e-8
        video = change["joints"][0]["video"]
        frame_index = change["joints"][0]["video_frame"]
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        assert ok
        error, person_index = closest(before, cache[video]["detections"][frame_index])
        assert error <= 30 and person_index >= 0
        person = cache[video]["detections"][frame_index][person_index]
        recovered = {x["joint"]: x for x in change["joints"]}
        image = draw_audit(frame, before, person, recovered,
                           f"APPLIED kick {key[0]}, track {key[1]}, frame {key[2]} (red=estimated)")
        audit_sheets[key[:2]].append(cv2.resize(image, (640, 360)))
    for (kick, track), tiles in audit_sheets.items():
        if len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(str(OUTPUT / f"applied_kick{kick}_track{track}.jpg"), sheet)
    for trimmed in summary["trimmed_rows"]:
        assert (trimmed["kick_id"], trimmed["track_id"], trimmed["frame"]) not in targets

    datasets = []
    for dataset in ("EPL", "Women", "CLT"):
        path = Path("Main Data") / f"{dataset}_penalty_pose_keypoints.csv"
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            dataset_fields = reader.fieldnames
            dataset_rows = list(reader)
        with path.open(newline="") as f:
            assert all(len(row) == len(dataset_fields) for row in csv.reader(f))
        bad = [r for r in dataset_rows if missing_joints(r)]
        centered_mismatches = 0
        max_formula_error = 0.0
        for row in dataset_rows:
            for j in JOINTS:
                for axis, center in (("x", "bbox_cx"), ("y", "bbox_cy")):
                    raw, centered = row[f"kp_{j}_{axis}"], row[f"kp_{j}_{axis}c"]
                    centered_mismatches += missing(raw) != missing(centered)
                    if not any(missing(x) for x in (raw, centered, row[center])):
                        max_formula_error = max(max_formula_error,
                                                abs(float(raw)-float(row[center])-float(centered)))
        assert centered_mismatches == 0
        assert max_formula_error < 1e-5
        if dataset == "EPL":
            assert not bad and all(r["kick_id"] != "16" for r in dataset_rows)
        elif dataset == "Women":
            assert not bad
        else:
            assert len(bad) == 6
            assert all(r["kick_id"] not in ("017-C-L.mp4", "029-C-C.mp4") for r in dataset_rows)
            assert all(r["kick_id"] == "026-C-L.mp4" and 0 <= float(r["frame"]) <= 5 for r in bad)
        datasets.append({"dataset": dataset, "rows": len(dataset_rows),
                         "columns": len(dataset_fields), "remaining_missing_rows": len(bad),
                         "max_centering_error": max_formula_error})
    result = {"datasets": datasets, "changed_cells": changed_cells,
              "existing_values_preserved": True, "source_hash_matches_patch": True,
              "backups_match_original_rows": True,
              "source_contact_frames_preserved": [
                  {"kick_id": r["kick_id"], "track_id": r["track_id"], "frame": r["frame"]}
                  for r in rows if r["kick_id"] in VIDEOS and r["last_touch"].strip() in ("1", "1.0")]}
    (OUTPUT / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--propose", action="store_true",
                        help="Emit a source-data patch and provenance after reviewing detections")
    parser.add_argument("--verify", action="store_true", help="Verify the applied patch and make final overlays")
    args = parser.parse_args()
    if args.verify:
        verify()
    elif args.propose:
        propose()
    else:
        scan()
