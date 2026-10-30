#!/usr/bin/env python3
"""Export one verified kicker track from run-up start to pre-contact frame."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


COCO = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
SKELETON = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)
LOWER = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--poses", type=Path, required=True)
    p.add_argument("--track-id", type=int, required=True)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--end-frame", type=int, required=True, help="inclusive, last pre-contact frame")
    p.add_argument("--contact-frame", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--max-track-gap", type=int, default=5)
    return p.parse_args()


def numeric(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else np.nan


def main() -> None:
    a = args()
    if not (a.start_frame <= a.end_frame < a.contact_frame):
        raise ValueError("Expected start_frame <= end_frame < contact_frame")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with a.poses.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [r for r in reader if int(r["track_id"]) == a.track_id and a.start_frame <= int(r["source_frame"]) <= a.end_frame]
    rows.sort(key=lambda r: int(r["source_frame"]))
    expected = list(range(a.start_frame, a.end_frame + 1))
    actual = [int(r["source_frame"]) for r in rows]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        runs=[]
        for frame in missing:
            if not runs or frame != runs[-1][-1] + 1: runs.append([frame])
            else: runs[-1].append(frame)
        if any(len(run) > a.max_track_gap for run in runs):
            raise RuntimeError(f"Kicker track gap exceeds {a.max_track_gap} frames: {missing[:20]}")
        by_frame={int(r["source_frame"]):dict(r) for r in rows}
        numeric_fields=[name for name in fieldnames if name not in {"source_frame","clip_frame","track_id"}]
        for run in runs:
            before,after=run[0]-1,run[-1]+1
            if before not in by_frame or after not in by_frame:
                raise RuntimeError(f"Cannot interpolate edge track gap: {run}")
            for frame in run:
                ratio=(frame-before)/(after-before); row=dict(by_frame[before])
                row["source_frame"]=str(frame); row["clip_frame"]=str(int(round((1-ratio)*int(by_frame[before]["clip_frame"])+ratio*int(by_frame[after]["clip_frame"]))))
                for name in numeric_fields:
                    try: row[name]=str((1-ratio)*float(by_frame[before][name])+ratio*float(by_frame[after][name]))
                    except (ValueError,TypeError): row[name]=""
                by_frame[frame]=row
        rows=[by_frame[frame] for frame in expected]
    track_interpolated_frames=set(expected)-set(actual)
    output_fields=fieldnames+["track_interpolated"]
    for row in rows: row["track_interpolated"]=int(int(row["source_frame"]) in track_interpolated_frames)
    actual=expected

    raw_path = a.output_dir / "kicker_runup_17j_raw.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader(); writer.writerows(rows)

    fps = float(cv2.VideoCapture(str(a.video)).get(cv2.CAP_PROP_FPS))
    frames = np.asarray(actual)
    pos = np.empty((len(rows), len(LOWER), 2), dtype=float)
    conf = np.empty((len(rows), len(LOWER)), dtype=float)
    bbox_h = np.asarray([float(r["bbox_h"]) for r in rows])
    for t, row in enumerate(rows):
        for j, name in enumerate(LOWER):
            pos[t, j] = [numeric(row, f"{name}_x"), numeric(row, f"{name}_y")]
            conf[t, j] = numeric(row, f"{name}_conf")
    missing = ~np.isfinite(pos)
    interpolated = missing.any(axis=2)
    for j in range(len(LOWER)):
        for axis in range(2):
            values = pos[:, j, axis]
            valid = np.isfinite(values)
            if valid.sum() < 2:
                raise RuntimeError(f"Insufficient valid values for {LOWER[j]} axis {axis}")
            pos[:, j, axis] = np.interp(frames, frames[valid], values[valid])
    scale = float(np.median(bbox_h))
    hip_center = (pos[:, 0] + pos[:, 1]) / 2
    norm = (pos - hip_center[:, None]) / scale
    velocity = np.zeros_like(norm)
    velocity[1:] = np.diff(norm, axis=0) * fps

    graph_path = a.output_dir / "kicker_runup_6j_graph.csv"
    graph_fields = ["source_frame", "time_sec", "track_id", "is_last_pre_contact"]
    for name in LOWER:
        graph_fields += [f"{name}_x_norm", f"{name}_y_norm", f"{name}_vx_per_sec", f"{name}_vy_per_sec", f"{name}_conf", f"{name}_interpolated"]
    graph_rows = []
    for t, row in enumerate(rows):
        out = {"source_frame": int(frames[t]), "time_sec": float(row["time_sec"]), "track_id": a.track_id, "is_last_pre_contact": int(frames[t] == a.end_frame)}
        for j, name in enumerate(LOWER):
            out.update({
                f"{name}_x_norm": norm[t, j, 0], f"{name}_y_norm": norm[t, j, 1],
                f"{name}_vx_per_sec": velocity[t, j, 0], f"{name}_vy_per_sec": velocity[t, j, 1],
                f"{name}_conf": conf[t, j], f"{name}_interpolated": int(interpolated[t, j]),
            })
        graph_rows.append(out)
    with graph_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=graph_fields)
        writer.writeheader(); writer.writerows(graph_rows)

    sample_indices = np.rint(np.linspace(0, len(rows) - 1, a.samples)).astype(int)
    sample_path = a.output_dir / f"kicker_runup_6j_{a.samples}samples.csv"
    sample_fields = ["sample_index", "source_frame", "time_sec"] + [f"{name}_{feature}" for name in LOWER for feature in ("x", "y", "vx", "vy")]
    with sample_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields); writer.writeheader()
        for sample_number, t in enumerate(sample_indices):
            out = {"sample_index": sample_number, "source_frame": int(frames[t]), "time_sec": float(rows[t]["time_sec"])}
            for j, name in enumerate(LOWER):
                out.update({f"{name}_x": norm[t,j,0], f"{name}_y": norm[t,j,1], f"{name}_vx": velocity[t,j,0], f"{name}_vy": velocity[t,j,1]})
            writer.writerow(out)

    capture = cv2.VideoCapture(str(a.video))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)
    overlay_path = a.output_dir / "kicker_only_overlay_mpeg4.mp4"
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for t, row in enumerate(rows):
        ok, image = capture.read()
        if not ok: raise RuntimeError(f"Could not read source frame {frames[t]}")
        points=[]
        for name in COCO:
            x,y=numeric(row,f"{name}_x"),numeric(row,f"{name}_y")
            points.append(None if not np.isfinite([x,y]).all() else (int(round(x)),int(round(y))))
        for left,right in SKELETON:
            if points[left] and points[right]: cv2.line(image,points[left],points[right],(0,255,255),4,cv2.LINE_AA)
        for point in points:
            if point: cv2.circle(image,point,6,(0,0,255),-1,cv2.LINE_AA)
        x1,y1,x2,y2=[int(round(float(row[k]))) for k in ("bbox_x1","bbox_y1","bbox_x2","bbox_y2")]
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
        cv2.putText(image,f"KICKER track={a.track_id} frame={int(frames[t])}",(max(10,x1),max(35,y1-12)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2,cv2.LINE_AA)
        writer.write(image)
    capture.release(); writer.release()

    metadata = {
        "track_id": a.track_id, "fps": fps,
        "runup_start_frame": a.start_frame, "runup_start_sec": a.start_frame / fps,
        "last_pre_contact_frame": a.end_frame, "last_pre_contact_sec": a.end_frame / fps,
        "contact_frame": a.contact_frame, "contact_sec": a.contact_frame / fps,
        "runup_frames": len(rows), "runup_duration_sec": len(rows) / fps,
        "normalization": "framewise hip midpoint; median run-up bbox height",
        "velocity": "normalized coordinate difference multiplied by source FPS",
        "interpolated_joint_frames": int(interpolated.any(axis=1).sum()),
        "interpolated_track_frames": len(track_interpolated_frames),
        "sampled_source_frames": frames[sample_indices].astype(int).tolist(),
        "boundary_review_required": True,
    }
    (a.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__": main()
