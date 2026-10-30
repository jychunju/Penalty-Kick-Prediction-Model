#!/usr/bin/env python3
"""Extract tracked YOLOv8 poses from a short video interval.

The raw CSV intentionally keeps every detected person and all 17 COCO
keypoints.  Kicker selection is a separate, auditable ranking step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--start", type=float, default=12.0, help="start time in seconds")
    parser.add_argument("--duration", type=float, default=21.0, help="duration in seconds")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/yolov8_pose_preview"))
    parser.add_argument("--model", default="yolov8m-pose.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--keypoint-confidence", type=float, default=0.25)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    return parser.parse_args()


def video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {path}")
    return writer


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise FileNotFoundError(args.input)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, round(args.start * fps))
    requested_frames = max(1, round(args.duration * fps))
    stop_frame = min(total_frames, start_frame + requested_frames)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    model = YOLO(args.model)
    overlay_path = args.output_dir / "pose_overlay.mp4"
    writer = video_writer(overlay_path, fps, width, height)
    rows: list[dict] = []
    observations: dict[int, list[dict]] = defaultdict(list)
    processed_frames = 0

    for source_frame in range(start_frame, stop_frame):
        ok, frame = capture.read()
        if not ok:
            break
        processed_frames += 1
        result = model.track(
            frame, persist=True, tracker=args.tracker, device=args.device,
            imgsz=args.imgsz, conf=args.confidence, verbose=False,
        )[0]
        annotated = result.plot(kpt_radius=4, line_width=2)
        if result.boxes is not None and result.keypoints is not None and len(result.boxes):
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            box_conf = result.boxes.conf.detach().cpu().numpy()
            if result.boxes.id is None:
                track_ids = np.full(len(boxes), -1, dtype=int)
            else:
                track_ids = result.boxes.id.detach().cpu().numpy().astype(int)
            xy = result.keypoints.xy.detach().cpu().numpy()
            kp_conf = result.keypoints.conf.detach().cpu().numpy()
            for box, confidence, track_id, keypoints, confidences in zip(
                boxes, box_conf, track_ids, xy, kp_conf
            ):
                x1, y1, x2, y2 = map(float, box)
                row = {
                    "source_frame": source_frame,
                    "clip_frame": source_frame - start_frame,
                    "time_sec": source_frame / fps,
                    "track_id": int(track_id),
                    "person_conf": float(confidence),
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                    "bbox_cx": (x1 + x2) / 2, "bbox_cy": (y1 + y2) / 2,
                    "bbox_w": x2 - x1, "bbox_h": y2 - y1,
                }
                for name, (x, y), conf in zip(COCO_KEYPOINTS, keypoints, confidences):
                    valid = bool(conf >= args.keypoint_confidence and x > 0 and y > 0)
                    row[f"{name}_x"] = float(x) if valid else ""
                    row[f"{name}_y"] = float(y) if valid else ""
                    row[f"{name}_conf"] = float(conf)
                rows.append(row)
                if track_id >= 0:
                    observations[int(track_id)].append(row)
        cv2.putText(
            annotated, f"source frame {source_frame}  t={source_frame / fps:.2f}s",
            (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(annotated)

    capture.release()
    writer.release()
    if not rows:
        raise RuntimeError("No people with pose detections were found")

    raw_csv = args.output_dir / "pose_keypoints_raw.csv"
    with raw_csv.open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    candidates = []
    for track_id, track_rows in observations.items():
        centers = np.asarray([[r["bbox_cx"], r["bbox_cy"]] for r in track_rows], dtype=float)
        heights = np.asarray([r["bbox_h"] for r in track_rows], dtype=float)
        displacement = np.linalg.norm(np.diff(centers, axis=0), axis=1).sum() if len(centers) > 1 else 0.0
        duration_frames = len({int(r["source_frame"]) for r in track_rows})
        lower_body_names = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
        visible = np.mean([
            sum(bool(r[f"{name}_x"] != "") for name in lower_body_names) / len(lower_body_names)
            for r in track_rows
        ])
        center_distance = np.linalg.norm(centers.mean(axis=0) / [width, height] - [0.5, 0.62])
        # A ranking aid, not a semantic kicker classifier. It favors long,
        # visible, moving tracks near the central pitch area.
        score = (
            math.log1p(duration_frames) + 2.0 * float(visible)
            + math.log1p(displacement / max(width, 1)) - float(center_distance)
        )
        candidates.append({
            "track_id": track_id,
            "frames": duration_frames,
            "first_time_sec": min(r["time_sec"] for r in track_rows),
            "last_time_sec": max(r["time_sec"] for r in track_rows),
            "mean_bbox_height": float(heights.mean()),
            "motion_pixels": float(displacement),
            "lower_body_visibility": float(visible),
            "heuristic_score": float(score),
        })
    candidates.sort(key=lambda item: item["heuristic_score"], reverse=True)
    for rank, candidate in enumerate(candidates, 1):
        candidate["candidate_rank"] = rank
    candidate_csv = args.output_dir / "track_candidates.csv"
    with candidate_csv.open("w", newline="") as handle:
        candidate_writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        candidate_writer.writeheader()
        candidate_writer.writerows(candidates)

    metadata = {
        "input": str(args.input.resolve()),
        "model": args.model,
        "ultralytics_version": __import__("ultralytics").__version__,
        "torch_version": torch.__version__,
        "device": args.device,
        "source_fps": fps,
        "source_size": [width, height],
        "start_sec": args.start,
        "duration_sec_requested": args.duration,
        "start_frame": start_frame,
        "stop_frame_exclusive": start_frame + processed_frames,
        "processed_video_frames": processed_frames,
        "pose_rows": len(rows),
        "tracked_people": len(candidates),
        "outputs": {
            "overlay": str(overlay_path),
            "raw_keypoints": str(raw_csv),
            "track_candidates": str(candidate_csv),
        },
        "note": "candidate_rank is heuristic and must be checked in pose_overlay.mp4",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
