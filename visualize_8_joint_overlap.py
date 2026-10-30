#!/usr/bin/env python3
"""Overlay five random or explicitly selected kicks at their 20 sample steps."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "Main Data/Combined_penalty_pose_keypoints_normalized_20frames_8joints.csv"
OUTPUT = ROOT / "runs/visualizations/random_5_8joints_20frames_overlap.mp4"
JOINTS = (5, 6, 11, 12, 13, 14, 15, 16)
EDGES = ((5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16))
COLORS = ((255, 200, 65), (70, 190, 255), (220, 95, 235),
          (110, 225, 100), (105, 110, 255))  # OpenCV BGR
WIDTH, HEIGHT = 1280, 900
FPS, HOLD_PER_STEP, CONTACT_HOLD, REPEATS = 24, 6, 24, 2


def read_selected(input_path, seed, chosen_keys=None):
    groups = defaultdict(list)
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            groups[(row["dataset"], row["kick_id"])].append(row)
    chosen = random.Random(seed).sample(sorted(groups), 5) if chosen_keys is None else list(chosen_keys)
    if len(chosen) != 5 or len(set(chosen)) != 5:
        raise ValueError("Select exactly five distinct dataset/kick_id pairs")
    absent = [key for key in chosen if key not in groups]
    if absent:
        raise ValueError(f"Requested kicks not found: {absent}")
    sequences = []
    for key in chosen:
        rows = groups[key]
        assert len(rows) == 20
        assert [int(row["sample_20_idx"]) for row in rows] == list(range(20))
        frames = [float(row["frame"]) for row in rows]
        assert all(a < b for a, b in zip(frames, frames[1:]))
        assert float(rows[-1]["last_touch"]) == 1
        poses = np.array([[[float(row[f"kp_{joint}_xn"]), float(row[f"kp_{joint}_yn"])]
                           for joint in JOINTS] for row in rows])
        assert poses.shape == (20, 8, 2) and np.isfinite(poses).all()
        pelvis = (poses[:, 2]+poses[:, 3])/2
        assert np.max(np.abs(pelvis)) < 1e-10
        sequences.append({"key": key, "rows": rows, "poses": poses})
    return sequences


def shared_transform(sequences):
    """A single isotropic screen transform for every kick and every frame."""
    all_points = np.concatenate([item["poses"].reshape(-1, 2) for item in sequences])
    lower, upper = all_points.min(axis=0)-0.18, all_points.max(axis=0)+0.18
    plot_lower, plot_upper = np.array([55., 130.]), np.array([850., 800.])
    scale = float(np.min((plot_upper-plot_lower)/(upper-lower)))
    anchor = (plot_lower+plot_upper)/2-(lower+upper)/2*scale
    return scale, anchor


def label(canvas, text, at, color=(225, 225, 225), size=0.58, width=1):
    cv2.putText(canvas, text, at, cv2.FONT_HERSHEY_SIMPLEX, size, color, width, cv2.LINE_AA)


def render_step(sequences, step, scale, anchor, seed=42, explicit=False):
    canvas = np.full((HEIGHT, WIDTH, 3), (25, 23, 22), dtype=np.uint8)
    selection = "selected" if explicit else "random"
    label(canvas, f"5 {selection} kicks | 8-joint normalized skeleton overlap", (30, 40), size=0.85, width=2)
    label(canvas, "Hip-centered / fixed torso scale / no extra rotation or interpolation", (30, 75), size=0.56)
    cv2.rectangle(canvas, (35, 110), (870, 820), (65, 61, 58), 1)
    origin = tuple(np.rint(anchor).astype(int))
    cv2.line(canvas, (35, origin[1]), (870, origin[1]), (58, 54, 51), 1)
    cv2.line(canvas, (origin[0], 110), (origin[0], 820), (58, 54, 51), 1)
    cv2.drawMarker(canvas, origin, (155, 155, 155), cv2.MARKER_CROSS, 18, 1)
    label(canvas, "pelvis (0, 0)", (50, origin[1]-10), color=(150, 150, 150), size=0.45)

    # Alpha-blended skeletons let intersecting poses remain visible. The same
    # physical coordinate scale is used for x and y throughout the entire video.
    for item, color in zip(sequences, COLORS):
        points = np.rint(item["poses"][step]*scale+anchor).astype(int)
        lookup = {joint: tuple(points[index]) for index, joint in enumerate(JOINTS)}
        overlay = canvas.copy()
        for a, b in EDGES:
            cv2.line(overlay, lookup[a], lookup[b], color, 3, cv2.LINE_AA)
        for point in points:
            cv2.circle(overlay, tuple(point), 5, color, -1, cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0)

    legend = "SELECTED KICKS" if explicit else f"SELECTED KICKS (seed {seed})"
    label(canvas, legend, (895, 135), size=0.57, width=2)
    for index, (item, color) in enumerate(zip(sequences, COLORS)):
        y = 185+index*100
        dataset, kick = item["key"]
        row = item["rows"][step]
        cv2.line(canvas, (895, y-7), (915, y-7), color, 4, cv2.LINE_AA)
        label(canvas, f"{dataset} / {kick}", (925, y), color, size=0.57, width=2)
        label(canvas, f"Source frame: {row['frame']}", (925, y+27), size=0.47)
        label(canvas, f"Camera: {row.get('camera_direction', '')}", (925, y+49), size=0.47)
    label(canvas, "Shoulders / hips / knees / ankles", (895, 725), size=0.44)
    label(canvas, "Aligned by sample progress", (895, 757), size=0.48)
    label(canvas, "NOT synchronized real time", (895, 784), size=0.48)
    label(canvas, f"Sample {step+1:02d} / 20", (35, 860), size=0.7, width=2)
    cv2.rectangle(canvas, (260, 844), (860, 857), (65, 61, 58), -1)
    cv2.rectangle(canvas, (260, 844), (260+int(600*step/19), 857), (160, 150, 135), -1)
    if step == 0:
        label(canvas, "FIRST FRAME", (925, 860), size=0.67, width=2)
    elif step == 19:
        label(canvas, "LAST TOUCH = 1", (925, 860), (100, 215, 255), size=0.67, width=2)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kick", action="append", metavar="DATASET:KICK_ID",
                        help="Repeat five times to select exact kicks in legend/color order")
    args = parser.parse_args()
    output = args.output
    preview, report = output.with_suffix(".png"), output.with_suffix(".json")
    if any(path.exists() for path in (output, preview, report)):
        raise FileExistsError("Refusing to overwrite an existing visualization")
    # The conda ffmpeg may lack libx264 even when system ffmpeg supports it.
    encoder = None
    for executable in dict.fromkeys(("/usr/bin/ffmpeg", shutil.which("ffmpeg"))):
        if executable and Path(executable).is_file():
            supported = subprocess.run([executable, "-hide_banner", "-encoders"],
                                       capture_output=True, text=True, check=True)
            if "libx264 " in supported.stdout:
                encoder = executable
                break
    if encoder is None:
        raise RuntimeError("An ffmpeg build with libx264 is required")
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    chosen_keys = None
    if args.kick:
        if any(":" not in value for value in args.kick):
            raise ValueError("Use --kick DATASET:KICK_ID")
        chosen_keys = [tuple(value.split(":", 1)) for value in args.kick]
    sequences = read_selected(args.input, args.seed, chosen_keys)
    scale, anchor = shared_transform(sequences)
    images = [render_step(sequences, step, scale, anchor, args.seed, chosen_keys is not None) for step in range(20)]
    # Static holds only slow the playback: no synthetic intermediate pose is
    # created. Repeat twice to make the short 20-pose sequence easier to inspect.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pose_overlap_") as directory:
        intermediate = Path(directory)/"render.mp4"
        writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"),
                                 FPS, (WIDTH, HEIGHT))
        if not writer.isOpened():
            raise RuntimeError("Could not open video writer")
        try:
            for _ in range(REPEATS):
                for image in images:
                    for _ in range(HOLD_PER_STEP):
                        writer.write(image)
                for _ in range(CONTACT_HOLD):
                    writer.write(images[-1])
        finally:
            writer.release()
        subprocess.run([encoder, "-hide_banner", "-loglevel", "error", "-n", "-i", str(intermediate),
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(output)], check=True)
    if not cv2.imwrite(str(preview), images[-1]):
        raise RuntimeError("Could not save preview")
    assert hashlib.sha256(args.input.read_bytes()).hexdigest() == input_hash
    metadata = {"input": str(args.input), "input_sha256": input_hash,
                "seed": args.seed if chosen_keys is None else None,
                "selection_mode": "random" if chosen_keys is None else "explicit",
                "output": str(output), "preview": str(preview), "joints": list(JOINTS),
                "coordinate_columns": "kp_*_xn / kp_*_yn (unchanged)",
                "selected_kicks": [{"dataset": item["key"][0], "kick_id": item["key"][1],
                                    "source_frames": [row["frame"] for row in item["rows"]]}
                                   for item in sequences],
                "alignment": "20 ordered sample indices: first frame through last_touch; NOT real-time alignment",
                "extra_coordinate_interpolation": False, "additional_pose_normalization": False,
                "display_pixels_per_normalized_unit": scale, "fps": FPS, "repeats": REPEATS,
                "video_frames": REPEATS*(20*HOLD_PER_STEP+CONTACT_HOLD),
                "duration_seconds": REPEATS*(20*HOLD_PER_STEP+CONTACT_HOLD)/FPS,
                "input_file_unchanged": True}
    report.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
