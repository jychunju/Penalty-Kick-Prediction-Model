#!/usr/bin/env python3
"""YOLOv8 kicker-only pose graphs with explicit, reviewed source-frame bounds.

Inference is cached independently of segment selection. No missing joints are
invented or interpolated; graph tensors carry an explicit validity mask.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle")
EDGES = np.array([(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7),
                  (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                  (11, 13), (13, 15), (12, 14), (14, 16)], dtype=np.int64)
FFMPEG = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else (shutil.which("ffmpeg") or "ffmpeg")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty output: {path}")
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames or fps <= 0:
        raise RuntimeError(f"Unreadable video: {path}")
    return fps, frames


def infer(video, spec, output, pose_model, ball_model, args):
    fps, frames = load_video(video)
    n = len(frames)
    x1, y1, x2, y2 = spec["roi_xyxy"]
    offset = np.array([x1, y1], dtype=np.float32)
    xy = np.full((n, 17, 2), np.nan, dtype=np.float32)
    conf = np.zeros((n, 17), dtype=np.float32)
    boxes = np.full((n, 4), np.nan, dtype=np.float32)
    person_conf = np.zeros(n, dtype=np.float32)
    det_index = np.full(n, -1, dtype=np.int32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)
    ball_conf = np.zeros(n, dtype=np.float32)
    prev = np.array(spec["seed_center_xy"], dtype=float)
    velocity = np.zeros(2)
    prev_h = None
    audit = []
    for start in range(0, n, args.batch):
        batch = [im[y1:y2, x1:x2] for im in frames[start:start + args.batch]]
        results = pose_model.predict(batch, imgsz=args.imgsz, conf=.15,
                                     device=args.device, verbose=False)
        balls = ball_model.predict(batch, imgsz=args.imgsz, conf=.04,
                                   classes=[32], device=args.device, verbose=False)
        for local, (res, br) in enumerate(zip(results, balls)):
            fi = start + local
            candidates = []
            if res.boxes is not None and res.keypoints is not None:
                bb = res.boxes.xyxy.cpu().numpy() + np.tile(offset, 2)
                bc = res.boxes.conf.cpu().numpy()
                pts = res.keypoints.data.cpu().numpy().copy()
                pts[:, :, :2] += offset
                scores = []
                for j, box in enumerate(bb):
                    center = (box[:2] + box[2:]) / 2
                    h = max(1., box[3] - box[1])
                    dist = np.linalg.norm(center - (prev + velocity)) / max(h, prev_h or h)
                    scale_penalty = 0 if prev_h is None else .5 * abs(np.log(h / prev_h))
                    score = dist + scale_penalty - .12 * bc[j]
                    scores.append(score)
                    candidates.append({"index": j, "bbox": box.tolist(),
                                       "confidence": float(bc[j]), "association_score": float(score)})
                if scores:
                    j = int(np.argmin(scores))
                    # Reject implausible identity jumps rather than selecting a teammate.
                    if scores[j] < 1.5:
                        center = (bb[j, :2] + bb[j, 2:]) / 2
                        velocity = .5 * velocity + .5 * (center - prev)
                        prev = center
                        prev_h = max(1., bb[j, 3] - bb[j, 1])
                        xy[fi] = pts[j, :, :2]
                        conf[fi] = pts[j, :, 2]
                        boxes[fi] = bb[j]
                        person_conf[fi] = bc[j]
                        det_index[fi] = j
            ball_candidates = []
            if br.boxes is not None:
                for box, confidence in zip(br.boxes.xyxy.cpu().numpy(), br.boxes.conf.cpu().numpy()):
                    center = (box[:2] + box[2:]) / 2 + offset
                    ball_candidates.append({"xy": center.tolist(), "confidence": float(confidence)})
                if ball_candidates and np.isfinite(boxes[fi]).all():
                    # Ball labels are diagnostics, never a substitute for boundary review.
                    ankles = xy[fi, 15:17][conf[fi, 15:17] >= .25]
                    target = ankles.mean(0) if len(ankles) else prev
                    candidate = min(ball_candidates, key=lambda b: np.linalg.norm(np.array(b["xy"]) - target))
                    ball_xy[fi] = candidate["xy"]
                    ball_conf[fi] = candidate["confidence"]
            audit.append({"source_frame": fi, "people": candidates,
                          "selected_detection": int(det_index[fi]), "balls": ball_candidates})
        print(f"  {video.name}: inferred {min(start + args.batch, n)}/{n}", flush=True)
    np.savez_compressed(output / "inference_all_frames.npz", xy=xy, confidence=conf,
                        boxes=boxes, person_confidence=person_conf, detection_index=det_index,
                        ball_xy=ball_xy, ball_confidence=ball_conf, fps=fps)
    (output / "detection_audit.json").write_text(json.dumps(audit))


def encode(temp, dest, fps=None):
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(temp)]
    if fps is not None:
        cmd += ["-vf", "setpts=2*PTS", "-r", str(fps)]
    cmd += ["-an", "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(dest)]
    subprocess.run(cmd, check=True)


def render(video, spec, output, args):
    fps, frames = load_video(video)
    cache = np.load(output / "inference_all_frames.npz")
    start, contact = spec["runup_start_frame"], spec["contact_frame"]
    if not (0 <= start <= contact < len(frames)):
        raise ValueError(f"Invalid boundaries for {video.name}: {start}, {contact}")
    indices = np.arange(start, contact + 1)
    xy, conf = cache["xy"][indices].copy(), cache["confidence"][indices]
    boxes, pconf = cache["boxes"][indices], cache["person_confidence"][indices]
    h, w = frames[0].shape[:2]
    valid = (conf >= args.keypoint_conf) & np.isfinite(xy).all(2)
    valid &= (xy[:, :, 0] > 0) & (xy[:, :, 0] < w) & (xy[:, :, 1] > 0) & (xy[:, :, 1] < h)
    # Occluded people can inherit a nearby player's leg despite high confidence.
    # Flag joints outside the kicker box plus a modest margin as missing.
    box_size = boxes[:, 2:] - boxes[:, :2]
    lo, hi = boxes[:, :2] - .15 * box_size, boxes[:, 2:] + .15 * box_size
    inside_box = (xy >= lo[:, None]).all(2) & (xy <= hi[:, None]).all(2)
    outside_bbox_count = int((valid & ~inside_box).sum())
    valid &= inside_box
    masked = np.where(valid[:, :, None], xy, np.nan)
    hip_ok = valid[:, 11] & valid[:, 12]
    root = np.where(hip_ok[:, None], (masked[:, 11] + masked[:, 12]) / 2, np.nan)
    heights = boxes[:, 3] - boxes[:, 1]
    scale = float(np.nanmedian(heights))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"No usable kicker scale: {video.name}")
    normalized = (masked - root[:, None]) / scale
    normalized_valid = valid & hip_ok[:, None]
    velocity = np.full_like(normalized, np.nan)
    velocity[1:] = np.diff(normalized, axis=0) * fps
    adjacency = np.zeros((17, 17), dtype=np.uint8)
    for a, b in EDGES:
        adjacency[a, b] = adjacency[b, a] = 1
    np.savez_compressed(output / "kicker_graph_17j.npz", xy_pixels=masked,
                        xy_normalized=normalized, velocity_normalized_per_sec=velocity,
                        confidence=conf, valid=valid, normalized_valid=normalized_valid,
                        edge_index=EDGES.T, adjacency=adjacency, source_frames=indices,
                        time_sec=indices / fps, joint_names=np.array(NAMES),
                        is_contact=(indices == contact) & (spec["contact_status"] != "uncertain_clip_end"),
                        is_contact_candidate=indices == contact, fps=fps, scale_px=scale)
    rows, graph_rows = [], []
    for t, fi in enumerate(indices):
        contact_marked = spec["contact_status"] != "uncertain_clip_end"
        common = {"video": video.name, "source_frame": int(fi), "time_sec": fi / fps,
                  "clip_frame": t, "is_contact": int(fi == contact and contact_marked),
                  "is_contact_candidate": int(fi == contact),
                  "pose_detected": int(np.isfinite(boxes[t]).all())}
        row = {**common, "person_conf": float(pconf[t]),
               "detection_index": int(cache["detection_index"][fi])}
        for k, value in zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), boxes[t]):
            row[k] = float(value) if np.isfinite(value) else ""
        row["ball_x"], row["ball_y"] = [float(v) if np.isfinite(v) else "" for v in cache["ball_xy"][fi]]
        row["ball_conf"] = float(cache["ball_confidence"][fi])
        gr = dict(common)
        for j, name in enumerate(NAMES):
            row[f"{name}_x"] = float(masked[t, j, 0]) if valid[t, j] else ""
            row[f"{name}_y"] = float(masked[t, j, 1]) if valid[t, j] else ""
            row[f"{name}_conf"] = float(conf[t, j])
            row[f"{name}_valid"] = int(valid[t, j])
            gr[f"{name}_conf"] = float(conf[t, j])
            gr[f"{name}_valid"] = int(normalized_valid[t, j])
            for axis, letter in enumerate("xy"):
                v = normalized[t, j, axis]
                gr[f"{name}_{letter}_norm"] = float(v) if np.isfinite(v) else ""
                v = velocity[t, j, axis]
                gr[f"{name}_v{letter}_per_sec"] = float(v) if np.isfinite(v) else ""
        rows.append(row)
        graph_rows.append(gr)
    write_csv(output / "kicker_keypoints_17j.csv", rows)
    write_csv(output / "kicker_graph_17j.csv", graph_rows)
    temp = output / "overlay_temp.mp4"
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Video writer failed: {temp}")
    review_indices = set(np.rint(np.linspace(0, len(indices) - 1, 8)).astype(int).tolist())
    thumbnails = []
    contact_still = None
    for t, fi in enumerate(indices):
        im = frames[fi].copy()
        for a, b in EDGES:
            if valid[t, a] and valid[t, b]:
                color = (0, 255, 255) if a < 11 else (0, 220, 0)
                cv2.line(im, tuple(np.rint(masked[t, a]).astype(int)),
                         tuple(np.rint(masked[t, b]).astype(int)), color, 3, cv2.LINE_AA)
        for j in np.flatnonzero(valid[t]):
            cv2.circle(im, tuple(np.rint(masked[t, j]).astype(int)), 4, (0, 70, 255), -1, cv2.LINE_AA)
        if np.isfinite(boxes[t]).all():
            x1, y1, x2, y2 = np.rint(boxes[t]).astype(int)
            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(im, "KICKER", (x1, max(25, y1 - 10)), 0, .7, (255, 180, 0), 2, cv2.LINE_AA)
        cv2.rectangle(im, (0, 0), (w, 88), (20, 20, 20), -1)
        cv2.putText(im, f"{video.stem} | source frame {fi} | t={fi / fps:.2f}s | joints {valid[t].sum()}/17",
                    (20, 34), 0, .8, (255, 255, 255), 2, cv2.LINE_AA)
        end_label = "CONTACT" if spec["contact_status"] == "visually_reviewed" else "CLIP END (contact uncertain)"
        cv2.putText(im, f"RUN-UP {start} -> {end_label} {contact} (inclusive) | yellow: torso/arms, green: legs",
                    (20, 68), 0, .7, (230, 230, 230), 2, cv2.LINE_AA)
        if fi == contact:
            cv2.rectangle(im, (0, h - 65), (w, h), (0, 0, 170), -1)
            label = "CONTACT FRAME (visual estimate +/-1 frame)" if spec["contact_status"] == "visually_reviewed" else "CLIP END - CONTACT UNCERTAIN"
            cv2.putText(im, label, (25, h - 22), 0, 1., (255, 255, 255), 2, cv2.LINE_AA)
            contact_still = im
        writer.write(im)
        if t in review_indices:
            # Crop the inspected kicker region at a readable size for QA.
            rx1, ry1, rx2, ry2 = spec["roi_xyxy"]
            thumb = cv2.resize(im[ry1:ry2, rx1:rx2], (480, 270))
            thumb = cv2.copyMakeBorder(thumb, 0, 30, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(thumb, f"frame {fi} | valid {valid[t].sum()}/17", (8, 292), 0, .55, (255, 255, 255), 1)
            thumbnails.append(thumb)
    writer.release()
    encode(temp, output / "kicker_overlay.mp4")
    encode(temp, output / "kicker_overlay_0.5x.mp4", fps)
    rx1, ry1, rx2, ry2 = spec["roi_xyxy"]
    zoom_filter = (f"[0:v]split=2[full][zoom];[full]scale=1280:720[a];"
                   f"[zoom]crop={rx2-rx1}:{ry2-ry1}:{rx1}:{ry1},"
                   "scale=640:640:force_original_aspect_ratio=decrease,"
                   "pad=640:720:(ow-iw)/2:(oh-ih)/2[b];"
                   "[a][b]hstack=inputs=2,setpts=2*PTS[out]")
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(temp), "-filter_complex", zoom_filter, "-map", "[out]", "-r", str(fps),
                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(output / "kicker_review_0.5x.mp4")], check=True)
    temp.unlink()
    if contact_still is not None:
        cv2.imwrite(str(output / "contact_frame.jpg"), contact_still)
    while len(thumbnails) < 8:
        thumbnails.append(np.zeros((300, 480, 3), np.uint8))
    sheet = np.vstack([np.hstack(thumbnails[:4]), np.hstack(thumbnails[4:8])])
    cv2.imwrite(str(output / "pose_review.jpg"), sheet)
    lower = valid[:, 11:17]
    meta = {"video": video.name, "input": str(video.resolve()), "source_fps": fps,
            "source_frames": len(frames), "source_size": [w, h],
            **spec, "extracted_frames": len(indices),
            "segment_end_frame": contact, "contact_candidate_frame": contact,
            "contact_frame": contact if spec["contact_status"] != "uncertain_clip_end" else None,
            "contact_included": spec["contact_status"] != "uncertain_clip_end",
            "pose_detected_frames": int(np.isfinite(boxes).all(1).sum()),
            "joint_valid_fraction": float(valid.mean()), "lower_body_valid_fraction": float(lower.mean()),
            "contact_valid_joints": int(valid[-1].sum()),
            "missing_pose_source_frames": indices[~np.isfinite(boxes).all(1)].tolist(),
            "normalization": "per-frame hip midpoint, divided by median segment bbox height",
            "scale_px": scale, "missing_values": "CSV empty / NPZ NaN, explicit validity masks; no interpolation",
            "model": args.pose_model, "ball_model": args.ball_model, "imgsz": args.imgsz,
            "keypoint_conf_threshold": args.keypoint_conf, "ultralytics": __import__("ultralytics").__version__,
            "outside_bbox_masked_joint_count": outside_bbox_count,
            "spatial_validity": "joint must lie inside kicker bbox expanded 15 percent in x/y",
            "association": "visually specified kicker seed, temporal center and bbox size continuity within ROI",
            "coordinate_system": "2D original source pixels; anatomical left/right, no mirroring",
            "torch": torch.__version__}
    (output / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("Main Data/penaltyfirst15"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/penaltyfirst15_pose"))
    p.add_argument("--manifest", type=Path)
    p.add_argument("--stage", choices=["infer", "render", "all"], default="all")
    p.add_argument("--pose-model", default="yolov8m-pose.pt")
    p.add_argument("--ball-model", default="yolov8m.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--keypoint-conf", type=float, default=.25)
    p.add_argument("--only-video", help="Process one filename; rebuild summaries from existing outputs")
    a = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.manifest or a.output_dir / "segments.json").read_text())
    actual = {x.name for x in a.input_dir.glob("*.mp4")}
    if actual != set(manifest):
        raise ValueError(f"Manifest/input mismatch: {actual ^ set(manifest)}")
    if a.only_video and a.only_video not in manifest:
        raise ValueError(f"Unknown video: {a.only_video}")
    models = None
    if a.stage in ("infer", "all"):
        models = (YOLO(a.pose_model), YOLO(a.ball_model))
    summaries, combined = [], []
    for name, spec in manifest.items():
        if a.only_video and name != a.only_video:
            continue
        video = a.input_dir / name
        output = a.output_dir / video.stem
        output.mkdir(parents=True, exist_ok=True)
        print(f"Processing {name} ({a.stage})", flush=True)
        if models:
            infer(video, spec, output, *models, a)
        if a.stage in ("render", "all"):
            meta, rows = render(video, spec, output, a)
            summaries.append(meta)
            combined.extend(rows)
    schema = {"joint_names": list(NAMES), "edge_index": EDGES.T.tolist(),
              "layout": "frames x 17 joints x 2 coordinates",
              "confidence_layout": "frames x 17", "source_frame_indexing": "zero-based, end inclusive",
              "source_documentation": "https://docs.ultralytics.com/tasks/pose/"}
    (a.output_dir / "graph_schema.json").write_text(json.dumps(schema, indent=2))
    if summaries:
        # Include existing per-video results when rendering a corrected clip.
        summaries, combined = [], []
        for name in manifest:
            output = a.output_dir / Path(name).stem
            if (output / "metadata.json").exists():
                summaries.append(json.loads((output / "metadata.json").read_text()))
                with (output / "kicker_keypoints_17j.csv").open(newline="") as f:
                    combined.extend(csv.DictReader(f))
        (a.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
        write_csv(a.output_dir / "all_kicker_keypoints_17j.csv", combined)
        write_csv(a.output_dir / "quality_summary.csv", [{k: m[k] for k in
                  ("video", "runup_start_frame", "contact_frame", "extracted_frames",
                   "contact_status", "pose_detected_frames", "joint_valid_fraction",
                   "lower_body_valid_fraction", "contact_valid_joints")} for m in summaries])
        if len(summaries) == len(manifest):
            clips = [a.output_dir / Path(name).stem / "kicker_review_0.5x.mp4" for name in manifest]
            concat_file = a.output_dir / "review_concat.txt"
            # ffconcat escaping for paths; subprocess arguments are never shell text.
            concat_file.write_text("".join("file '" + str(path.resolve()).replace("'", "'\\''") + "'\n" for path in clips))
            subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
                            "-safe", "0", "-i", str(concat_file), "-c", "copy", "-movflags", "+faststart",
                            str(a.output_dir / "all_kickers_review_0.5x.mp4")], check=True)
            shutil.copyfile(a.output_dir / "all_kicker_keypoints_17j.csv",
                            a.input_dir.parent / f"{a.input_dir.name}_pose_keypoints.csv")
            shutil.copyfile(a.output_dir / "summary.json",
                            a.input_dir.parent / f"{a.input_dir.name}_pose_summary.json")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
