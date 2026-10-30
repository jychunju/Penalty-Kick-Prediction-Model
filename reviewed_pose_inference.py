"""Existing YOLO inference with optional visually reviewed identity anchors.

Anchors select a detection only; no joint coordinates are interpolated.
The original first15 extractor and its default association remain unchanged.
"""
from extract_penaltyfirst15_pose import *
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
    anchors = np.array(spec.get("association_anchors", []), dtype=float)
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
                    if len(anchors):
                        target = np.array([np.interp(fi, anchors[:, 0], anchors[:, 1]), np.interp(fi, anchors[:, 0], anchors[:, 2])])
                        anchor_distance = np.linalg.norm(center - target) / h
                        score = anchor_distance + .15 * min(dist, 2.) - .12 * bc[j]
                        if anchor_distance > .85:
                            score = 10. + anchor_distance
                    scores.append(score)
                    candidates.append({"index": j, "bbox": box.tolist(),
                                       "confidence": float(bc[j]), "association_score": float(score),
                                       "keypoints": pts[j].tolist()})
                if scores:
                    j = int(np.argmin(scores))
                    # Reject implausible identity jumps rather than selecting a teammate.
                    if scores[j] < 1.5 and fi not in spec.get("rejected_detection_frames", []):
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
