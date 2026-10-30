# YOLOv8 Pose short-clip validation

The extractor keeps all tracked people and all 17 COCO keypoints. It does not
silently treat the top-ranked track as the kicker; inspect the overlay first.

```bash
python extract_yolov8_pose.py \
  YTDown.com_YouTube_Media_VGF4MkWO40Y_Every-Penalty-Goal-from-Last-Season-Serie-A-2022-23_001_1080p.mp4 \
  --start 12 --duration 21 \
  --model yolov8m-pose.pt --device 0 --imgsz 960 \
  --output-dir runs/yolov8_pose_preview
```

Outputs:

- `pose_overlay.mp4`: every pose and ByteTrack ID drawn on the source clip
- `pose_keypoints_raw.csv`: boxes plus all COCO keypoint coordinates/confidence
- `track_candidates.csv`: auditable heuristic ranking for kicker candidates
- `metadata.json`: model and source-video provenance

After a human confirms the kicker track and contact frame, create the ST-GCN
sample from a fixed pre-contact time window. Do not count replay views as new
penalty kicks.
