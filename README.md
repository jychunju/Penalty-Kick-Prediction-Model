# Penalty Kick Prediction — Final Pre3 ST-GCN

This repository contains the **final central-camera, six-joint ST-GCN**, its
trained checkpoint, and the preprocessing, training, and inference code needed
to use it. Earlier model comparisons and their checkpoints have been removed
from the current branch.

## Model

The model predicts the canonicalized kick direction from the kicker's
perspective: **L = 0, R = 1**. Center-directed kicks are excluded.

| Property | Final model |
| --- | --- |
| Input | `(batch, 3, 12)` normalized XY coordinates |
| Frames | Consecutive original frames `[t-2, t-1, t]`, including annotated contact `t` |
| COCO joints, in order | 11, 12, 13, 14, 15, 16: left/right hips, knees, ankles |
| Feature order | `kp_11_xn, kp_11_yn, kp_12_xn, kp_12_yn, ..., kp_16_xn, kp_16_yn` |
| Graph | Hip-to-hip, hip-to-knee, and knee-to-ankle connections |
| ST-GCN widths | 16 → 16 → 32; temporal kernels of 5; final temporal stride of 2 |
| Head | Global average pooling, dropout 0.3, and a two-class linear layer |
| Trainable parameters | 11,066 |
| Checkpoint | `weights/final_pre3.pt`, seed 42, selected epoch 25 |

Only the twelve coordinate features enter the model. Angles, velocities, camera
labels, dataset identifiers, contact flags, and outcomes are not model inputs.

## Setup

Python 3.10 was used to validate this package.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
```

The requirements list direct dependencies; it is not a version-locked environment.
Pose extraction additionally requires FFmpeg and separately obtained
`yolov8m-pose.pt` and `yolov8m.pt` pretrained weights. These are not the final
classifier checkpoint and are not bundled here.

## Inference

Prepare a NumPy `.npy` array with shape `(3, 12)` for one kick or `(N, 3, 12)`
for multiple kicks, using the preprocessing and feature order described below.

```bash
python predict.py normalized_pre3.npy
```

The command prints JSON with `label`, `prob_L`, and `prob_R` for each kick.
It runs on CPU and does not require the training datasets.

```python
import numpy as np
from predict import predict

inputs = np.load("normalized_pre3.npy", allow_pickle=False)
predictions = predict(inputs)
```

The checkpoint retains its original training metadata, which includes NumPy
objects. The loader uses `weights_only=False`; load only checkpoints you trust.
The exact original checkpoint is preserved, and its SHA-256 is recorded in
`weights/metadata.json`.

## Preprocessing

The saved classifier expects coordinates processed in the same way as training:

1. Extract the reviewed kicker's pose and retain the source frame numbers,
   camera metadata, kicking foot, and contact annotation. The 150penalties
   extractor uses the reviewed ROI/identity/contact annotations in
   `annotations/150penalties_segments.json`.
2. Apply the dataset's original missing-coordinate handling. The 150penalties
   pipeline trims leading frames until the eight selected joints are observed,
   fills internal bracketed gaps, and does not extrapolate boundary joints.
3. Canonicalize left-foot kicks to a right-foot representation using the original
   reflection and anatomical left/right swaps. Swap the L/R target labels too.
4. Center each frame at the midpoint of the hips. Divide coordinates by the
   median shoulder-midpoint-to-hip-midpoint distance over the **full retained
   sequence**. Shoulders determine this scale even though the final model uses
   only hips, knees, and ankles.
5. Reproduce the original 20-frame eligibility table and matching source
   sequence, then select consecutive source frames `[t-2, t-1, t]`. These are
   not the last three uniformly sampled frames. The original common eligibility
   rule also requires a complete contact-inclusive five-frame window.
6. Keep camera C for EPL, Women, and 150penalties. CLT is treated as camera C
   under the original dataset convention. Keep only L/R kicks and select the
   twelve normalized coordinates in the exact order listed above.

The preprocessing code is retained as separate stages:

| Files | Purpose |
| --- | --- |
| `extract_150penalties_pose.py`, `extract_penaltyfirst15_pose.py`, `reviewed_pose_inference.py` | Reviewed video-to-pose extraction and its shared implementation |
| `preprocess_150penalties.py`, `preprocess_penaltyfirst15.py` | Dataset adaptation, interpolation, canonicalization, sampling, and shared angle/schema helpers |
| `canonicalize_main_data.py`, `merge_main_data.py` | Original legacy dataset canonicalization and integration |
| `add_kicking_foot.py`, `normalize_combined_20frames.py`, `interpolate_lower_body.py` | Legacy preprocessing utilities |
| `spatial_normalize_pose.py` | Pelvis centering and full-sequence median torso normalization |
| `extract_uniform_20frames.py`, `extract_8_joint_pose.py` | Original eligibility/sampling table and intermediate joint selection |
| `merge_model_inputs.py` | Concatenate the legacy and processed 150penalties tables with a union schema |
| `prepare_windows.py` | Recover the matching source sequence and final consecutive pre3 window |

Some intermediate tables contain eight joints and angle columns to preserve the
original preprocessing and eligibility checks. These extra columns are discarded
before the final model receives its input. `preprocess_penaltyfirst15.py` also
provides shared functions imported by the 150penalties preprocessor; its standalone
first15 workflow is not required to run inference.

The scripts require the corresponding source CSVs and reviewed annotations;
they do not download or recreate missing datasets. Use each script's `--help`
for available options. For example, after preparing the original legacy table:

```bash
python extract_150penalties_pose.py --stage all --workers 1
python preprocess_150penalties.py --assume-clip-end-contact
python merge_model_inputs.py
```

The `--assume-clip-end-contact` flag reproduces the original treatment of clips
whose contact endpoint was uncertain; it must not be interpreted as visually
verified contact. The legacy reference table must already exist before running
the 150penalties preprocessing checks.

## Training

The original training protocol is retained in `train.py`; `config.json` selects
only the final contact-inclusive pre3 model. Place these files under `Main Data/`:

```text
Combined_penalty_pose_keypoints_normalized.csv
150penalties_pose_keypoints_normalized.csv
Combined_penalty_pose_keypoints_normalized_20frames_8joints_angles_scaled_with_150penalties.csv
```

The last file is produced by `merge_model_inputs.py` from the two processed
20-frame source tables. Training also reads the first two full-frame tables to
recover the exact consecutive contact window.

```bash
python train.py --dry-run
python train.py --config config.json
```

`--dry-run` validates inputs and model shape without training or writing results;
it still requires the input data. The config defaults to CPU and new output
paths. Set `device` to an available CUDA device for GPU training. Existing output
directories are protected from overwrite.

EPL, Women, and 150penalties are split at the kick level into 132 training,
28 validation, and 28 integrated-test kicks. CLT is a separate 21-kick external
holdout. AdamW uses learning rate 0.001, weight decay 0.001, batch size 16,
train-only class weights, and gradient clipping at 1.0. Training runs for up to
150 epochs with 25 epochs of early-stopping patience. Checkpoint selection uses
validation macro F1, with validation loss as the tie-breaker. Test sets do not
control checkpoint selection.

## Saved Model Results

| Split | Kicks | Accuracy |
| --- | ---: | ---: |
| Training | 132 | 94.70% |
| Validation | 28 | 100.00% |
| Integrated test | 28 | 85.71% (24/28) |
| External CLT test | 21 | 61.90% (13/21) |

Metrics and original configuration are recorded in `weights/metadata.json`.
Historical absolute paths in that file identify the original training sources;
current runtime paths are configured in `config.json`.

This is a single-seed exploratory result with kick-level splits, not a
player- or match-disjoint evaluation. Contact is included, and the normalization
scale uses the full retained sequence. The model therefore represents a
contact-aligned retrospective experiment; it does not establish strictly
pre-contact or live prediction performance. Output directions use the
right-foot canonicalized convention, so left-foot results must be mapped back
if original-foot directions are needed.

Raw videos, dataset CSVs, generated predictions, and unrelated experiment weights
are not included. No new training was performed while packaging this repository.

## Acknowledgments

Special thanks to **Illia Reshtanenko**
([Football RE FOUNDS](https://www.youtube.com/channel/UCsy6t5_MuFJSjwAGSUVXsxQ))
for collecting and editing the penalty kick video data used in this project.
