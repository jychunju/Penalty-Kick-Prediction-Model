# YSAL: proposed four-channel ST-GCN baseline

Status: awaiting user confirmation. No training, optimizer steps, checkpoints,
package installation, or environment changes have been performed.
Only model architecture and a proposed configuration are provided so far;
the data loader and training runner are not implemented yet.

## Task and scope

Predict L/C/R from the kicker's perspective, using one observed sequence of
12 joints. Labels: L=0, C=1, R=2. Keep all work and outputs inside YSAL.
The last_touch marker denotes LAST ball contact, not necessarily contact onset.
This baseline is therefore an annotated-last-contact-aligned retrospective
experiment, not a demonstrated live, strictly pre-first-contact predictor.

## Data audit and proposed sampling adjustment

The CSV has 4,396 rows, 88 kicks, and 103 candidate groups defined by
(kick_id, camera_direction, track_id). These groups are NOT verified clip IDs.
No attempt will be made to join different tracks or camera views automatically.
Exclude a group if its frame field is invalid, duplicated, or non-increasing in
original row order, or it does not have exactly one last_touch=1 marker.
These checks reduce ambiguity but cannot prove clip identity without sources.

For each remaining group, end at contact-1 and take 16 samples with fixed stride.
Require every original frame row in the entire selected interval to exist.
Do not compress time across gaps or pad short clips in the first experiment.
Fill internal joint-coordinate gaps of at most 4 original frames by linear
interpolation; boundary gaps of at most 2 frames use nearest observed values.
Reject any larger run, or total missing joint fraction above 20% in the interval.
All imputation endpoints must be INSIDE the observation interval. Boundary
holding is an explicit approximation, not a recovered observation.

Audited candidate counts under these exact missing-coordinate rules:

| Stride | Interval | Candidate sequences | Unique kicks | Kick labels L/C/R |
| --- | --- | ---: | ---: | --- |
| 1 | contact-16 through contact-1 | 44 | 41 | 19 / 7 / 15 |
| 2 | contact-31 through contact-1 | 30 | 30 | 11 / 5 / 14 |

Propose stride=1 for the first run to retain more independent kicks. This is a
change from the earlier illustrative stride=2 suggestion and needs confirmation.
Counts are preliminary until finite-number, bbox-scale, and label consistency
checks also pass in the loader. Record all inclusion/exclusion reasons in a
manifest. If stricter checks materially change counts, report before training.

## Input preprocessing

1. Select the permitted temporal interval before any filling or normalization.
2. Validate numeric coordinates, positive bbox height, and consistent labels.
3. Apply the limited imputation policy above, retaining provenance internally.
4. At each time, subtract the midpoint of kp_11/kp_12 from all joint positions.
5. Divide all frames by one scale: median bbox_h in this observation interval.
6. Select 16 frames in chronological order. Compute backward differences of
   normalized positions divided by ORIGINAL frame-number difference.
7. Set both velocities at the first selected frame to zero. Other velocities
   are derivatives of the reconstructed trajectories, including limited
   imputation; they are not claimed to be measured physical velocities.

FPS is not verified, so velocities are in normalized coordinate per original
frame, not per second or m/s. This supports fixed-FPS comparisons without
inventing a frame rate. Verify common FPS from source metadata when available.
Input shape is (N,4,16,12,1), channels [x,y,vx,vy]. No confidence/mask channel,
kicking-foot metadata, camera code, goal, track ID, or contact marker is fed in.
The joint order is kp_5 through kp_16. There is no coordinate rotation or flip.

## Architecture

model.py is a compact ST-GCN variant, not an exact reproduction of the original
large architecture. It uses spatial graph convolution, temporal convolution,
BatchNorm, residual connections, and dropout, following the broad structure at
https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py .

Graph edges: bilateral shoulders/hips, shoulder-elbow-wrist,
shoulder-hip, hip-knee-ankle. Three fixed adjacency partitions aggregate
same-distance (including self), nearer, and farther nodes relative to the two
hip roots. Each partition uses incoming-degree normalization; no adaptive edges.

| Stage | Output excluding singleton person axis |
| --- | --- |
| Per-joint/channel input BatchNorm | N x 4 x 16 x 12 |
| Spatial GCN + temporal kernel 5 + residual | N x 32 x 16 x 12 |
| Spatial GCN + temporal kernel 5 + residual | N x 32 x 16 x 12 |
| Spatial GCN + temporal kernel 5, stride 2 + residual | N x 64 x 8 x 12 |
| Average over time and joints | N x 64 |
| Dropout + linear classifier | N x 3 logits |

Dropout=0.3. Random initialization, no pretrained weights. Softmax is only for
reporting probabilities; cross entropy receives logits directly. Temporal
convolution is not causal within the window; the entire window is already
observed at the prediction endpoint, and no post-endpoint frame is used.

## Proposed training and evaluation (requires confirmation)

Use seed 42. Split unique kicks with class stratification: first reserve 20%
for test, then 25% of remaining kicks for validation (about 60/20/20 overall).
With 41 retained kicks, this gives 24 train / 8 validation / 9 test kicks.
Map every sequence of a kick to that kick's split; never split views across
sets. Only training data update normalization statistics or determine class
weights. At evaluation, average probabilities of retained sequences per kick
for the PRIMARY kick-level metrics, and also report single-sequence metrics.
Kick-averaged results can use multiple views and must not be portrayed as
single-camera performance. The split cannot ensure player/match independence
because those identifiers are absent.

AdamW, lr=1e-3, weight_decay=1e-4, batch=8, at most 150 epochs. Cross entropy
uses inverse TRAIN kick-class frequency weights. No augmentation initially.
Clip gradient norm to 1.0. Select best checkpoint on validation kick macro-F1,
breaking ties by lower validation loss; stop after 25 epochs without a strict
macro-F1 improvement. Evaluate test once after selection, not every epoch.

Report macro-F1 (all three labels), accuracy, per-class recall, confusion matrix,
class counts, and a training-majority-class baseline. Save exact split IDs,
preprocessing manifest, CSV SHA-256, configuration, seed, metrics, and checkpoint
under runs/baseline_4ch. One available GPU is sufficient; choose it only after
checking occupancy at execution time. Do not launch multiple GPU workers.

With only 41 retained kicks and 7 Center examples, results are exploratory and
single-split metrics will be unstable. Generalization claims require more data
and/or further grouped evaluation, separately agreed after this first run.

## Approval scope

Confirm the 4-channel compact architecture, stride=1 sampling adjustment,
limited gap filling, and single grouped baseline run. Only after confirmation
implement and verify the data loader/training runner, then launch training.
