# Frame × joint comparison

Run `python experiment_frames_joints.py`, then `python validate_frames_joints.py`.
Unit tests: `python -m unittest test_experiment_frames_joints`.
Use a new `output_dir` in a copied config for reruns; outputs are never overwritten.

The experiment continues the C-camera-only L/R task, with the same 141 development
kicks and unchanged protected Test. CLT is treated as camera C by user confirmation.
The original dataset × label 5-fold splits, inner stopping sets and seeds 42/43/44
are reused exactly. Each fold has 95–96 training kicks, 17 stopping kicks and
28–29 outer-validation kicks. No Test features enter any model.

## Inputs

- 8 joints: COCO 5,6,11,12,13,14,15,16 (shoulders, hips, knees, ankles).
- 6 joints: COCO 11,12,13,14,15,16 (hips, knees, ankles).
- Only normalized xy coordinates are used; angles are excluded in all conditions.
- Pelvis centering and clip-median torso normalization stay exactly as exported.
  Removing shoulder inputs does not recompute the shoulder-derived torso scale.
- Frames are subsets of the existing ordered 20 samples, not newly extracted raw
  frames or fixed physical durations. First and contact endpoints are retained:
  20 = all; 10 = [0,2,4,6,8,11,13,15,17,19]; 5 = [0,5,10,14,19]; 3 = [0,10,19].

## Models and training

Small ST-GCN and small 1D-CNN each receive all eight T×J combinations. Hidden
widths and temporal kernels remain fixed. Six-joint ST-GCN uses the appropriately
remapped hip/knee/ankle graph. Parameter counts are ST-GCN 11074/11066 and CNN
4034/3706 for 8/6 joints; frame count does not change parameter count.

AdamW lr=.001, weight decay=.001, dropout=.3, batch=16, max epochs=150,
early-stop patience=25, gradient clipping=1 and train-only class weights match
the reference experiment. Checkpoints use inner-stop macro F1, then loss.
Outer validation never selects epochs. Short sequences increase temporal padding
and alter the effective receptive field, so this is a fixed-architecture ablation.

There are 240 scored fits: 60 equivalent 8-joint 20/10-frame reference fits are
reused after verifying data/folds/code and exact model initialization/forward
equivalence; 180 fits are newly trained. All source/prior artifacts are preserved.

## Results and limits

`REPORT.md`, `comparison.csv` and `train_validation_curves.png` show train accuracy,
outer-validation accuracy/SD, macro F1 and train−validation gap. Paired frame/joint
comparisons in `comparison.json` use identical seed/fold pairs. Validation checks
all checkpoint predictions and training scores, including reference fits.

Fold/seed SD is descriptive, not a confidence interval. A bigger generalization
gap is suggestive, not definitive evidence of overfitting; information loss,
optimization and receptive-field effects also matter. This remains kick-level
validation, not player/match-level generalization. No final all-development model
or independent Test evaluation is performed.
