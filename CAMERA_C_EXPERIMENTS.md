# C-camera-only L/R experiments

Run `python experiment_camera_c.py`, then `python validate_camera_c.py`.
The default output is `runs/main_data_camera_c_lr_comparison/`.
Existing output directories are never overwritten; use a copied config with a
new `output_dir` for another run (`--config /absolute/path/config.json`).

## Data and isolation

- CLT camera is explicitly treated as C, as confirmed by the user. Do not infer
  camera direction from CLT filenames. Only a new experiment CSV is corrected;
  source metadata, labels and coordinates remain unchanged.
- EPL and Women require their existing `camera_direction` to be C.
- The prediction target remains `kick_direction` L/R. Direction C is excluded.
- The source contains 167 eligible kicks; 26 are in the existing protected Test.
  Experiments use only the remaining 141 existing Train/Validation kicks:
  CLT 18 (9 L/9 R), EPL 21 (12 L/9 R), Women 102 (57 L/45 R).
- Each kick contributes one ordered 20-frame sequence. No raw frames are
  re-extracted, re-interpolated, renormalized or reflected.
- Outer 5-fold splits are stratified by dataset × L/R, seed 42. Every condition
  shares these splits and the same inner stopping sets (17 kicks). Inner training
  has 95–96 kicks; outer validation has 28–29 kicks.
- Checkpoints are selected by inner-stop macro F1, then loss. Outer validation
  never controls stopping. Existing Test is neither trained on nor evaluated.
- Train RNG seeds 42/43/44 are repeated on the same folds: 105 fits total.

## Predeclared settings

| Condition | Model | Frames | Features | Dropout / weight decay |
|---|---|---|---|---|
| stgcn20_xy | Small ST-GCN | 20 | 16 xy | 0.3 / 0.001 |
| stgcn10_xy | Small ST-GCN | Uniform 10 | 16 xy | 0.3 / 0.001 |
| stgcn_last10_xy | Small ST-GCN | Last 10 | 16 xy | 0.3 / 0.001 |
| stgcn20_xy_angles | Small ST-GCN | 20 | 16 xy + 4 angles | 0.3 / 0.001 |
| cnn20_xy | Small 1D-CNN | 20 | 16 xy | 0.3 / 0.001 |
| cnn10_xy | Small 1D-CNN | Uniform 10 | 16 xy | 0.3 / 0.001 |
| stgcn10_xy_regularized | Small ST-GCN | Uniform 10 | 16 xy | 0.5 / 0.01 |

All settings use AdamW, learning rate 0.001, batch size 16, maximum 150 epochs,
early-stopping patience 25, gradient clipping 1.0, and inverse training-class
frequency loss weights. Goal and metadata are not model inputs.
Uniform10 uses indices `[0,2,4,6,8,11,13,15,17,19]`; Last10 uses `[10,...,19]`.
These are sampled-index intervals, not fixed physical durations.

## Validation and interpretation

Unit tests: `python -m unittest test_experiment_camera_c`.
Artifact validation checks unchanged source/prior-artifact hashes, corrected
metadata, exact feature/label preservation, disjoint splits, inner-only epoch
selection, checkpoint loading and reproduction of outer-validation predictions.

`REPORT.md` and `comparison.csv` report accuracy/macro F1, descriptive fold/seed
SD, training accuracy, majority baseline and dataset-specific OOF performance.
Folds and seeds are correlated; SD is not a confidence interval. The C-camera
subset has different counts, source ratios and folds from prior experiments:
cross-experiment differences do not isolate the causal effect of filtering.
These are kick-level splits, not player/match-level generalization estimates.
