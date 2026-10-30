# 8-joint / 20-frame L/R baseline

## Input and task

- Source: `Main Data/Combined_penalty_pose_keypoints_normalized_20frames_8joints_angles_scaled.csv`.
- Exclude 14 whole C-direction kicks; retain 212 kicks / 4,240 rows.
- Predict canonicalized `kick_direction` only: L=0, R=1. No success/failure task.
- Each sample is `(20 frames, 20 features)`: 16 normalized xy coordinates for
  joints 5, 6, 11–16, then four normalized knee/hip angle columns.
- Raw coordinates, goal, source dataset, camera, IDs, source frame numbers,
  and all other metadata are NOT model inputs. Metadata remains in split CSVs
  for traceability. No extra interpolation, flipping, or frame selection.

## Split

Seed 42; target Train/Validation/Test = 70/15/15. All 20 frames of a
`(dataset, kick_id)` stay together, and the three kick sets are disjoint.
Integer quotas minimize dataset-percentage discrepancies while preserving
both dataset totals and split sizes. Within the source quotas, L/R allocations
are stratified, with both labels present for every dataset in every split.
Validation and test have identical dataset and overall direction counts.

| Split | Kicks | CLT | EPL | Women | L / R |
|---|---:|---:|---:|---:|---:|
| Train | 148 | 15 (10.14%) | 44 (29.73%) | 89 (60.14%) | 76 / 72 |
| Validation | 32 | 3 (9.38%) | 10 (31.25%) | 19 (59.38%) | 17 / 15 |
| Test | 32 | 3 (9.38%) | 10 (31.25%) | 19 (59.38%) | 17 / 15 |

Artifacts: `data/main_data_8j20_angles_lr/` contains the three CSVs,
`samples.npz`, `split_manifest.csv`, `summary.json`, and excluded-C list.
Split CSVs retain original column values and order within each kick.

This is a kick-level split, NOT a player-level or match-level split; the same
player/match may occur in different partitions. Dataset proportions alone do
not establish generalization to new players, matches, or cameras.

## Model and training

The coordinate branch reuses `model.py` ST-GCN blocks with a graph for the
eight selected joints, input `(batch, 2, 20, 8)`, widths 32/32/64. A separate
temporal convolution branch processes the four angle sequences. Global pooled
features are concatenated (64+16 dimensions) and classified into L/R.
Total trainable parameters: 44,778. No pretrained weights.

Batch size 16, AdamW, initial LR 0.001, weight decay 0.001, dropout 0.3,
gradient clipping 1.0. Cross-entropy class weights are computed from training
kicks only. Batch normalization running statistics are updated on train only.
Learning-rate scheduling uses validation loss. Checkpoint selection uses
validation macro F1, with validation loss as tie-breaker. Maximum 150 epochs;
early stopping after 30 epochs without checkpoint improvement.

The test set is evaluated once, after checkpoint selection. Source CSV hash
is checked before/after training; it is never overwritten.

## First run results

Run directory: `runs/main_data_8j20_angles_lr_stgcn_seed42/`.
Stopped at epoch 42; selected checkpoint from epoch 12.

| Split | Accuracy | Macro F1 |
|---|---:|---:|
| Train (selected checkpoint) | 80.41% | 0.8036 |
| Validation | 62.50% | 0.6235 |
| Test | 53.125% (17/32) | 0.5308 |

The training-majority class is L. Always predicting L also achieves 17/32
test accuracy, so this exploratory model has NOT demonstrated an accuracy
improvement over that baseline. Only one seed was evaluated, and the test
contains just 32 kicks (CLT: 3). Do not interpret per-dataset percentages as
stable estimates. Further model selection should use validation or train-only
cross-validation, not tune against this already-inspected test set.

## Commands

```bash
python -m unittest -v test_train_8j20_angles_lr.py
python train_8j20_angles_lr.py --dry-run
python train_8j20_angles_lr.py --config train_8j20_angles_lr_config.json
```

Existing data/run directories are never overwritten. For a repeat experiment,
copy the config to a new file and set distinct `data_dir` and `output_dir`.
The run includes feature order, label mapping, split membership, source hash,
code snapshots, epoch history, `best.pt`, `metrics.json`, and test predictions.
