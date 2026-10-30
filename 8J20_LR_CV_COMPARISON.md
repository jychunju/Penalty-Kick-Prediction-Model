# Paired small-model / angle-feature comparison

## Scope

Input is the unchanged 8-joint, 20-frame, normalized L/R dataset. Only the
existing `data/main_data_8j20_angles_lr/train.csv` and `validation.csv` feature
files are loaded. The original Test 32 kicks are excluded from the experiment.
Goal, camera, dataset ID and other metadata are never model inputs. No extra
normalization, interpolation, smoothing, flipping, frame selection, or outlier
removal is performed.

## Four predefined candidates

- `small_stgcn_xy`: ST-GCN widths 16/16/32, normalized xy only (16 features).
- `small_stgcn_xy_angles`: same coordinate branch + four-angle temporal branch
  (20 features). Its pooled angle branch has width 8.
- `small_cnn_xy`: temporal 1D-CNN widths 16/32, normalized xy only.
- `small_cnn_xy_angles`: same temporal coordinate branch + the same angle
  branch used in the small ST-GCN.

Both architectures pool their temporal features before a linear L/R head.
The two architectures do not have identical parameter counts; counts are
reported. Comparing angles within one architecture is a feature-plus-capacity
ablation, not proof that any observed difference is solely due to angle values.
No velocity features or pretrained weights are used.

## Evaluation design

1. Keep the original Test membership and existing artifacts fixed.
2. Combine the original Train+Valid: 180 kicks, 93 L / 87 R; CLT 18, EPL 54,
   Women 108. Group by `(dataset, kick_id)`, so a kick's frames cannot be split.
3. Make five identical outer folds across all candidates, stratified by
   `dataset × L/R`. Each outer validation fold contains 36 kicks; the remaining
   144 are split into inner training 122 and early-stop validation 22.
4. Checkpoints, learning-rate scheduling and early stopping use only the
   inner 22 kicks. The outer 36 kicks are evaluated once after checkpoint
   selection. They are not inspected during epochs.
5. Repeat with training seeds 42/43/44 on the same folds and inner splits:
   4 candidates × 5 folds × 3 seeds = 60 fits.
6. Rank candidates by mean outer-fold macro F1. Report accuracy and macro F1
   with descriptive SD over 15 fold/seed scores. These scores are correlated,
   so that SD is not a confidence interval or a significance test.

All models use the same fixed hyperparameters: AdamW LR 0.001, weight decay
0.001, dropout 0.3, batch size 16, gradient clipping 1.0, maximum 150 epochs,
25 epochs of early-stopping patience. Class weights are computed from each
inner training fold only. Batch-normalization statistics are updated only on
inner training batches. The learning-rate scheduler uses inner validation loss.

No hyperparameter search is included. The already-tested original larger
ST-GCN is not rerun under this protocol, so the comparison cannot establish
that a candidate beats it. CV results must not be directly compared to its
single-split Test accuracy. A CV winner is an exploratory model-selection
candidate, not a confirmed independent-Test improvement.

Kick-level grouping does not ensure player-level or match-level separation.

## Artifacts and verification

Output: `runs/main_data_8j20_lr_cv_comparison/`.

- `run_info.json`, `folds.json`, `development_only.npz`: exact feature order,
  memberships, dataset/label counts, config, protected Test IDs and file hashes.
- Each variant: 15 checkpoints, histories, metrics and validation predictions;
  three out-of-fold prediction CSVs and per-seed metrics over 180 kicks.
- `comparison.json`, `comparison.csv`, `REPORT.md`: cross-model summary and
  CV-selected candidate. No final model is fitted on all 180 kicks and no
  new Test evaluation is performed.
- `coordinate_jump_inspection_flags.json`: development-only ankle sample
  displacements larger than two normalized torso units. Flags are inspection
  candidates, NOT confirmed detection errors. No cleaning is applied.
- Original source CSV, Train/Valid/Test CSVs, original NPZ and baseline
  checkpoint/metrics/split files are hash-checked before/after the experiment.

## Commands

```bash
python -m unittest -v test_compare_8j20_lr_cv.py
python -u compare_8j20_lr_cv.py --config compare_8j20_lr_cv_config.json
```

The default config runs one variant per GPU on GPUs 0–3. Existing output
directories are never overwritten. Change `output_dir` in a new config for
another experiment. The protected Test CSV is hashed for integrity but its
features are not parsed or supplied to a model by this experiment.
