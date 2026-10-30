# Consecutive precontact ST-GCN comparison

Run `python experiment_precontact.py`, then `python validate_precontact.py`.
Tests: `python -m unittest test_experiment_precontact test_experiment_frames_joints`.
Existing outputs are never overwritten; choose a new output directory for reruns.

There are exactly four conditions: original consecutive 5/3 frames × 6/8 joints.
Default `include_contact=false` selects `[t-5,...,t-1]` and `[t-3,...,t-1]`, where
`t` is the original `last_touch=1` frame. These are NOT the final 5/3 of the existing
20 uniform samples. `include_contact=true` explicitly selects `[t-4,...,t]` and
`[t-2,...,t]` instead. Do not interpret raw frame counts as identical durations;
FPS differs or is absent across videos.

## Data integrity

All 141 original C-camera development kicks have complete consecutive 5-frame
windows. CLT is treated as camera C by user confirmation. Exact camera/tracker
sequence correspondence is recovered from the prior sampler's CSV-line audit;
its 20 sampled coordinates must reproduce the existing development features.
No alternate camera/track selection, missing-frame substitution, interpolation,
horizontal flip, label modification or normalization recomputation occurs.

Model inputs contain normalized xy only. Eight joints are COCO 5,6,11,12,13,14,
15,16; six joints remove shoulders, preserving the already-computed torso scale.
The full source CSV is unchanged. Exported window CSVs preserve all source columns
and append sequence/window audit fields, including effective camera direction.
With contact excluded, all exported `last_touch` values are zero by design.

## Training and evaluation

Original 5 source×label folds, inner stopping sets, and seeds 42/43/44 are fixed:
95–96 train, 17 stopping, 28–29 outer-validation kicks per model. Four conditions
× five folds × three seeds = 60 new fits, with the same small ST-GCN and AdamW
settings as before (lr .001, wd .001, dropout .3, batch 16, 150 max epochs,
patience 25, train-class loss weighting and gradient clipping 1).

Checkpoints use inner-stop macro F1, then loss. Outer validation is never used
for epoch selection. Output includes training accuracy, validation accuracy/SD,
macro F1, generalization gaps and paired differences from earlier uniform windows.
Differences reflect temporal coverage and endpoint availability, not frame count
alone. Fold/seed SD is descriptive, not a confidence interval.

The old Test has already been exposed by the previous experiment. No Test
reevaluation occurs here, and prior Test artifacts are hash-protected. This is
exploratory kick-level CV, not a fresh independent generalization confirmation.
