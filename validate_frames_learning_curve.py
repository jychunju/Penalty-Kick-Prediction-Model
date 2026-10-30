#!/usr/bin/env python3
"""Verify frame/data-size fits, baseline reuse and protected-Test integrity."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from experiment_frames_learning_curve import BASELINE, VariableLengthSTGCN, frame_indices
from train_8j20_angles_lr import ROOT, hash_file, save_json
from validate_8j20_lr_cv import check_predictions


def validate(out):
    info = json.loads((out/"run_info.json").read_text())
    comparison = json.loads((out/"comparison.json").read_text())
    tasks = json.loads((out/"tasks.json").read_text())
    folds = json.loads((out/"folds.json").read_text())
    subsets = json.loads((out/"training_subsets.json").read_text())
    reference = ROOT/info["config"]["reference_run"]
    for path, digest in info["protected_file_hashes"].items():
        assert hash_file(Path(path)) == digest, path
    for name, digest in info["source_code_hashes"].items():
        assert hash_file(ROOT/name) == digest, name
    with np.load(out/"development_only.npz") as data:
        x, y = data["X"].copy(), data["y"].copy()
        groups = data["group_id"].tolist()
    with np.load(reference/"development_only.npz") as data:
        np.testing.assert_array_equal(x, data["X"][:, :, :16])
        np.testing.assert_array_equal(y, data["y"])
        assert groups == data["group_id"].tolist()
    assert x.shape == (180, 20, 16) and np.isfinite(x).all()
    assert not (set(groups) & set(info["protected_test_groups"]))
    for manifest in subsets:
        selections = manifest["subsets"]
        assert [len(selections[str(f)]) for f in [.25, .5, .75, 1.]] == [31, 61, 92, 122]
        assert set(selections['0.25']) < set(selections['0.5']) < set(selections['0.75']) < set(selections['1.0'])
        fold = next(f for f in folds if f["fold"] == manifest["fold"])
        assert selections["1.0"] == fold["indices"]["inner_train"]
    assert len(tasks) == 75 and len({task["task_key"] for task in tasks}) == 75
    metrics_by_condition = {}
    checked = 0
    for condition, result in comparison["results"].items():
        runs = []
        for run in result["runs"]:
            seed, fold_id = run["seed"], run["fold"]
            fold = next(f for f in folds if f["fold"] == fold_id)
            directory = Path(run["artifact_directory"])
            history = json.loads((directory/"history.json").read_text())
            stored = json.loads((directory/"metrics.json").read_text())
            best = max(history, key=lambda row: (row["early_stop"]["macro_f1"], -row["early_stop"]["loss"]))
            assert best["epoch"] == stored["best_epoch"] == run["best_epoch"]
            assert best["early_stop"] == run["early_stop"]
            assert all("validation" not in row and "test" not in row for row in history)
            assert run["test_evaluations"] == 0 and run["outer_validation_evaluations"] == 1
            with (directory/"validation_predictions.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            recalculated = check_predictions(rows, fold["group_ids"]["validation"])
            for key, value in recalculated.items():
                assert value == run["validation"][key]
            assert all(row["actual"] == ("L", "R")[y[groups.index(row["group_id"])]] for row in rows)
            checkpoint = torch.load(directory/"best.pt", map_location="cpu")
            assert checkpoint["epoch"] == run["best_epoch"]
            model = VariableLengthSTGCN()
            model.load_state_dict(checkpoint["model"], strict=True)
            assert sum(p.numel() for p in model.parameters()) == run["parameters"] == 11074
            if condition == BASELINE:
                assert directory == reference/"small_stgcn_xy"/f"seed_{seed}"/f"fold_{fold_id}"
                assert run["reused_reference"] and run["training_kicks"] == 122
            else:
                task = next(t for t in tasks if t["condition"] == condition and t["seed"] == seed and t["fold"] == fold_id)
                assert task["frame_indices"] == frame_indices(condition) == checkpoint["frame_indices"]
                assert run["frames"] == len(task["frame_indices"])
                assert task["indices"]["inner_train"] == checkpoint["training_indices"]
                assert len(checkpoint["training_indices"]) == run["training_kicks"]
                for phase in ("early_stop", "validation"):
                    assert task["indices"][phase] == fold["indices"][phase]
                for a, b in itertools.combinations(task["indices"], 2):
                    assert not (set(task["indices"][a]) & set(task["indices"][b]))
                assert set(task["indices"]["inner_train"]) <= set(fold["indices"]["inner_train"])
                checked += 1
            runs.append(stored)
        assert len(runs) == 15
        metrics_by_condition[condition] = runs
        for metric in ("accuracy", "macro_f1"):
            values = [run["validation"][metric] for run in runs]
            assert abs(np.mean(values)-result["cv"][metric]["mean"]) < 1e-12
            assert abs(np.std(values, ddof=1)-result["cv"][metric]["std"]) < 1e-12
    assert checked == comparison["new_fits"] == 75
    assert comparison["total_scored_fits"] == 90 and comparison["test_evaluations"] == 0
    assert (out/"learning_curve.png").stat().st_size > 0
    result = {"status": "PASS", "new_fits_checked": 75, "reference_fits_checked": 15,
              "checkpoints_loaded": 90, "validation_predictions_checked": 90*36,
              "nested_training_subsets": True, "fixed_validation_sets": True,
              "unchanged_parameter_count": 11074, "source_features_unchanged": True,
              "protected_files_unchanged": True, "test_evaluations": 0}
    save_json(out/"artifact_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT/"runs/main_data_8j_lr_frames_learning_curve")
    args = parser.parse_args()
    validate(args.directory)
