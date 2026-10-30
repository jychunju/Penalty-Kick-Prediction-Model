#!/usr/bin/env python3
"""Independently reconcile CV artifacts; never evaluate a protected Test kick."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from compare_8j20_lr_cv import SmallPoseClassifier
from train_8j20_angles_lr import ROOT, hash_file, load_samples, save_json


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def check_predictions(rows, expected_groups):
    assert len(rows) == len(expected_groups)
    assert {row["group_id"] for row in rows} == set(expected_groups)
    truth = np.array([0 if row["actual"] == "L" else 1 for row in rows])
    probabilities = np.array([[float(row["prob_L"]), float(row["prob_R"])] for row in rows])
    predicted = probabilities.argmax(axis=1)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert all(row["predicted"] == ("L", "R")[p] for row, p in zip(rows, predicted))
    return {"accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "confusion_matrix_L_R": confusion_matrix(truth, predicted, labels=[0, 1]).tolist()}


def validate(directory):
    info = json.loads((directory/"run_info.json").read_text())
    comparison = json.loads((directory/"comparison.json").read_text())
    folds = json.loads((directory/"folds.json").read_text())
    config = info["config"]
    for path, expected in info["protected_file_hashes"].items():
        assert hash_file(Path(path)) == expected, path
    for name, expected in info["source_code_hashes"].items():
        assert hash_file(ROOT/name) == expected, name
    protected = set(info["protected_test_groups"])
    with np.load(directory/"development_only.npz") as data:
        x, y = data["X"].copy(), data["y"].copy()
        groups = data["group_id"].tolist()
    assert x.shape == (180, 20, 20) and np.isfinite(x).all()
    assert len(set(groups)) == 180 and not(set(groups) & protected)
    originals = []
    for name in ("train", "validation"):
        _, samples, excluded = load_samples(ROOT/config["data_dir"]/f"{name}.csv")
        assert not excluded
        originals.extend(samples)
    originals = {sample["group_id"]: sample for sample in originals}
    assert set(originals) == set(groups)
    for i, group in enumerate(groups):
        np.testing.assert_array_equal(x[i], originals[group]["x"])
        assert y[i] == (0 if originals[group]["label"] == "L" else 1)
    covered = []
    for fold in folds:
        ids = fold["group_ids"]
        for a, b in itertools.combinations(ids, 2):
            assert not(set(ids[a]) & set(ids[b]))
        assert set.union(*(set(v) for v in ids.values())) == set(groups)
        for name, indices in fold["indices"].items():
            assert ids[name] == [groups[i] for i in indices]
            assert not(set(ids[name]) & protected)
        covered.extend(ids["validation"])
    assert len(covered) == 180 and len(set(covered)) == 180
    total = 0
    for variant in config["variants"]:
        variant_dir = directory/variant
        summary = json.loads((variant_dir/"summary.json").read_text())
        runs = []
        for seed in config["training_seeds"]:
            out_of_fold = read_rows(variant_dir/f"oof_seed_{seed}.csv")
            oof_metrics = check_predictions(out_of_fold, groups)
            for key, value in oof_metrics.items():
                assert value == summary["per_seed_oof"][str(seed)][key]
            oof_by_group = {row["group_id"]: row for row in out_of_fold}
            for fold in folds:
                fit = variant_dir/f"seed_{seed}"/f"fold_{fold['fold']}"
                metrics = json.loads((fit/"metrics.json").read_text())
                history = json.loads((fit/"history.json").read_text())
                best = max(history, key=lambda row: (row["early_stop"]["macro_f1"], -row["early_stop"]["loss"]))
                assert best["epoch"] == metrics["best_epoch"]
                assert best["early_stop"] == metrics["early_stop"]
                assert all("validation" not in row and "test" not in row for row in history)
                assert metrics["outer_validation_evaluations"] == 1 and metrics["test_evaluations"] == 0
                assert metrics["epochs_completed"] == len(history) <= config["epochs"]
                rows = read_rows(fit/"validation_predictions.csv")
                calculated = check_predictions(rows, fold["group_ids"]["validation"])
                for key, value in calculated.items():
                    assert value == metrics["validation"][key]
                for row in rows:
                    assert row["actual"] == originals[row["group_id"]]["label"]
                    assert row["dataset"] == originals[row["group_id"]]["dataset"]
                    assert not row["group_id"] in protected
                    assert all(abs(float(row[col])-float(oof_by_group[row["group_id"]][col])) < 1e-7
                               for col in ("prob_L", "prob_R"))
                checkpoint = torch.load(fit/"best.pt", map_location="cpu")
                assert checkpoint["epoch"] == metrics["best_epoch"]
                assert len(checkpoint["feature_columns"]) == (20 if variant.endswith("_angles") else 16)
                model = SmallPoseClassifier(variant, config["dropout"])
                model.load_state_dict(checkpoint["model"], strict=True)
                assert sum(p.numel() for p in model.parameters()) == metrics["parameters"]
                runs.append(metrics)
                total += 1
        assert len(runs) == 15
        report_row = next(row for row in comparison["results"] if row["variant"] == variant)
        for metric in ("accuracy", "macro_f1"):
            values = [run["validation"][metric] for run in runs]
            assert abs(np.mean(values)-report_row["cv"][metric]["mean"]) < 1e-12
            assert abs(np.std(values, ddof=1)-report_row["cv"][metric]["std"]) < 1e-12
    assert total == comparison["total_fits"] == 60
    assert comparison["test_evaluations"] == 0
    result = {"status": "PASS", "fits_checked": total, "checkpoints_loaded": total,
              "development_kicks": 180, "protected_test_kicks": len(protected),
              "validation_predictions_checked": total*36, "oof_predictions_checked": 4*3*180,
              "source_features_unchanged": True, "protected_files_unchanged": True,
              "inner_only_checkpoint_selection": True, "test_evaluations": 0}
    save_json(directory/"artifact_validation.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT/"runs/main_data_8j20_lr_cv_comparison")
    args = parser.parse_args()
    validate(args.directory)
