#!/usr/bin/env python3
"""Paired 5-fold x 3-seed model/feature comparison; existing test untouched.

Only the previously exported train and validation feature CSVs are loaded.
For each outer fold, an INNER split selects epochs and schedules learning rates.
Outer validation is evaluated once per fitted model, never used to stop epochs.
Four independent ordinary training processes can run on four available GPUs.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import STGCNBlock, make_adjacency
from train_8j20_angles_lr import (
    EDGES, FEATURES, LABELS, ROOT, evaluate, hash_file, load_samples,
    save_json, score, split_summary,
)

VARIANTS = ("small_stgcn_xy", "small_stgcn_xy_angles", "small_cnn_xy", "small_cnn_xy_angles")


class SmallPoseClassifier(nn.Module):
    """A small coordinate branch, optionally fused with four angle sequences."""
    def __init__(self, variant, dropout=0.3):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.use_angles = variant.endswith("_angles")
        self.feature_count = 20 if self.use_angles else 16
        self.graph = "stgcn" in variant
        self.coordinate_norm = nn.BatchNorm1d(16)
        if self.graph:
            self.register_buffer("adjacency", make_adjacency(8, EDGES, roots=(2, 3)))
            self.blocks = nn.ModuleList([STGCNBlock(2, 16, dropout=dropout),
                                        STGCNBlock(16, 16, dropout=dropout),
                                        STGCNBlock(16, 32, stride=2, dropout=dropout)])
        else:
            self.coordinate_cnn = nn.Sequential(
                nn.Conv1d(16, 16, 5, padding=2, bias=False), nn.BatchNorm1d(16),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(16, 32, 5, padding=2, bias=False), nn.BatchNorm1d(32),
                nn.ReLU(), nn.Dropout(dropout), nn.AdaptiveAvgPool1d(1))
        if self.use_angles:
            self.angle_branch = nn.Sequential(
                nn.BatchNorm1d(4), nn.Conv1d(4, 8, 5, padding=2), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(8, 8, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(40 if self.use_angles else 32, 2))

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (20, self.feature_count):
            raise ValueError(f"Expected (batch, 20, {self.feature_count})")
        coordinates = self.coordinate_norm(x[:, :, :16].transpose(1, 2))
        if self.graph:
            coordinates = coordinates.reshape(x.shape[0], 8, 2, 20).permute(0, 2, 3, 1)
            for block in self.blocks:
                coordinates = block(coordinates, self.adjacency)
            pooled = coordinates.mean(dim=(2, 3))
        else:
            pooled = self.coordinate_cnn(coordinates).squeeze(-1)
        if self.use_angles:
            angles = self.angle_branch(x[:, :, 16:].transpose(1, 2)).squeeze(-1)
            pooled = torch.cat((pooled, angles), dim=1)
        return self.head(pooled)


def make_folds(samples, protected_test_groups, n_splits=5, split_seed=42, stop_fraction=0.15):
    """Same source x direction folds and inner splits for every model and seed."""
    group_ids = [sample["group_id"] for sample in samples]
    if len(set(group_ids)) != len(samples) or set(group_ids) & set(protected_test_groups):
        raise ValueError("Duplicate kicks or protected test kicks in development data")
    strata = np.array([f"{s['dataset']}|{s['label']}" for s in samples])
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    folds = []
    for fold, (training, validation) in enumerate(splitter.split(np.zeros(len(samples)), strata), 1):
        inner_train, early_stop = train_test_split(
            training, test_size=stop_fraction, random_state=split_seed+100+fold, stratify=strata[training])
        indices = {"inner_train": sorted(inner_train.tolist()), "early_stop": sorted(early_stop.tolist()),
                   "validation": sorted(validation.tolist())}
        ids = {name: [group_ids[i] for i in values] for name, values in indices.items()}
        for a, b in itertools.combinations(ids, 2):
            assert not (set(ids[a]) & set(ids[b]))
        assert set.union(*(set(v) for v in ids.values())) == set(group_ids)
        folds.append({"fold": fold, "indices": indices, "group_ids": ids,
                      "distributions": {name: split_summary(samples, values) for name, values in indices.items()}})
    validation_occurrences = [i for fold in folds for i in fold["indices"]["validation"]]
    assert sorted(validation_occurrences) == list(range(len(samples)))
    return folds


def summarize_runs(runs):
    """Descriptive variability, not confidence intervals (folds share training data)."""
    def moments(values):
        return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)), "max": float(np.max(values))}
    return {"num_fits": len(runs),
            "accuracy": moments([run["validation"]["accuracy"] for run in runs]),
            "macro_f1": moments([run["validation"]["macro_f1"] for run in runs]),
            "training_accuracy": moments([run["train"]["accuracy"] for run in runs]),
            "best_epoch": moments([run["best_epoch"] for run in runs])}


def write_predictions(path, samples, indices, probabilities, seed, fold):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "kick_id", "group_id", "actual", "predicted", "prob_L", "prob_R", "seed", "fold"])
        writer.writeheader()
        for i, probability in zip(indices, probabilities):
            writer.writerow({"dataset": samples[i]["dataset"], "kick_id": samples[i]["kick_id"],
                             "group_id": samples[i]["group_id"], "actual": samples[i]["label"],
                             "predicted": LABELS[int(probability.argmax())], "prob_L": float(probability[0]),
                             "prob_R": float(probability[1]), "seed": seed, "fold": fold})


def fit_one(config, variant, x, y, samples, fold, seed, device, directory):
    fit_seed = seed+1000*fold["fold"]
    random.seed(fit_seed); np.random.seed(fit_seed); torch.manual_seed(fit_seed); torch.cuda.manual_seed_all(fit_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model = SmallPoseClassifier(variant, config["dropout"]).to(device)
    train, stopping, outer = [fold["indices"][name] for name in ("inner_train", "early_stop", "validation")]
    counts = torch.bincount(y[train], minlength=2).float()
    weights = counts.sum()/(2*counts)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    loader = DataLoader(TensorDataset(x[train], y[train]), batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(fit_seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5)
    directory.mkdir(parents=True, exist_ok=False)
    best, best_epoch, history = (-1., -float("inf")), 0, []
    start = time.perf_counter()
    for epoch in range(1, config["epochs"]+1):
        model.train()
        total_loss, count = 0., 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            total_loss += float(loss.detach())*len(yb)
            count += len(yb)
        # The outer fold is deliberately NOT passed to this evaluation.
        stopping_metrics, _ = evaluate(model, x, y, stopping, samples, device)
        candidate = (stopping_metrics["macro_f1"], -stopping_metrics["loss"])
        history.append({"epoch": epoch, "train_loss": total_loss/count, "early_stop": stopping_metrics,
                        "lr": optimizer.param_groups[0]["lr"]})
        if candidate > best:
            best, best_epoch = candidate, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "early_stop": stopping_metrics,
                        "variant": variant, "feature_columns": list(FEATURES[:x.shape[-1]]), "config": config,
                        "seed": seed, "fit_seed": fit_seed, "fold": fold["fold"], "class_weights": weights.tolist()}, directory/"best.pt")
        scheduler.step(stopping_metrics["loss"])
        if epoch-best_epoch >= config["early_stopping_patience"]:
            break
    checkpoint = torch.load(directory/"best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    # Exactly one outer-fold evaluation, after inner-only epoch selection.
    validation, probabilities = evaluate(model, x, y, outer, samples, device)
    training, _ = evaluate(model, x, y, train, samples, device)
    majority_label = int(torch.argmax(counts))
    baseline_probs = np.zeros((len(outer), 2)); baseline_probs[:, majority_label] = 1
    baseline = score(y[outer].numpy(), baseline_probs, np.array([samples[i]["dataset"] for i in outer]))
    metrics = {"variant": variant, "seed": seed, "fit_seed": fit_seed, "fold": fold["fold"],
               "parameters": sum(p.numel() for p in model.parameters()), "best_epoch": best_epoch,
               "epochs_completed": len(history), "elapsed_seconds": time.perf_counter()-start,
               "train": training, "early_stop": checkpoint["early_stop"], "validation": validation,
               "majority_baseline": baseline, "outer_validation_evaluations": 1, "test_evaluations": 0}
    save_json(directory/"history.json", history)
    save_json(directory/"metrics.json", metrics)
    write_predictions(directory/"validation_predictions.csv", samples, outer, probabilities, seed, fold["fold"])
    return metrics, probabilities


def worker(config, variant):
    torch.set_num_threads(config["torch_threads_per_worker"])
    root = ROOT/config["output_dir"]
    info = json.loads((root/"run_info.json").read_text())
    folds = json.loads((root/"folds.json").read_text())
    with np.load(root/"development_only.npz") as data:
        features = 20 if variant.endswith("_angles") else 16
        x = torch.from_numpy(data["X"][:, :, :features].copy())
        y = torch.from_numpy(data["y"].copy())
        samples = [{"group_id": group, "dataset": source, "kick_id": kick, "label": LABELS[int(target)]}
                   for group, source, kick, target in zip(data["group_id"].tolist(), data["dataset"].tolist(), data["kick_id"].tolist(), data["y"].tolist())]
    device = torch.device(config["devices"][config["variants"].index(variant) % len(config["devices"])])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    directory = root/variant
    directory.mkdir(exist_ok=False)
    model = SmallPoseClassifier(variant, config["dropout"])
    (directory/"model_architecture.txt").write_text(str(model), encoding="utf-8")
    runs, per_seed_oof = [], {}
    started = time.perf_counter()
    for seed in config["training_seeds"]:
        oof = np.full((len(samples), 2), np.nan, dtype=np.float32)
        for fold in folds:
            metrics, probabilities = fit_one(config, variant, x, y, samples, fold, seed, device,
                                             directory/f"seed_{seed}"/f"fold_{fold['fold']}")
            runs.append(metrics)
            oof[fold["indices"]["validation"]] = probabilities
            save_json(directory/"progress.json", {"completed_fits": len(runs), "total_fits": len(folds)*len(config["training_seeds"]),
                                                 "last_fold": fold["fold"], "last_seed": seed})
            print(json.dumps({"variant": variant, "seed": seed, "fold": fold["fold"],
                              "accuracy": metrics["validation"]["accuracy"], "macro_f1": metrics["validation"]["macro_f1"],
                              "best_epoch": metrics["best_epoch"]}), flush=True)
        assert np.isfinite(oof).all()
        per_seed_oof[str(seed)] = score(y.numpy(), oof, np.array([s["dataset"] for s in samples]))
        write_predictions(directory/f"oof_seed_{seed}.csv", samples, list(range(len(samples))), oof, seed, "see folds.json")
    summary = {"variant": variant, "feature_count": features, "parameters": runs[0]["parameters"],
               "cv": summarize_runs(runs), "per_seed_oof": per_seed_oof,
               "runs": runs, "elapsed_seconds": time.perf_counter()-started, "test_evaluations": 0,
               "protected_test_groups_excluded": info["protected_test_kicks"]}
    save_json(directory/"summary.json", summary)


def prepare(config):
    out = ROOT/config["output_dir"]
    if out.exists():
        raise FileExistsError("Output exists; choose a new output_dir instead of overwriting")
    data = ROOT/config["data_dir"]
    baseline = ROOT/config["baseline_run"]
    split = json.loads((baseline/"split.json").read_text())
    baseline_info = json.loads((baseline/"run_info.json").read_text())
    protected_paths = [Path(baseline_info["input_csv"]), data/"train.csv", data/"validation.csv", data/"test.csv",
                       data/"samples.npz", baseline/"split.json", baseline/"run_info.json", baseline/"best.pt", baseline/"metrics.json"]
    hashes = {str(path): hash_file(path) for path in protected_paths}
    if hashes[str(Path(baseline_info["input_csv"]))] != baseline_info["input_sha256"]:
        raise ValueError("Source CSV no longer matches original split experiment")
    # No protected Test feature CSV is parsed or passed to a model.
    _, training, excluded_train = load_samples(data/"train.csv")
    _, validation, excluded_validation = load_samples(data/"validation.csv")
    if excluded_train or excluded_validation:
        raise ValueError("Unexpected C in exported L/R partitions")
    assert {s["group_id"] for s in training} == set(split["train"])
    assert {s["group_id"] for s in validation} == set(split["validation"])
    samples = sorted(training+validation, key=lambda s: s["group_id"])
    folds = make_folds(samples, split["test"], config["n_splits"], config["split_seed"], config["early_stop_fraction"])
    x = np.stack([s["x"] for s in samples])
    y = np.array([LABELS.index(s["label"]) for s in samples], dtype=np.int64)
    info = {"config": config, "development_shape": list(x.shape), "development": split_summary(samples, list(range(len(samples)))),
            "protected_test_kicks": len(split["test"]), "protected_test_groups": split["test"],
            "protected_file_hashes": hashes, "input_feature_columns": list(FEATURES),
            "test_features_loaded": False, "test_evaluations": 0, "preprocessing_changes": False,
            "fold_checkpoint_selection": "Inner early-stop macro F1 then loss; outer fold never used for checkpoint selection",
            "model_comparison_metric": "Mean outer-fold macro F1 over 5 folds x 3 training seeds",
            "source_code_hashes": {name: hash_file(ROOT/name) for name in ("compare_8j20_lr_cv.py", "train_8j20_angles_lr.py", "model.py")},
            "notes": "Kick-level, not player/match-level. Repeated folds/seeds are correlated; SD is descriptive, not a confidence interval."}
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/"run_info.json", info)
    save_json(out/"folds.json", folds)
    np.savez_compressed(out/"development_only.npz", X=x, y=y, group_id=np.array([s["group_id"] for s in samples]),
                        dataset=np.array([s["dataset"] for s in samples]), kick_id=np.array([s["kick_id"] for s in samples]))
    # Flag large selected-sample jumps for inspection, not automatic filtering.
    flags = []
    for sample in samples:
        for joint in (15, 16):
            index = (5, 6, 11, 12, 13, 14, 15, 16).index(joint)
            differences = np.linalg.norm(np.diff(sample["x"][:, index*2:index*2+2], axis=0), axis=1)
            for step in np.flatnonzero(differences > 2):
                flags.append({"group_id": sample["group_id"], "joint": joint,
                              "from_frame": sample["rows"][int(step)]["frame"], "to_frame": sample["rows"][int(step)+1]["frame"],
                              "distance_torso_units": float(differences[step])})
    save_json(out/"coordinate_jump_inspection_flags.json", {"threshold": 2, "flags": flags,
              "note": "Observed sample jumps are NOT proof of errors; no kick or coordinate was changed. Test not inspected."})
    return info


def aggregate(config):
    out = ROOT/config["output_dir"]
    info = json.loads((out/"run_info.json").read_text())
    variants = [json.loads((out/variant/"summary.json").read_text()) for variant in config["variants"]]
    for path, digest in info["protected_file_hashes"].items():
        if hash_file(Path(path)) != digest:
            raise RuntimeError(f"Protected file changed: {path}")
    ranked = sorted(variants, key=lambda row: (row["cv"]["macro_f1"]["mean"], row["cv"]["accuracy"]["mean"]), reverse=True)
    baseline_runs = [run["majority_baseline"] for run in variants[0]["runs"]]
    summary = {"development_kicks": info["development"]["kicks"], "protected_test_kicks": info["protected_test_kicks"],
               "n_splits": config["n_splits"], "training_seeds": config["training_seeds"],
               "total_fits": sum(row["cv"]["num_fits"] for row in variants), "test_evaluations": 0,
               "selected_by_cv_macro_f1": ranked[0]["variant"],
               "results": [{key: row[key] for key in ("variant", "parameters", "feature_count", "cv", "per_seed_oof")} for row in ranked],
               "majority_baseline": {"accuracy_mean": float(np.mean([r["accuracy"] for r in baseline_runs])),
                                     "macro_f1_mean": float(np.mean([r["macro_f1"] for r in baseline_runs]))},
               "protected_files_unchanged": True,
               "interpretation": "Exploratory model-selection CV, not independent Test accuracy. No hyperparameter sweep or outlier removal. SD is over 15 correlated fold/seed scores."}
    save_json(out/"comparison.json", summary)
    with (out/"comparison.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "parameters", "features", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"])
        writer.writeheader()
        for row in ranked:
            writer.writerow({"variant": row["variant"], "parameters": row["parameters"], "features": row["feature_count"],
                             **{f"{metric}_{moment}": row["cv"][metric][moment] for metric in ("accuracy", "macro_f1") for moment in ("mean", "std")}})
    lines = ["# 8-joint L/R paired comparison", "", "Only original Train+Valid (180 kicks); Test (32 kicks) was not evaluated.",
             "Each outer fold has an inner epoch-selection split. Original CSV and baseline artifacts unchanged.", "",
             "| Model | Features | Parameters | CV accuracy mean ± SD | CV macro F1 mean ± SD |",
             "|---|---:|---:|---:|---:|"]
    for row in ranked:
        accuracy, f1 = row["cv"]["accuracy"], row["cv"]["macro_f1"]
        lines.append(f"| {row['variant']} | {row['feature_count']} | {row['parameters']} | {accuracy['mean']*100:.2f}% ± {accuracy['std']*100:.2f}pp | {f1['mean']:.4f} ± {f1['std']:.4f} |")
    lines += ["", f"CV-selected candidate: **{ranked[0]['variant']}**.",
              "Variability is descriptive across 5 folds × 3 seeds, not an independent-sample confidence interval.",
              "Do not compare these CV scores directly with the previous single-split Test score.",
              "No final model was fitted on all 180 kicks and no Test score was used for selection.",
              "Coordinate jump flags are inspection candidates only; no cleaning was applied."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("FINAL "+json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def orchestrate(config, config_path):
    if tuple(config["variants"]) != VARIANTS:
        raise ValueError("Expected the four predefined architecture/feature variants")
    if not config["devices"]:
        raise ValueError("At least one device is required")
    info = prepare(config)
    out = ROOT/config["output_dir"]
    print(json.dumps({"prepared": str(out), "development": info["development"],
                      "protected_test": info["protected_test_kicks"], "fits": len(config["variants"])*config["n_splits"]*len(config["training_seeds"])}), flush=True)
    processes, handles, previous = {}, [], {}
    try:
        for variant in config["variants"]:
            handle = (out/f"{variant}.log").open("x", encoding="utf-8")
            handles.append(handle)
            processes[variant] = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                                   "--config", str(config_path.resolve()), "--worker", variant],
                                                  stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
        while processes:
            for variant, process in list(processes.items()):
                progress_path = out/variant/"progress.json"
                if progress_path.exists():
                    try:
                        progress = json.loads(progress_path.read_text())
                    except json.JSONDecodeError:
                        continue
                    if previous.get(variant) != progress["completed_fits"]:
                        previous[variant] = progress["completed_fits"]
                        print(json.dumps({"variant": variant, **progress}), flush=True)
                status = process.poll()
                if status is not None:
                    if status:
                        raise RuntimeError(f"Worker {variant} failed with exit {status}; see its log")
                    del processes[variant]
            if processes:
                time.sleep(1)
    finally:
        for process in processes.values():
            process.terminate()
        for process in processes.values():
            process.wait()
        for handle in handles:
            handle.close()
    return aggregate(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"compare_8j20_lr_cv_config.json")
    parser.add_argument("--worker", choices=VARIANTS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.worker:
        worker(config, args.worker)
    else:
        orchestrate(config, args.config)
