#!/usr/bin/env python3
"""Paired frame-count ablation and nested learning curve; Test stays untouched.

Reuse the previous small-ST-GCN xy-only 20-frame/full-data reference (15 fits).
Train 75 new fits for uniform10, last10, and full20 with 25/50/75% training data.
All conditions share outer folds, inner stopping sets, training RNG seeds,
architecture, and optimizer settings. Training subsets are nested/stratified.
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
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from compare_8j20_lr_cv import SmallPoseClassifier, summarize_runs, write_predictions
from train_8j20_angles_lr import ROOT, evaluate, hash_file, save_json, score, split_summary

BASELINE = "full20_train100"
CONDITIONS = ("uniform10_train100", "last10_train100", "full20_train25", "full20_train50", "full20_train75")
ALL_CONDITIONS = (BASELINE, *CONDITIONS)


class VariableLengthSTGCN(SmallPoseClassifier):
    """Exactly the reference 11,074-parameter model with dynamic time reshape."""
    def __init__(self, dropout=0.3):
        super().__init__("small_stgcn_xy", dropout)

    def forward(self, x):
        if x.ndim != 3 or x.shape[2] != 16 or x.shape[1] < 2:
            raise ValueError("Expected (batch, time>=2, 16 normalized xy features)")
        n, frames, _ = x.shape
        coordinates = self.coordinate_norm(x.transpose(1, 2))
        coordinates = coordinates.reshape(n, 8, 2, frames).permute(0, 2, 3, 1)
        for block in self.blocks:
            coordinates = block(coordinates, self.adjacency)
        return self.head(coordinates.mean(dim=(2, 3)))


def frame_indices(condition):
    """Only select existing samples; no interpolation or physical-time claims."""
    if condition not in ALL_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if condition.startswith("uniform10"):
        return np.rint(np.linspace(0, 19, 10)).astype(int).tolist()
    if condition.startswith("last10"):
        return list(range(10, 20))
    return list(range(20))


def nested_training_subsets(samples, train_indices, fractions, seed):
    """A source x label balanced order makes all fraction subsets nested.

    Shuffle inside each stratum, then repeatedly choose the largest deficit
    against the ideal stratum count for the next prefix length. Subset identity
    stays fixed across training seeds and frame conditions. The 100% subset is
    exactly the existing sorted inner training set, preserving the reference.
    """
    if not fractions or any(f <= 0 or f > 1 for f in fractions):
        raise ValueError("Fractions must be in (0, 1]")
    buckets = defaultdict(list)
    for i in sorted(train_indices):
        buckets[(samples[i]["dataset"], samples[i]["label"])].append(i)
    rng = random.Random(seed)
    for key in sorted(buckets):
        rng.shuffle(buckets[key])
    totals = {key: len(value) for key, value in buckets.items()}
    chosen = Counter()
    order = []
    n = len(train_indices)
    for step in range(1, n+1):
        available = [key for key in sorted(buckets) if chosen[key] < totals[key]]
        key = max(available, key=lambda key: step*totals[key]/n-chosen[key])
        order.append(buckets[key][chosen[key]])
        chosen[key] += 1
    subsets = {}
    for fraction in fractions:
        size = min(n, max(1, int(np.floor(n*fraction+0.5))))
        selected = sorted(order[:size])
        if {(samples[i]["dataset"], samples[i]["label"]) for i in selected} != set(buckets):
            raise ValueError("Training subset too small to retain every source/label stratum")
        subsets[str(fraction)] = selected
    assert sorted(order) == sorted(train_indices)
    return subsets


def task_key(condition, seed, fold):
    return f"{condition}/seed_{seed}/fold_{fold}"


def prepare(config):
    reference, out = ROOT/config["reference_run"], ROOT/config["output_dir"]
    if out.exists():
        raise FileExistsError("Choose a new output_dir; existing results are preserved")
    if config["training_fractions"] != [0.25, 0.5, 0.75, 1.0]:
        raise ValueError("Expected predefined 25/50/75/100% learning curve")
    old_info = json.loads((reference/"run_info.json").read_text())
    validation = json.loads((reference/"artifact_validation.json").read_text())
    if validation["status"] != "PASS":
        raise ValueError("Reference experiment is not validated")
    for path, digest in old_info["protected_file_hashes"].items():
        if hash_file(Path(path)) != digest:
            raise ValueError(f"Original artifact changed: {path}")
    for name, digest in old_info["source_code_hashes"].items():
        if hash_file(ROOT/name) != digest:
            raise ValueError(f"Reference code changed: {name}")
    folds = json.loads((reference/"folds.json").read_text())
    with np.load(reference/"development_only.npz") as data:
        x, y = data["X"][:, :, :16].copy(), data["y"].copy()
        groups, sources, kicks = [data[key].astype(str).copy() for key in ("group_id", "dataset", "kick_id")]
    samples = [{"group_id": group, "dataset": source, "kick_id": kick, "label": ("L", "R")[int(target)]}
               for group, source, kick, target in zip(groups, sources, kicks, y)]
    assert x.shape == (180, 20, 16) and np.isfinite(x).all()
    assert not set(groups) & set(old_info["protected_test_groups"])
    train_cfg = old_info["config"]
    conditions = {condition: {"frame_indices": frame_indices(condition),
                              "training_fraction": (int(condition.split("train")[1])/100),
                              "reused_reference": condition == BASELINE} for condition in ALL_CONDITIONS}
    tasks = []
    fraction_by_condition = {condition: str(conditions[condition]["training_fraction"]) for condition in ALL_CONDITIONS}
    subset_manifests = []
    for fold in folds:
        subsets = nested_training_subsets(samples, fold["indices"]["inner_train"],
                                          config["training_fractions"], config["subset_seed"]+fold["fold"])
        assert subsets["1.0"] == fold["indices"]["inner_train"]
        subset_manifests.append({"fold": fold["fold"], "subsets": subsets,
                                 "distributions": {fraction: split_summary(samples, indices) for fraction, indices in subsets.items()}})
    for seed in train_cfg["training_seeds"]:
        for fold in folds:
            subsets = next(item["subsets"] for item in subset_manifests if item["fold"] == fold["fold"])
            for condition in CONDITIONS:
                indices = {**fold["indices"], "inner_train": subsets[fraction_by_condition[condition]]}
                for a, b in itertools.combinations(indices, 2):
                    assert not (set(indices[a]) & set(indices[b]))
                tasks.append({"condition": condition, "seed": seed, "fold": fold["fold"], "indices": indices,
                              "frame_indices": conditions[condition]["frame_indices"], "task_key": task_key(condition, seed, fold["fold"])})
    protected_hashes = dict(old_info["protected_file_hashes"])
    for name in ("run_info.json", "folds.json", "development_only.npz", "comparison.json", "artifact_validation.json"):
        protected_hashes[str(reference/name)] = hash_file(reference/name)
    for path in (reference/"small_stgcn_xy").rglob("*"):
        if path.is_file():
            protected_hashes[str(path)] = hash_file(path)
    info = {"config": config, "training_config": train_cfg, "conditions": conditions,
            "development_shape": list(x.shape), "development": split_summary(samples, list(range(len(samples)))),
            "protected_test_groups": old_info["protected_test_groups"], "protected_test_kicks": old_info["protected_test_kicks"],
            "protected_file_hashes": protected_hashes, "new_fits": len(tasks), "reused_reference_fits": 15,
            "test_evaluations": 0, "test_features_loaded": False, "angles_used": False, "parameters": 11074,
            "source_code_hashes": {name: hash_file(ROOT/name) for name in ("experiment_frames_learning_curve.py", "compare_8j20_lr_cv.py", "train_8j20_angles_lr.py", "model.py")},
            "notes": ["Uniform10 is uniform over existing sample indices, not raw physical time.",
                      "Last10 covers the second half of sampled sequence, not a fixed number of seconds.",
                      "Learning curve varies inner TRAIN kicks only; 22 stopping and 36 outer validation kicks stay fixed.",
                      "31/61/92/122 train kicks have nested source x direction stratified membership, fixed across seeds.",
                      "Equal max epochs/batch size, not equal gradient update budget; descriptive small-sample learning curve.",
                      "Earlier 20-frame/full-data fits are reused only after verifying identical data/folds/code/hyperparameters."]}
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/"run_info.json", info)
    save_json(out/"folds.json", folds)
    save_json(out/"training_subsets.json", subset_manifests)
    save_json(out/"tasks.json", tasks)
    np.savez_compressed(out/"development_only.npz", X=x, y=y, group_id=groups, dataset=sources, kick_id=kicks)
    return info


def fit_task(info, task, x, y, samples, device, directory):
    config = info["training_config"]
    seed, fold = task["seed"], task["fold"]
    fit_seed = seed+1000*fold
    random.seed(fit_seed); np.random.seed(fit_seed); torch.manual_seed(fit_seed); torch.cuda.manual_seed_all(fit_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model = VariableLengthSTGCN(config["dropout"]).to(device)
    assert sum(p.numel() for p in model.parameters()) == info["parameters"]
    selected_x = x[:, task["frame_indices"], :]
    train, stopping, outer = [task["indices"][name] for name in ("inner_train", "early_stop", "validation")]
    counts = torch.bincount(y[train], minlength=2).float()
    weights = counts.sum()/(2*counts)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    loader = DataLoader(TensorDataset(selected_x[train], y[train]), batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(fit_seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5)
    directory.mkdir(parents=True, exist_ok=False)
    best, best_epoch, history = (-1., -float("inf")), 0, []
    start = time.perf_counter()
    for epoch in range(1, config["epochs"]+1):
        model.train(); total_loss, count = 0., 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            total_loss += float(loss.detach())*len(yb); count += len(yb)
        stopping_metrics, _ = evaluate(model, selected_x, y, stopping, samples, device)
        candidate = (stopping_metrics["macro_f1"], -stopping_metrics["loss"])
        history.append({"epoch": epoch, "train_loss": total_loss/count, "early_stop": stopping_metrics,
                        "lr": optimizer.param_groups[0]["lr"]})
        if candidate > best:
            best, best_epoch = candidate, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "early_stop": stopping_metrics,
                        "condition": task["condition"], "config": info["config"], "training_config": config,
                        "seed": seed, "fit_seed": fit_seed, "fold": fold, "class_weights": weights.tolist(),
                        "frame_indices": task["frame_indices"], "training_indices": train}, directory/"best.pt")
        scheduler.step(stopping_metrics["loss"])
        if epoch-best_epoch >= config["early_stopping_patience"]:
            break
    checkpoint = torch.load(directory/"best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    validation, probabilities = evaluate(model, selected_x, y, outer, samples, device)
    training, _ = evaluate(model, selected_x, y, train, samples, device)
    baseline_probs = np.zeros((len(outer), 2)); baseline_probs[:, int(torch.argmax(counts))] = 1
    baseline = score(y[outer].numpy(), baseline_probs, np.array([samples[i]["dataset"] for i in outer]))
    metrics = {"condition": task["condition"], "seed": seed, "fold": fold, "fit_seed": fit_seed,
               "parameters": info["parameters"], "frames": len(task["frame_indices"]), "frame_indices": task["frame_indices"],
               "training_kicks": len(train), "best_epoch": best_epoch, "epochs_completed": len(history),
               "train": training, "early_stop": checkpoint["early_stop"], "validation": validation,
               "majority_baseline": baseline, "elapsed_seconds": time.perf_counter()-start,
               "outer_validation_evaluations": 1, "test_evaluations": 0}
    save_json(directory/"history.json", history)
    save_json(directory/"metrics.json", metrics)
    write_predictions(directory/"validation_predictions.csv", samples, outer, probabilities, seed, fold)
    return metrics


def worker(config, worker_id):
    out = ROOT/config["output_dir"]
    info = json.loads((out/"run_info.json").read_text())
    tasks = json.loads((out/"tasks.json").read_text())
    torch.set_num_threads(info["training_config"]["torch_threads_per_worker"])
    device = torch.device(config["devices"][worker_id])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    with np.load(out/"development_only.npz") as data:
        x, y = torch.from_numpy(data["X"].copy()), torch.from_numpy(data["y"].copy())
        samples = [{"group_id": group, "dataset": source, "kick_id": kick, "label": ("L", "R")[int(target)]}
                   for group, source, kick, target in zip(data["group_id"].tolist(), data["dataset"].tolist(), data["kick_id"].tolist(), data["y"].tolist())]
    assigned = tasks[worker_id::len(config["devices"])]
    for completed, task in enumerate(assigned, 1):
        result = fit_task(info, task, x, y, samples, device, out/task["task_key"])
        save_json(out/f"worker_{worker_id}_progress.json", {"completed_fits": completed, "total_fits": len(assigned),
                                                         "condition": task["condition"], "seed": task["seed"], "fold": task["fold"]})
        print(json.dumps({"task": task["task_key"], "accuracy": result["validation"]["accuracy"],
                          "macro_f1": result["validation"]["macro_f1"], "best_epoch": result["best_epoch"]}), flush=True)


def aggregate(config):
    out, reference = ROOT/config["output_dir"], ROOT/config["reference_run"]
    info = json.loads((out/"run_info.json").read_text())
    tasks = json.loads((out/"tasks.json").read_text())
    folds = json.loads((out/"folds.json").read_text())
    with np.load(out/"development_only.npz") as data:
        groups, sources, y = data["group_id"].tolist(), data["dataset"].astype(str), data["y"].copy()
    position = {group: i for i, group in enumerate(groups)}
    results = {}
    for condition in ALL_CONDITIONS:
        runs, seed_probabilities = [], {}
        for seed in info["training_config"]["training_seeds"]:
            oof = np.full((len(groups), 2), np.nan, dtype=np.float32)
            for fold in folds:
                if condition == BASELINE:
                    directory = reference/"small_stgcn_xy"/f"seed_{seed}"/f"fold_{fold['fold']}"
                else:
                    directory = out/task_key(condition, seed, fold["fold"])
                metrics = json.loads((directory/"metrics.json").read_text())
                record = {**metrics, "condition": condition, "frames": len(frame_indices(condition)),
                          "training_kicks": metrics["train"]["num_kicks"], "reused_reference": condition == BASELINE,
                          "artifact_directory": str(directory)}
                runs.append(record)
                with (directory/"validation_predictions.csv").open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                assert {row["group_id"] for row in rows} == set(fold["group_ids"]["validation"])
                for row in rows:
                    i = position[row["group_id"]]
                    assert np.isnan(oof[i]).all()
                    assert row["actual"] == ("L", "R")[y[i]]
                    oof[i] = [float(row["prob_L"]), float(row["prob_R"])]
            assert np.isfinite(oof).all()
            seed_probabilities[str(seed)] = score(y, oof, sources)
        results[condition] = {"condition": condition, **info["conditions"][condition],
                              "frames": len(frame_indices(condition)), "parameters": 11074,
                              "training_kicks": sorted({run["training_kicks"] for run in runs}),
                              "cv": summarize_runs(runs), "per_seed_oof": seed_probabilities, "runs": runs}
    baseline_runs = {(run["seed"], run["fold"]): run for run in results[BASELINE]["runs"]}
    for condition, result in results.items():
        paired = {}
        for metric in ("accuracy", "macro_f1"):
            deltas = np.array([run["validation"][metric]-baseline_runs[(run["seed"], run["fold"])]["validation"][metric]
                               for run in result["runs"]])
            paired[metric] = {"mean_delta": float(deltas.mean()), "std_delta": float(deltas.std(ddof=1)),
                              "wins": int((deltas > 1e-12).sum()), "ties": int((np.abs(deltas) <= 1e-12).sum()),
                              "losses": int((deltas < -1e-12).sum())}
        result["paired_against_full20_train100"] = paired
    temporal_conditions = (BASELINE, "uniform10_train100", "last10_train100")
    learning_conditions = ("full20_train25", "full20_train50", "full20_train75", BASELINE)
    chosen = max(temporal_conditions, key=lambda key: (results[key]["cv"]["macro_f1"]["mean"], results[key]["cv"]["accuracy"]["mean"]))
    for path, digest in info["protected_file_hashes"].items():
        if hash_file(Path(path)) != digest:
            raise RuntimeError(f"Protected file changed: {path}")
    summary = {"development_kicks": 180, "protected_test_kicks": 32, "new_fits": len(tasks),
               "reused_reference_fits": 15, "total_scored_fits": sum(r["cv"]["num_fits"] for r in results.values()),
               "test_evaluations": 0, "parameters_all_conditions": 11074,
               "temporal_conditions": list(temporal_conditions), "learning_conditions": list(learning_conditions),
               "selected_temporal_condition_by_cv_macro_f1": chosen, "results": results,
               "protected_files_unchanged": True,
               "interpretation": "Exploratory paired CV, not independent Test accuracy. SD is descriptive over 15 correlated fold/seed scores. Frame window and data fraction are separate experiments."}
    save_json(out/"comparison.json", summary)
    for filename, conditions in (("frame_comparison.csv", temporal_conditions), ("learning_curve.csv", learning_conditions)):
        with (out/filename).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["condition", "frames", "training_kicks", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "training_accuracy_mean"])
            writer.writeheader()
            for key in conditions:
                result = results[key]
                writer.writerow({"condition": key, "frames": result["frames"], "training_kicks": result["training_kicks"][0],
                                 **{f"{metric}_{stat}": result["cv"][metric][stat] for metric in ("accuracy", "macro_f1") for stat in ("mean", "std")},
                                 "training_accuracy_mean": result["cv"]["training_accuracy"]["mean"]})
    # A learning-curve visualization is useful for the data-size question.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    counts = [results[key]["training_kicks"][0] for key in learning_conditions]
    for metric, label, color in (("training_accuracy", "Train", "tab:blue"), ("accuracy", "Outer CV validation", "tab:orange")):
        means = [100*results[key]["cv"][metric]["mean"] for key in learning_conditions]
        deviations = [100*results[key]["cv"][metric]["std"] for key in learning_conditions]
        ax.errorbar(counts, means, yerr=deviations, marker="o", capsize=4, color=color, label=label)
    ax.set(xlabel="Independent training kicks per fold", ylabel="Accuracy (%)", ylim=(20, 100), xticks=counts,
           title="Full 20 frames | fixed validation sets | 5 folds x 3 seeds")
    ax.text(0.02, 0.02, "Error bars: descriptive SD, not confidence intervals", transform=ax.transAxes, fontsize=8)
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(out/"learning_curve.png", dpi=160); plt.close(fig)
    lines = ["# Frame-count / training-size experiments", "", "Small ST-GCN xy-only; 11,074 parameters in every condition.",
             "Same 5 folds × 3 seeds. Inner-only stopping. Original Test 32 kicks not evaluated.",
             "The identical previous full20/train100 reference fits are reused (15); 75 new fits are trained.", "",
             "## Frame comparison", "", "| Condition | Frames | Training kicks | CV accuracy mean ± SD | CV macro F1 |",
             "|---|---:|---:|---:|---:|"]
    for key in temporal_conditions:
        r = results[key]
        lines.append(f"| {key} | {r['frames']} | {r['training_kicks'][0]} | {100*r['cv']['accuracy']['mean']:.2f}% ± {100*r['cv']['accuracy']['std']:.2f}pp | {r['cv']['macro_f1']['mean']:.4f} |")
    lines += ["", f"CV-selected temporal candidate: **{chosen}**.",
              "Uniform10 selects existing sample indices [0,2,4,6,8,11,13,15,17,19]. Last10 selects indices 10–19.",
              "Neither is a fixed physical-time window. No coordinates or source CSVs were changed.", "",
              "## Learning curve (full 20 frames)", "", "| Training fraction | Training kicks | Train accuracy | CV accuracy mean ± SD | CV macro F1 |",
              "|---|---:|---:|---:|---:|"]
    for key in learning_conditions:
        r = results[key]
        lines.append(f"| {100*r['training_fraction']:.0f}% | {r['training_kicks'][0]} | {100*r['cv']['training_accuracy']['mean']:.2f}% | {100*r['cv']['accuracy']['mean']:.2f}% ± {100*r['cv']['accuracy']['std']:.2f}pp | {r['cv']['macro_f1']['mean']:.4f} |")
    lines += ["", "![Learning curve](learning_curve.png)", "",
              "Training subsets are nested and source × L/R balanced; inner stopping 22 and outer validation 36 kicks are fixed.",
              "Maximum epochs and batch size are fixed, not gradient-update budget. Learning-curve conclusions are descriptive.",
              "SD is over correlated fold/seed scores, not a confidence interval. No final model/Test evaluation is performed.",
              "Data outside this small observed range may behave differently; this does not establish that more data cannot help."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("FINAL "+json.dumps({"selected_temporal": chosen, "new_fits": len(tasks), "test_evaluations": 0,
          "results": {key: {"accuracy": r["cv"]["accuracy"], "macro_f1": r["cv"]["macro_f1"], "train_kicks": r["training_kicks"]} for key, r in results.items()}}), flush=True)
    return summary


def orchestrate(config, config_path):
    if not config["devices"]:
        raise ValueError("At least one device is required")
    info = prepare(config)
    out = ROOT/config["output_dir"]
    print(json.dumps({"prepared": str(out), "new_fits": info["new_fits"], "reused_fits": 15,
                      "protected_test": 32, "conditions": info["conditions"]}), flush=True)
    processes, handles, previous = {}, [], {}
    try:
        for slot in range(len(config["devices"])):
            handle = (out/f"worker_{slot}.log").open("x", encoding="utf-8")
            handles.append(handle)
            processes[slot] = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                               "--config", str(config_path.resolve()), "--worker", str(slot)],
                                              stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
        while processes:
            for slot, process in list(processes.items()):
                progress_path = out/f"worker_{slot}_progress.json"
                if progress_path.exists():
                    try:
                        progress = json.loads(progress_path.read_text())
                    except json.JSONDecodeError:
                        continue
                    if previous.get(slot) != progress["completed_fits"]:
                        previous[slot] = progress["completed_fits"]
                        print(json.dumps({"worker": slot, **progress}), flush=True)
                status = process.poll()
                if status is not None:
                    if status:
                        raise RuntimeError(f"Worker {slot} failed (exit {status}); see worker_{slot}.log")
                    del processes[slot]
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
    parser.add_argument("--config", type=Path, default=ROOT/"experiment_frames_learning_curve_config.json")
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.worker is None:
        orchestrate(config, args.config)
    else:
        worker(config, args.worker)
