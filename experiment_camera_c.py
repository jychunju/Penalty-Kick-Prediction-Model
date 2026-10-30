#!/usr/bin/env python3
"""C-camera-only L/R experiments, with the existing Test kept protected.

CLT camera C is an explicit user correction, NOT inferred from its filenames.
Only existing Train/Validation feature exports are parsed for training. Source
CSVs and old artifacts remain untouched; corrected metadata lives in a new copy.
Same source x label outer folds, stopping sets and seeds across all conditions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from compare_8j20_lr_cv import SmallPoseClassifier, make_folds, summarize_runs, write_predictions
from train_8j20_angles_lr import (
    FEATURES, LABELS, ROOT, evaluate, hash_file, load_samples, save_json, score, split_summary,
)

# Predeclared conditions; no adaptive selection after seeing outer-fold results.
CONDITIONS = {
    "stgcn20_xy": {"variant": "small_stgcn_xy", "window": "full20"},
    "stgcn10_xy": {"variant": "small_stgcn_xy", "window": "uniform10"},
    "stgcn_last10_xy": {"variant": "small_stgcn_xy", "window": "last10"},
    "stgcn20_xy_angles": {"variant": "small_stgcn_xy_angles", "window": "full20"},
    "cnn20_xy": {"variant": "small_cnn_xy", "window": "full20"},
    "cnn10_xy": {"variant": "small_cnn_xy", "window": "uniform10"},
    "stgcn10_xy_regularized": {
        "variant": "small_stgcn_xy", "window": "uniform10", "dropout": 0.5, "weight_decay": 0.01,
    },
}


def selected_frames(window):
    """Select existing chronological samples only; never interpolate or rotate."""
    if window == "full20":
        return list(range(20))
    if window == "uniform10":
        return np.rint(np.linspace(0, 19, 10)).astype(int).tolist()
    if window == "last10":
        return list(range(10, 20))
    raise ValueError(window)


class DynamicPoseClassifier(SmallPoseClassifier):
    """Same original parameters/operations, allowing a variable time dimension."""
    def forward(self, x):
        if x.ndim != 3 or x.shape[2] != self.feature_count or x.shape[1] < 2:
            raise ValueError("Expected (batch, time>=2, feature_count)")
        n, frames, _ = x.shape
        coordinates = self.coordinate_norm(x[:, :, :16].transpose(1, 2))
        if self.graph:
            coordinates = coordinates.reshape(n, 8, 2, frames).permute(0, 2, 3, 1)
            for block in self.blocks:
                coordinates = block(coordinates, self.adjacency)
            pooled = coordinates.mean(dim=(2, 3))
        else:
            pooled = self.coordinate_cnn(coordinates).squeeze(-1)
        if self.use_angles:
            angles = self.angle_branch(x[:, :, 16:].transpose(1, 2)).squeeze(-1)
            pooled = torch.cat((pooled, angles), dim=1)
        return self.head(pooled)


def effective_camera(sample):
    cameras = {r["camera_direction"].strip().upper() for r in sample["rows"]}
    if len(cameras) != 1:
        raise ValueError(f"Mixed camera metadata: {sample['group_id']}")
    if sample["dataset"] == "CLT":
        return "C"  # User-confirmed; filename suffix is not a camera annotation.
    camera = next(iter(cameras))
    if camera not in ("L", "R", "C"):
        raise ValueError(f"Unknown camera: {sample['group_id']}")
    return camera


def select_c_samples(samples, protected):
    """Exclude protected kicks before camera filtering; clone rows for correction."""
    kept, excluded = [], []
    for sample in samples:
        if sample["group_id"] in protected:
            raise ValueError("Protected Test kick reached development loading")
        camera = effective_camera(sample)
        if camera != "C":
            excluded.append({"group_id": sample["group_id"], "camera_direction": camera})
            continue
        clone = dict(sample)
        clone["rows"] = [dict(row, camera_direction="C") for row in sample["rows"]]
        kept.append(clone)
    kept.sort(key=lambda sample: sample["group_id"])
    return kept, excluded


def prepare(config):
    out, data, baseline = [ROOT/config[k] for k in ("output_dir", "data_dir", "baseline_run")]
    if out.exists():
        raise FileExistsError("Choose a new output_dir; never overwrite old experiments")
    split = json.loads((baseline/"split.json").read_text())
    old_info = json.loads((baseline/"run_info.json").read_text())
    source = ROOT/config["input_csv"]
    if hash_file(source) != old_info["input_sha256"]:
        raise ValueError("Source changed since the original train/validation/Test split")
    protected_paths = [source, baseline/"split.json", baseline/"run_info.json", baseline/"best.pt", baseline/"metrics.json",
                       *(data/f"{name}.csv" for name in ("train", "validation", "test")), data/"samples.npz"]
    protected_paths += list((ROOT/"Main Data").glob("*_penalty_pose_keypoints*.csv"))
    hashes = {str(p): hash_file(p) for p in protected_paths}
    development = []
    for phase in ("train", "validation"):
        fields, samples, excluded = load_samples(data/f"{phase}.csv")
        assert not excluded and {s["group_id"] for s in samples} == set(split[phase])
        development.extend(samples)
    samples, excluded = select_c_samples(development, set(split["test"]))
    folds = make_folds(samples, split["test"], config["n_splits"], config["split_seed"], config["early_stop_fraction"])
    x = np.stack([s["x"] for s in samples]); y = np.array([LABELS.index(s["label"]) for s in samples], dtype=np.int64)
    assert np.isfinite(x).all() and x.shape[1:] == (20, 20)
    tasks = [{"condition": name, "seed": seed, "fold": fold["fold"],
              "task_key": f"{name}/seed_{seed}/fold_{fold['fold']}"}
             for seed in config["training_seeds"] for fold in folds for name in CONDITIONS]
    info = {
        "config": config, "conditions": CONDITIONS, "development_shape": list(x.shape),
        "development": split_summary(samples, list(range(len(samples)))),
        "protected_test_groups": split["test"], "protected_test_kicks": len(split["test"]),
        "protected_file_hashes": hashes, "excluded_non_c_development": excluded,
        "source_code_hashes": {name: hash_file(ROOT/name) for name in (
            "experiment_camera_c.py", "compare_8j20_lr_cv.py", "train_8j20_angles_lr.py", "model.py")},
        "test_evaluations": 0, "test_features_used": False, "total_fits": len(tasks),
        "camera_rule": "CLT=C by user confirmation; EPL/Women use original camera_direction=C",
        "target_rule": "L/R kick_direction only; camera C is not kick_direction C",
        "checkpoint_selection": "Inner stopping macro F1 then loss; outer fold not used for stopping",
        "notes": ["Kick-level, not player/match-level split.",
                  "Fold/seed SD is descriptive, not a confidence interval.",
                  "Uniform10 selects sample indices, not uniform physical seconds; last10 is not fixed seconds.",
                  "Existing source metadata, labels, coordinates and prior experiments are untouched."],
    }
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/"run_info.json", info); save_json(out/"folds.json", folds); save_json(out/"tasks.json", tasks)
    np.savez_compressed(out/"development_only.npz", X=x, y=y,
                        group_id=np.array([s["group_id"] for s in samples]),
                        dataset=np.array([s["dataset"] for s in samples]), kick_id=np.array([s["kick_id"] for s in samples]))
    with (out/"camera_c_development.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for sample in samples:
            writer.writerows(sample["rows"])
    return info


def load_development(out):
    with np.load(out/"development_only.npz") as data:
        x, y = torch.from_numpy(data["X"].copy()), torch.from_numpy(data["y"].copy())
        samples = [{"group_id": g, "dataset": ds, "kick_id": k, "label": LABELS[int(v)]}
                   for g, ds, k, v in zip(data["group_id"].tolist(), data["dataset"].tolist(),
                                          data["kick_id"].tolist(), data["y"].tolist())]
    return x, y, samples


def fit_task(info, task, fold, full_x, y, samples, device, directory):
    spec = CONDITIONS[task["condition"]]; config = {**info["config"], **spec}
    seed = task["seed"]; fit_seed = seed + 1000*fold["fold"]
    random.seed(fit_seed); np.random.seed(fit_seed); torch.manual_seed(fit_seed); torch.cuda.manual_seed_all(fit_seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model = DynamicPoseClassifier(spec["variant"], config["dropout"]).to(device)
    frames = selected_frames(spec["window"])
    x = full_x[:, frames, :model.feature_count]
    train, stopping, outer = [fold["indices"][k] for k in ("inner_train", "early_stop", "validation")]
    counts = torch.bincount(y[train], minlength=2).float(); assert (counts > 0).all()
    weights = counts.sum()/(2*counts)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    loader = DataLoader(TensorDataset(x[train], y[train]), batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(fit_seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5)
    directory.mkdir(parents=True, exist_ok=False)
    best, best_epoch, history = (-1., -float("inf")), 0, []; started = time.perf_counter()
    for epoch in range(1, config["epochs"]+1):
        model.train(); total_loss, count = 0., 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"]); optimizer.step()
            total_loss += float(loss.detach())*len(yb); count += len(yb)
        stopping_metrics, _ = evaluate(model, x, y, stopping, samples, device)
        candidate = (stopping_metrics["macro_f1"], -stopping_metrics["loss"])
        history.append({"epoch": epoch, "train_loss": total_loss/count, "early_stop": stopping_metrics,
                        "lr": optimizer.param_groups[0]["lr"]})
        if candidate > best:
            best, best_epoch = candidate, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "condition": task["condition"],
                        "variant": spec["variant"], "config": config, "feature_columns": list(FEATURES[:model.feature_count]),
                        "frame_indices": frames, "training_indices": train, "seed": seed, "fit_seed": fit_seed,
                        "fold": fold["fold"], "early_stop": stopping_metrics, "class_weights": weights.tolist()}, directory/"best.pt")
        scheduler.step(stopping_metrics["loss"])
        if epoch-best_epoch >= config["early_stopping_patience"]:
            break
    checkpoint = torch.load(directory/"best.pt", map_location=device); model.load_state_dict(checkpoint["model"])
    validation, probabilities = evaluate(model, x, y, outer, samples, device)
    training, _ = evaluate(model, x, y, train, samples, device)
    majority_probs = np.zeros((len(outer), 2)); majority_probs[:, int(torch.argmax(counts))] = 1
    baseline = score(y[outer].numpy(), majority_probs, np.array([samples[i]["dataset"] for i in outer]))
    metrics = {**task, "fit_seed": fit_seed, "variant": spec["variant"], "frames": len(frames),
               "feature_count": model.feature_count, "frame_indices": frames, "dropout": config["dropout"],
               "weight_decay": config["weight_decay"], "parameters": sum(p.numel() for p in model.parameters()),
               "training_kicks": len(train), "best_epoch": best_epoch, "epochs_completed": len(history),
               "elapsed_seconds": time.perf_counter()-started, "train": training, "early_stop": checkpoint["early_stop"],
               "validation": validation, "majority_baseline": baseline, "outer_validation_evaluations": 1, "test_evaluations": 0}
    save_json(directory/"history.json", history); save_json(directory/"metrics.json", metrics)
    write_predictions(directory/"validation_predictions.csv", samples, outer, probabilities, seed, fold["fold"])
    return metrics


def worker(config, worker_id):
    out = ROOT/config["output_dir"]; info = json.loads((out/"run_info.json").read_text())
    tasks = json.loads((out/"tasks.json").read_text()); folds = json.loads((out/"folds.json").read_text())
    torch.set_num_threads(config["torch_threads_per_worker"])
    device = torch.device(config["devices"][worker_id])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    x, y, samples = load_development(out); assigned = tasks[worker_id::len(config["devices"])]
    for done, task in enumerate(assigned, 1):
        fold = next(f for f in folds if f["fold"] == task["fold"])
        metrics = fit_task(info, task, fold, x, y, samples, device, out/task["task_key"])
        save_json(out/f"worker_{worker_id}_progress.json", {"completed_fits": done, "total_fits": len(assigned), "last_task": task["task_key"]})
        print(json.dumps({"task": task["task_key"], "accuracy": metrics["validation"]["accuracy"],
                          "macro_f1": metrics["validation"]["macro_f1"], "best_epoch": metrics["best_epoch"]}), flush=True)


def aggregate(config):
    out = ROOT/config["output_dir"]; info = json.loads((out/"run_info.json").read_text())
    tasks = json.loads((out/"tasks.json").read_text()); folds = json.loads((out/"folds.json").read_text())
    _, y, samples = load_development(out); groups = [s["group_id"] for s in samples]
    sources = np.array([s["dataset"] for s in samples]); results = []
    for name, spec in CONDITIONS.items():
        runs, oof_scores = [], {}
        for seed in config["training_seeds"]:
            oof = np.full((len(samples), 2), np.nan, dtype=np.float32)
            for fold in folds:
                task = next(t for t in tasks if (t["condition"],t["seed"],t["fold"]) == (name,seed,fold["fold"]))
                directory = out/task["task_key"]; metrics = json.loads((directory/"metrics.json").read_text()); runs.append(metrics)
                with (directory/"validation_predictions.csv").open() as f:
                    rows = list(csv.DictReader(f))
                assert [r["group_id"] for r in rows] == fold["group_ids"]["validation"]
                oof[fold["indices"]["validation"]] = [[float(r["prob_L"]),float(r["prob_R"])] for r in rows]
            assert np.isfinite(oof).all()
            oof_scores[str(seed)] = score(y.numpy(), oof, sources)
            write_predictions(out/name/f"oof_seed_{seed}.csv", samples, list(range(len(samples))), oof, seed, "see folds.json")
        result = {"condition": name, "spec": spec, "cv": summarize_runs(runs), "per_seed_oof": oof_scores, "runs": runs,
                  "by_dataset": {ds: {metric: {"mean": float(np.mean([s["by_dataset"][ds][metric] for s in oof_scores.values()])),
                                                     "std": float(np.std([s["by_dataset"][ds][metric] for s in oof_scores.values()], ddof=1))}
                                         for metric in ("accuracy", "macro_f1")} for ds in sorted(set(sources))}}
        results.append(result)
    reference = next(r for r in results if r["condition"] == "stgcn20_xy")
    for result in results:
        delta = np.array([r["validation"]["accuracy"]-b["validation"]["accuracy"] for r,b in zip(result["runs"],reference["runs"])])
        result["paired_accuracy_vs_stgcn20"] = {"mean_delta": float(delta.mean()), "wins": int((delta>0).sum()),
                                                "ties": int((delta==0).sum()), "losses": int((delta<0).sum())}
    results.sort(key=lambda r:r["cv"]["macro_f1"]["mean"], reverse=True)
    baseline_values = [r["majority_baseline"]["accuracy"] for r in reference["runs"]]
    comparison = {"results": results, "selected_by_cv_macro_f1": results[0]["condition"], "total_fits": len(tasks),
                  "majority_accuracy_mean": float(np.mean(baseline_values)), "test_evaluations": 0}
    save_json(out/"comparison.json", comparison)
    with (out/"comparison.csv").open("x", newline="") as f:
        columns = ["condition","frames","feature_count","parameters","accuracy_mean","accuracy_std","macro_f1_mean","macro_f1_std"]
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in results:
            run=r["runs"][0]
            writer.writerow({"condition":r["condition"],**{k:run[k] for k in columns[1:4]},
                             **{f"{metric}_{stat}":r["cv"][metric][stat] for metric in ("accuracy","macro_f1") for stat in ("mean","std")}})
    lines = ["# C-camera-only L/R comparison", "", f"Development: {len(samples)} kicks; {info['development']['datasets']}; {info['development']['labels']}.",
             "CLT camera=C is user-confirmed; original CSVs are unchanged. Kick-direction C remains excluded.",
             "5 source×label stratified outer folds × 3 training seeds. Inner stopping only; protected Test never evaluated.", "",
             "| Condition | Accuracy mean ± SD | Macro F1 | Train accuracy |", "|---|---:|---:|---:|"]
    for r in results:
        cv=r["cv"];lines.append(f"| {r['condition']} | {100*cv['accuracy']['mean']:.2f}% ± {100*cv['accuracy']['std']:.2f} pp | {cv['macro_f1']['mean']:.4f} | {100*cv['training_accuracy']['mean']:.2f}% |")
    lines += ["", f"Training-fold majority baseline: {100*comparison['majority_accuracy_mean']:.2f}% mean accuracy.",
              "", "## Dataset OOF performance (mean over 3 seeds)", "", "| Condition | CLT accuracy | EPL accuracy | Women accuracy |", "|---|---:|---:|---:|"]
    for r in results:
        values=[f"{100*r['by_dataset'][ds]['accuracy']['mean']:.2f}%" for ds in ("CLT","EPL","Women")]
        lines.append(f"| {r['condition']} | " + " | ".join(values) + " |")
    lines += ["", "## Interpretation limits", "", "Fold/seed SD is descriptive, not a confidence interval; fits share data.",
              "C-camera restriction changes the population, sample size and folds. Do not read differences from earlier all-camera CV as causal effects.",
              "The best condition is a CV-selected candidate, not a new independent Test result. No all-development final model was fitted.",
              "Uniform10 is uniform over sampled indices; last10 is the last half of those indices, not fixed physical seconds.",
              "Split is kick-level; player/match-level generalization is not established."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return comparison


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,default=ROOT/"experiment_camera_c_config.json")
    parser.add_argument("--worker",type=int);parser.add_argument("--aggregate",action="store_true")
    args=parser.parse_args();config=json.loads(args.config.read_text());out=ROOT/config["output_dir"]
    if args.worker is not None:
        worker(config,args.worker);return
    if args.aggregate:
        aggregate(config);return
    info=prepare(config);print(json.dumps({"prepared":info["development"],"fits":info["total_fits"]}),flush=True)
    processes=[]
    try:
        for worker_id in range(len(config["devices"])):
            log=(out/f"worker_{worker_id}.log").open("x")
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--config",str(args.config.resolve()),"--worker",str(worker_id)],stdout=log,stderr=subprocess.STDOUT)
            processes.append((process,log))
        while any(p.poll() is None for p,_ in processes):
            failures=[p.returncode for p,_ in processes if p.poll() not in (None,0)]
            if failures:
                raise RuntimeError(f"Worker failed: {failures}; inspect worker logs")
            time.sleep(5)
        if any(p.returncode for p,_ in processes):
            raise RuntimeError("Training worker failed")
        aggregate(config);print("COMPLETED: "+str(out/"REPORT.md"),flush=True)
    finally:
        for p,log in processes:
            if p.poll() is None:
                p.terminate();p.wait()
            log.close()


if __name__ == "__main__":
    main()
