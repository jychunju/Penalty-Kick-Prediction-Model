#!/usr/bin/env python3
"""Train the final central-camera six-joint, contact-inclusive pre3 ST-GCN.

CLT remains an external holdout. Checkpoint selection uses validation only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# CUDA matrix multiplication requires this setting for deterministic execution.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import PenaltySTGCN
from prepare_windows import windows, FEATURES6

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")
LABELS = ("L", "R")
JOINTS = (5, 6, 11, 12, 13, 14, 15, 16)
EDGES = ((0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (4, 6), (3, 5), (5, 7))
ANGLES = ("knee_angle_l_n", "knee_angle_r_n", "hip_angle_l_n", "hip_angle_r_n")
FEATURES = tuple(f"kp_{joint}_{axis}n" for joint in JOINTS for axis in "xy") + ANGLES


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rounded_candidates(ideals, total):
    """Enumerate all floor/ceil allocations that conserve a group's total."""
    choices = [sorted({math.floor(float(v)), math.ceil(float(v))}) for v in ideals]
    return [np.array(values, dtype=int) for values in itertools.product(*choices)
            if sum(values) == total]


def allocate_source_counts(source_counts, split_sizes):
    """Choose joint integer quotas minimizing source percentage discrepancies.

    Every source/split count stays at floor or ceil of its ideal count; both
    source totals and global split sizes are preserved exactly. Small holdouts
    matter equally to train because errors are measured as proportions.
    """
    sources = sorted(source_counts)
    sizes = np.asarray(split_sizes, dtype=int)
    total = sum(source_counts.values())
    ideals = np.array([source_counts[source]*sizes/total for source in sources])
    candidates = [rounded_candidates(ideal, source_counts[source])
                  for source, ideal in zip(sources, ideals)]
    feasible = []
    for candidate in itertools.product(*candidates):
        allocation = np.stack(candidate)
        if np.array_equal(allocation.sum(axis=0), sizes):
            error = float(np.square((allocation-ideals)/sizes).sum())
            feasible.append((error, allocation))
    if not feasible:
        raise ValueError("No proportional integer allocation satisfies the split sizes")
    allocation = min(feasible, key=lambda pair: pair[0])[1]
    return {source: allocation[i] for i, source in enumerate(sources)}


def stratified_kick_split(samples, fractions, seed):
    """First conserve source proportions, then allocate L/R within each source.

    The second allocation minimizes overall L/R percentage error, with
    within-source L/R error as the tie-breaker. Randomness affects kick identity,
    not the carefully rounded source/label quotas.
    """
    fractions = np.asarray(fractions, dtype=float)
    if len(fractions) != 3 or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("Need three positive split fractions summing to one")
    n = len(samples)
    sizes = np.floor(fractions*n).astype(int)
    for i in np.argsort(-(fractions*n-sizes), kind="stable")[:n-int(sizes.sum())]:
        sizes[i] += 1
    if np.any(sizes == 0):
        raise ValueError("Not enough samples for three nonempty splits")
    by_source = defaultdict(list)
    for i, sample in enumerate(samples):
        by_source[sample["dataset"]].append(i)
    quotas = allocate_source_counts({key: len(value) for key, value in by_source.items()}, sizes)
    sources = sorted(by_source)
    left_totals = {source: sum(samples[i]["label"] == "L" for i in by_source[source])
                   for source in sources}
    left_candidates = [rounded_candidates(quotas[source]*left_totals[source]/len(by_source[source]),
                                          left_totals[source]) for source in sources]
    best = None
    for candidate in itertools.product(*left_candidates):
        if any(np.any(left < 1) or np.any(quotas[source]-left < 1)
               for source, left in zip(sources, candidate)):
            continue
        allocation = np.stack(candidate)
        overall_error = float(np.square(allocation.sum(axis=0)/sizes-sum(left_totals.values())/n).sum())
        within_error = sum(float(np.square(left/quotas[source]-left_totals[source]/len(by_source[source])).sum())
                           for source, left in zip(sources, candidate))
        score = (round(overall_error, 14), round(within_error, 14))
        if best is None or score < best[0]:
            best = (score, allocation)
    if best is None:
        raise ValueError("Not enough kicks for source + L/R stratification in every split")
    rng = random.Random(seed)
    parts = {name: [] for name in SPLITS}
    for source_index, source in enumerate(sources):
        for label in LABELS:
            indices = sorted((i for i in by_source[source] if samples[i]["label"] == label),
                             key=lambda i: samples[i]["kick_id"])
            rng.shuffle(indices)
            counts = best[1][source_index] if label == "L" else quotas[source]-best[1][source_index]
            cursor = 0
            for name, count in zip(SPLITS, counts):
                parts[name].extend(indices[cursor:cursor+int(count)])
                cursor += int(count)
            assert cursor == len(indices)
    for name in parts:
        parts[name].sort()
    flat = [i for values in parts.values() for i in values]
    assert len(flat) == n and sorted(flat) == list(range(n))
    assert [len(parts[name]) for name in SPLITS] == sizes.tolist()
    return parts


def load_samples(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields, rows = list(reader.fieldnames or []), list(reader)
    if not {"dataset", "kick_id", "kick_direction", "frame", "sample_20_idx", "sequence_id", "last_touch", *FEATURES} <= set(fields):
        raise ValueError("Required input columns are missing")
    if len(fields) != len(set(fields)) or any(None in row or None in row.values() for row in rows):
        raise ValueError("Malformed CSV")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["kick_id"])].append(row)
    samples, excluded = [], []
    for (dataset, kick), group in groups.items():
        labels = {row["kick_direction"] for row in group}
        if len(labels) != 1:
            raise ValueError(f"Conflicting direction labels in {dataset}/{kick}")
        label = next(iter(labels))
        if label == "C":
            excluded.append({"dataset": dataset, "kick_id": kick, "label": label})
            continue
        if label not in LABELS:
            raise ValueError(f"Unexpected direction {label}")
        if len(group) != 20 or [int(row["sample_20_idx"]) for row in group] != list(range(20)):
            raise ValueError(f"Expected 20 ordered samples in {dataset}/{kick}")
        times = np.array([float(row["frame"]) for row in group])
        if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
            raise ValueError(f"Invalid frame order in {dataset}/{kick}")
        if len({row["sequence_id"] for row in group}) != 1:
            raise ValueError("Multiple sequences in one kick")
        if float(group[-1]["last_touch"]) != 1:
            raise ValueError("Missing final last_touch")
        x = np.array([[float(row[column]) for column in FEATURES] for row in group], dtype=np.float32)
        if x.shape != (20, 20) or not np.isfinite(x).all():
            raise ValueError(f"Invalid features in {dataset}/{kick}")
        samples.append({"dataset": dataset, "kick_id": kick, "label": label,
                        "group_id": f"{dataset}:{kick}", "rows": group, "x": x})
    return fields, samples, excluded


def split_summary(samples, indices):
    selected = [samples[i] for i in indices]
    count = len(selected)
    sources = Counter(sample["dataset"] for sample in selected)
    return {"kicks": count, "rows": count*20, "labels": dict(Counter(s["label"] for s in selected)),
            "datasets": dict(sources), "dataset_percent": {source: 100*n/count for source, n in sources.items()},
            "dataset_label_counts": {source: dict(Counter(s["label"] for s in selected if s["dataset"] == source))
                                     for source in sorted(sources)}}


def score(truth, probabilities, sources):
    predicted = probabilities.argmax(axis=1)
    def one(mask):
        y, p = truth[mask], predicted[mask]
        return {"num_kicks": len(y), "accuracy": float(accuracy_score(y, p)),
                "macro_f1": float(f1_score(y, p, labels=[0, 1], average="macro", zero_division=0)),
                "recall_L_R": recall_score(y, p, labels=[0, 1], average=None, zero_division=0).tolist(),
                "confusion_matrix_L_R": confusion_matrix(y, p, labels=[0, 1]).tolist()}
    result = one(np.ones(len(truth), dtype=bool))
    result["by_dataset"] = {source: one(sources == source) for source in sorted(set(sources))}
    return result


@torch.no_grad()
def evaluate(model, x, y, indices, samples, device):
    model.eval()
    logits = model(x[indices].to(device)).cpu()
    loss = float(nn.functional.cross_entropy(logits, y[indices]))
    probabilities = logits.softmax(dim=1).numpy()
    result = score(y[indices].numpy(), probabilities, np.array([samples[i]["dataset"] for i in indices]))
    result["loss"] = loss
    return result, probabilities


def write_data(fields, samples, excluded, parts, data_dir, info):
    data_dir.mkdir(parents=True, exist_ok=False)
    membership = {i: name for name, indices in parts.items() for i in indices}
    for name, indices in parts.items():
        with (data_dir/f"{name}.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for i in indices:
                writer.writerows(samples[i]["rows"])
    manifest_fields = ["dataset", "kick_id", "group_id", "label", "split"]
    with (data_dir/"split_manifest.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields, lineterminator="\n")
        writer.writeheader()
        for i, sample in enumerate(samples):
            writer.writerow({**{key: sample[key] for key in manifest_fields[:-1]}, "split": membership[i]})
    np.savez_compressed(data_dir/"samples.npz", X=np.stack([s["x"] for s in samples]),
                        y=np.array([LABELS.index(s["label"]) for s in samples]),
                        group_id=np.array([s["group_id"] for s in samples]),
                        dataset=np.array([s["dataset"] for s in samples]),
                        split=np.array([membership[i] for i in range(len(samples))]), features=np.array(FEATURES))
    save_json(data_dir/"summary.json", info)
    save_json(data_dir/"excluded_c_kicks.json", excluded)


def run(config, dry_run=False):
    if config["condition"] != "pre3" or config["frames"] != 3 or not config["include_contact"] or tuple(config["joint_ids"]) != (11,12,13,14,15,16) or config["angle_columns"] != []:
        raise ValueError("This model expects contact-inclusive pre3, 6 lower-body joints and XY only")
    if config["labels"] != {"L": 0, "R": 1} or config["class_weighting"] != "inverse_train_kick_frequency":
        raise ValueError("Unsupported labels or class weighting")
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.set_num_threads(config["torch_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Deterministic reductions and seeded shuffle for reproducible single-GPU runs.
    torch.use_deterministic_algorithms(True)
    input_path = ROOT/config["input_csv"]
    input_hash = hash_file(input_path)
    fields, samples, excluded = load_samples(input_path)
    camera_exclusions=[s["group_id"] for s in samples if s["dataset"] != "CLT" and s["rows"][0]["camera_direction"] != "C"]
    samples=[s for s in samples if s["group_id"] not in camera_exclusions]
    _,eligibility_audit=windows(samples,"pre5",include_contact=True)
    excluded_windows=[a["group_id"] for a in eligibility_audit if "missing_frames" in a]
    samples=[s for s in samples if s["group_id"] not in excluded_windows]
    integrated = [i for i, sample in enumerate(samples) if sample["dataset"] != "CLT"]
    if {samples[i]["dataset"] for i in integrated} != {"EPL", "Women", "150penalties"}:
        raise ValueError("Expected exactly EPL, Women, and 150penalties in integrated pool")
    local=stratified_kick_split([samples[i] for i in integrated],config["split_fractions"],seed)
    parts={name:[integrated[i] for i in indices] for name,indices in local.items()}
    parts["test_clt"]=[i for i,s in enumerate(samples) if s["dataset"]=="CLT"]
    assert parts["test_clt"] and set(itertools.chain.from_iterable(parts.values())) == set(range(len(samples)))
    assert sum(map(len, parts.values())) == len(samples)
    assert all(samples[i]["dataset"] != "CLT" for name in SPLITS for i in parts[name])
    feature_keys = [hashlib.sha256(sample["x"].tobytes()).hexdigest() for sample in samples]
    membership = {i: name for name, indices in parts.items() for i in indices}
    duplicate_splits = defaultdict(set)
    for i, key in enumerate(feature_keys): duplicate_splits[key].add(membership[i])
    assert all(len(names) == 1 for names in duplicate_splits.values()), "Identical input sequences leaked across splits"
    window_x, window_audit = windows(samples, config["condition"],include_contact=True)
    x = torch.from_numpy(window_x)
    y = torch.tensor([LABELS.index(s["label"]) for s in samples], dtype=torch.long)
    info = {"input_csv": str(input_path), "input_sha256": input_hash, "full_source_hashes": {str(ROOT/"Main Data"/name):hash_file(ROOT/"Main Data"/name) for name in ["Combined_penalty_pose_keypoints_normalized.csv","150penalties_pose_keypoints_normalized.csv"]}, "input_shape": list(x.shape),
            "camera_exclusions": camera_exclusions, "camera_rule": "EPL/Women/150penalties: annotated C only; CLT=C by prior user confirmation", "include_contact": True, "missing_precontact_exclusions": excluded_windows, "feature_columns": list(FEATURES6), "condition": config["condition"], "window_audit": window_audit, "label_mapping": config["labels"], "excluded_c_kicks": len(excluded),
            "all": split_summary(samples, list(range(len(samples)))),
            **{name: split_summary(samples, indices) for name, indices in parts.items()},
            "split_method": "Central camera pool source-proportion optimized 70/15/15 split; CLT external only; common eligibility for all conditions",
            "grouping": "dataset + kick_id; all 20 frames kept together; CLT external holdout",
            "integrated_pool": split_summary(samples, integrated),
            "leakage_checks": {"disjoint_kicks": True, "identical_features_not_across_splits": True, "CLT_absent_from_train_validation_integrated_test": True},
            "ignored_as_model_input": ["goal", "dataset", "camera_direction", "r_or_l", "kick_id", "frame", "all other metadata and raw coordinates"],
            "class_weights_derived_from": "training kicks only", "config": config}
    device = torch.device("cpu" if dry_run else config["device"])
    model = PenaltySTGCN(config["dropout"]).to(device)
    info["model"] = {"name": "6-joint XY-only small ST-GCN", "parameters": sum(p.numel() for p in model.parameters()),
                     "device": str(device), "coordinates_shape": ["batch", 2, int(x.shape[1]), 6], "angles_used": False}
    print(json.dumps(info, indent=2), flush=True)
    if dry_run:
        model.eval()
        with torch.no_grad():
            output = model(x[:2].to(device))
        assert tuple(output.shape) == (2, 2)
        print(f"DRY_RUN_PASS: {tuple(x[:2].shape)} -> (2, 2); no files written", flush=True)
        return info
    out, data_dir = ROOT/config["output_dir"], ROOT/config["data_dir"]
    if out.exists() or data_dir.exists():
        raise FileExistsError("Run or data directory exists; choose new paths to preserve previous results")
    write_data(fields, samples, excluded, parts, data_dir, info)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/"run_info.json", info)
    np.savez_compressed(data_dir/"selected_windows.npz", X=window_x, group_id=np.array([s["group_id"] for s in samples]), features=np.array(FEATURES6))
    for name, indices in parts.items():
        with (data_dir/f"{name}_selected_windows.csv").open("x", newline="") as f:
            w=csv.DictWriter(f,fieldnames=["dataset","kick_id","label","step","source_frame",*FEATURES6]);w.writeheader()
            for i in indices:
                for step in range(window_x.shape[1]):
                    w.writerow({"dataset":samples[i]["dataset"],"kick_id":samples[i]["kick_id"],"label":samples[i]["label"],"step":step,"source_frame":window_audit[i]["frames"][step],**dict(zip(FEATURES6,map(float,window_x[i,step])))})
    save_json(out/"split.json", {name: [samples[i]["group_id"] for i in indices] for name, indices in parts.items()})
    (out/"model_architecture.txt").write_text(str(model), encoding="utf-8")
    shutil_source = Path(__file__).read_text(encoding="utf-8")
    (out/"training_code.py").write_text(shutil_source, encoding="utf-8")
    (out/"model_blocks_code.py").write_text((ROOT/"model.py").read_text(encoding="utf-8"), encoding="utf-8")
    counts = torch.bincount(y[parts["train"]], minlength=2).float()
    weights = counts.sum()/(2*counts)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    loader = DataLoader(TensorDataset(x[parts["train"]], y[parts["train"]]), batch_size=config["batch_size"],
                        shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5)
    best_score, best_epoch, history = (-1., -float("inf")), 0, []
    start = time.perf_counter()
    for epoch in range(1, config["epochs"]+1):
        model.train()
        loss_sum, count = 0., 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            loss_sum += float(loss.detach())*len(yb)
            count += len(yb)
        validation, _ = evaluate(model, x, y, parts["validation"], samples, device)
        row = {"epoch": epoch, "train_loss": loss_sum/count, "validation": validation,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        candidate = (validation["macro_f1"], -validation["loss"])
        if candidate > best_score:
            best_score, best_epoch = candidate, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation": validation,
                        "features": list(FEATURES6), "config": config, "train_class_weights": weights.tolist()}, out/"best.pt")
        scheduler.step(validation["loss"])
        save_json(out/"history.json", history)
        print(json.dumps({"epoch": epoch, "train_loss": row["train_loss"], "val_accuracy": validation["accuracy"],
                          "val_macro_f1": validation["macro_f1"], "val_loss": validation["loss"], "best_epoch": best_epoch}), flush=True)
        if epoch-best_epoch >= config["early_stopping_patience"]:
            break
    checkpoint = torch.load(out/"best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    # Test is untouched during epochs, checkpoint selection, and scheduling.
    test, probabilities = evaluate(model, x, y, parts["test"], samples, device)
    clt_test, clt_probabilities = evaluate(model, x, y, parts["test_clt"], samples, device)
    training, _ = evaluate(model, x, y, parts["train"], samples, device)
    assert hash_file(input_path) == input_hash
    assert all(hash_file(Path(p))==h for p,h in info["full_source_hashes"].items())
    final = {"best_epoch": best_epoch, "epochs_completed": len(history), "elapsed_seconds": time.perf_counter()-start,
             "selection_metric": "validation macro F1, then validation loss", "train": training,
             "validation": checkpoint["validation"], "test_integrated": test, "test_clt": clt_test, "train_class_weights_L_R": weights.tolist(),
             "input_file_unchanged": True, "test_evaluations_per_set": {"integrated": 1, "CLT": 1},
             "note": "Single exploratory seed; no player/match-level separation or cross-validation. Goal not used."}
    save_json(out/"metrics.json", final)
    for name, indices, probs in [("test_integrated", parts["test"], probabilities), ("test_clt", parts["test_clt"], clt_probabilities)]:
        with (out/f"{name}_predictions.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset", "kick_id", "actual", "predicted", "prob_L", "prob_R", "contact_status"])
            writer.writeheader()
            for i, probability in zip(indices, probs):
                writer.writerow({"dataset": samples[i]["dataset"], "kick_id": samples[i]["kick_id"],
                                 "actual": samples[i]["label"], "predicted": LABELS[int(probability.argmax())],
                                 "prob_L": float(probability[0]), "prob_R": float(probability[1]),
                                 "contact_status": samples[i]["rows"][0].get("contact_status", "")})
    print("FINAL "+json.dumps(final), flush=True)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")), dry_run=args.dry_run)
