"""Build the four-channel penalty-kick dataset and train PenaltySTGCN with DDP."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from model import PenaltySTGCN


LABEL_NAMES = ["L", "C", "R"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fill_joint(values: np.ndarray, max_interior: int, max_boundary: int) -> np.ndarray | None:
    """Fill only short NaN runs; return None when a run exceeds policy."""
    result = values.astype(np.float64, copy=True)
    n = len(result)
    i = 0
    while i < n:
        if np.isfinite(result[i]):
            i += 1
            continue
        start = i
        while i < n and not np.isfinite(result[i]):
            i += 1
        end = i
        run = end - start
        boundary = start == 0 or end == n
        if run > (max_boundary if boundary else max_interior):
            return None
        if start == 0 and end == n:
            return None
        if start == 0:
            result[start:end] = result[end]
        elif end == n:
            result[start:end] = result[start - 1]
        else:
            result[start:end] = np.linspace(result[start - 1], result[end], run + 2)[1:-1]
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_samples(csv_path: Path, cfg: dict) -> tuple[list[dict], list[dict]]:
    data_cfg = cfg["data"]
    with csv_path.open(newline="") as handle:
        rows = [{k: v.strip() for k, v in row.items()} for row in csv.DictReader(handle)]
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["kick_id"], row["camera_direction"], row["track_id"])].append(row)

    samples, manifest = [], []
    joint_ids = data_cfg["joint_ids"]
    frames = data_cfg["frames"]
    stride = data_cfg["frame_stride"]
    margin = data_cfg["contact_margin_frames"]
    for key, group in groups.items():
        record = {"kick_id": key[0], "camera_direction": key[1], "track_id": key[2]}
        reasons = []
        if any(not row["frame"].isdigit() for row in group):
            reasons.append("invalid_frame")
        frame_numbers = [int(row["frame"]) for row in group if row["frame"].isdigit()]
        if len(frame_numbers) != len(set(frame_numbers)):
            reasons.append("duplicate_frame")
        if any(b <= a for a, b in zip(frame_numbers, frame_numbers[1:])):
            reasons.append("non_increasing_frame_order")
        labels = {row["kick_direction"] for row in group}
        if len(labels) != 1 or next(iter(labels), "") not in data_cfg["labels"]:
            reasons.append("invalid_or_inconsistent_label")
        touches = [int(row["frame"]) for row in group
                   if row["last_touch"] == "1" and row["frame"].isdigit()]
        if len(touches) != 1:
            reasons.append(f"last_touch_count_{len(touches)}")
        if reasons:
            record.update(included=False, reasons=reasons)
            manifest.append(record)
            continue

        by_frame = {int(row["frame"]): row for row in group}
        selected = [touches[0] - margin - stride * i for i in reversed(range(frames))]
        full_interval = range(selected[0], selected[-1] + 1)
        if any(frame not in by_frame for frame in full_interval):
            record.update(included=False, reasons=["missing_frame_in_interval"])
            manifest.append(record)
            continue
        interval_rows = [by_frame[frame] for frame in full_interval]
        positions = np.full((len(interval_rows), len(joint_ids), 2), np.nan, dtype=np.float64)
        for t, row in enumerate(interval_rows):
            for v, joint in enumerate(joint_ids):
                for axis, suffix in enumerate(("x", "y")):
                    value = row[f"kp_{joint}_{suffix}"]
                    if value:
                        try:
                            positions[t, v, axis] = float(value)
                        except ValueError:
                            pass
        original_missing = int(np.isnan(positions).any(axis=2).sum())
        missing_fraction = original_missing / (len(interval_rows) * len(joint_ids))
        if missing_fraction > data_cfg["max_missing_joint_fraction"]:
            record.update(included=False, reasons=["missing_fraction_exceeded"])
            manifest.append(record)
            continue
        failed = False
        for v in range(len(joint_ids)):
            for axis in range(2):
                filled = fill_joint(positions[:, v, axis],
                                    data_cfg["max_interior_missing_frames"],
                                    data_cfg["max_boundary_missing_frames"])
                if filled is None:
                    failed = True
                    break
                positions[:, v, axis] = filled
            if failed:
                break
        if failed or not np.isfinite(positions).all():
            record.update(included=False, reasons=["unfillable_joint_gap"])
            manifest.append(record)
            continue

        try:
            scales = np.asarray([float(row["bbox_h"]) for row in interval_rows])
        except ValueError:
            scales = np.asarray([])
        scale = float(np.median(scales)) if scales.size else float("nan")
        if not np.isfinite(scale) or scale <= 0:
            record.update(included=False, reasons=["invalid_bbox_scale"])
            manifest.append(record)
            continue
        hip_center = (positions[:, 6, :] + positions[:, 7, :]) / 2.0
        normalized = (positions - hip_center[:, None, :]) / scale
        offsets = np.asarray(selected) - selected[0]
        sampled = normalized[offsets]
        velocity = np.zeros_like(sampled)
        velocity[1:] = np.diff(sampled, axis=0) / np.diff(selected)[:, None, None]
        features = np.concatenate((sampled, velocity), axis=2).transpose(2, 0, 1)
        if not np.isfinite(features).all():
            record.update(included=False, reasons=["non_finite_features"])
            manifest.append(record)
            continue
        label = next(iter(labels))
        sample = {
            "x": features[..., None].astype(np.float32),
            "y": data_cfg["labels"][label],
            "label": label,
            "kick_id": key[0],
            "camera_direction": key[1],
            "track_id": key[2],
            "selected_frames": selected,
        }
        samples.append(sample)
        record.update(included=True, reasons=[], label=label,
                      selected_frames=selected, missing_joint_observations=original_missing)
        manifest.append(record)
    return samples, manifest


def grouped_split(samples: list[dict], cfg: dict) -> dict[str, list[str]]:
    label_by_kick: dict[str, str] = {}
    for sample in samples:
        previous = label_by_kick.setdefault(sample["kick_id"], sample["label"])
        if previous != sample["label"]:
            raise ValueError(f"Inconsistent label across kick {sample['kick_id']}")
    kicks = sorted(label_by_kick, key=lambda value: int(value) if value.isdigit() else value)
    labels = [label_by_kick[kick] for kick in kicks]
    trainval, test = train_test_split(
        kicks, test_size=cfg["training_proposal"]["test_fraction"],
        random_state=cfg["training_proposal"]["seed"], stratify=labels)
    trainval_labels = [label_by_kick[kick] for kick in trainval]
    train, val = train_test_split(
        trainval, test_size=cfg["training_proposal"]["validation_fraction_of_remaining"],
        random_state=cfg["training_proposal"]["seed"], stratify=trainval_labels)
    return {"train": sorted(train), "validation": sorted(val), "test": sorted(test)}


class KickDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return torch.from_numpy(sample["x"]), sample["y"], sample["kick_id"]


def metrics_from_records(records: list[tuple[str, int, list[float]]]) -> dict:
    by_kick: dict[str, list] = defaultdict(list)
    targets = {}
    for kick, target, probability in records:
        by_kick[kick].append(np.asarray(probability))
        targets[kick] = target
    ordered = sorted(by_kick)
    truth = np.asarray([targets[kick] for kick in ordered])
    pred = np.asarray([np.mean(by_kick[kick], axis=0).argmax() for kick in ordered])
    return {
        "loss": None,
        "num_kicks": len(ordered),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "recall_L_C_R": recall_score(truth, pred, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
        "confusion_matrix_L_C_R": confusion_matrix(truth, pred, labels=[0, 1, 2]).tolist(),
    }


def gather_records(local_records):
    if not dist.is_initialized():
        return local_records
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_records)
    return [item for part in gathered for item in part]


@torch.no_grad()
def evaluate(model, dataset, device, rank, world_size, criterion):
    model.eval()
    indices = list(range(rank, len(dataset), world_size))
    loader = DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=32, shuffle=False)
    records, total_loss, total_items = [], 0.0, 0
    for x, y, kick_ids in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += float(criterion(logits, y).item()) * len(y)
        total_items += len(y)
        probs = logits.softmax(dim=1).cpu().tolist()
        records.extend((kick, int(target), probability)
                       for kick, target, probability in zip(kick_ids, y.cpu().tolist(), probs))
    records = gather_records(records)
    loss_tensor = torch.tensor([total_loss, total_items], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(loss_tensor)
    result = metrics_from_records(records)
    result["loss"] = float(loss_tensor[0] / max(loss_tensor[1], 1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline_config.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--disable-early-stopping", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    with (root / args.config).open() as handle:
        cfg = json.load(handle)
    train_cfg = cfg["training_proposal"]
    seed_everything(train_cfg["seed"])

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world_size, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.dry_run else "cpu")
    seed_everything(train_cfg["seed"] + rank)

    csv_path = root / cfg["data"]["csv"]
    samples, manifest = build_samples(csv_path, cfg)
    splits = grouped_split(samples, cfg)
    split_by_kick = {kick: name for name, kicks in splits.items() for kick in kicks}
    partitioned = {name: [sample for sample in samples if split_by_kick[sample["kick_id"]] == name]
                   for name in splits}
    if rank == 0:
        out_dir = root / train_cfg["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "manifest.json").open("w") as handle:
            json.dump(manifest, handle, indent=2)
        with (out_dir / "split.json").open("w") as handle:
            json.dump(splits, handle, indent=2)
        run_info = {
            "csv_sha256": sha256(csv_path), "world_size": world_size,
            "sample_counts": {name: len(values) for name, values in partitioned.items()},
            "kick_counts": {name: len(splits[name]) for name in splits},
            "kick_label_counts": {name: dict(Counter(
                next(sample["label"] for sample in samples if sample["kick_id"] == kick)
                for kick in splits[name])) for name in splits},
            "config": cfg,
        }
        with (out_dir / "run_info.json").open("w") as handle:
            json.dump(run_info, handle, indent=2)
        print(json.dumps(run_info, indent=2), flush=True)
    if args.dry_run:
        model = PenaltySTGCN(cfg["model"]["dropout"]).to(device).eval()
        x, _, _ = next(iter(DataLoader(KickDataset(partitioned["train"]), batch_size=2)))
        with torch.no_grad():
            output = model(x.to(device))
        if rank == 0:
            print(f"DRY RUN PASS input={tuple(x.shape)} output={tuple(output.shape)}", flush=True)
        return

    if world_size != 4:
        raise RuntimeError(f"This confirmed run requires 4 processes/GPUs, got {world_size}")
    train_dataset = KickDataset(partitioned["train"])
    val_dataset = KickDataset(partitioned["validation"])
    test_dataset = KickDataset(partitioned["test"])
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=train_cfg["seed"], drop_last=False)
    loader = DataLoader(train_dataset, batch_size=train_cfg["per_gpu_batch_size"],
                        sampler=sampler, num_workers=0, pin_memory=True)
    model = PenaltySTGCN(cfg["model"]["dropout"]).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train_kick_counts = Counter(
        next(sample["label"] for sample in samples if sample["kick_id"] == kick)
        for kick in splits["train"])
    weights = torch.tensor([1.0 / train_kick_counts[name] for name in LABEL_NAMES], device=device)
    weights = weights / weights.sum() * len(LABEL_NAMES)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"],
                                  weight_decay=train_cfg["weight_decay"])
    epochs = args.max_epochs or train_cfg["max_epochs"]
    best_f1, best_loss, stale = -1.0, float("inf"), 0
    history = []
    out_dir = root / train_cfg["output_dir"]
    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        local_loss, local_items = 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip_norm"])
            optimizer.step()
            local_loss += float(loss.item()) * len(y)
            local_items += len(y)
        train_totals = torch.tensor([local_loss, local_items], device=device, dtype=torch.float64)
        dist.all_reduce(train_totals)
        val_metrics = evaluate(model, val_dataset, device, rank, world_size, criterion)
        improved = (val_metrics["macro_f1"] > best_f1 + 1e-12 or
                    (abs(val_metrics["macro_f1"] - best_f1) <= 1e-12 and val_metrics["loss"] < best_loss))
        if rank == 0:
            row = {"epoch": epoch, "train_loss": float(train_totals[0] / train_totals[1]),
                   "validation": val_metrics}
            history.append(row)
            print(json.dumps(row), flush=True)
            if improved:
                best_f1, best_loss, stale = val_metrics["macro_f1"], val_metrics["loss"], 0
                torch.save({"epoch": epoch, "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(), "validation": val_metrics,
                            "config": cfg, "splits": splits}, out_dir / "best.pt")
            else:
                stale += 1
            with (out_dir / "history.json").open("w") as handle:
                json.dump(history, handle, indent=2)
        state = torch.tensor([best_f1, best_loss, stale], device=device, dtype=torch.float64)
        dist.broadcast(state, src=0)
        best_f1, best_loss, stale = float(state[0]), float(state[1]), int(state[2])
        if not args.disable_early_stopping and stale >= train_cfg["early_stopping_patience"]:
            break

    dist.barrier()
    checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model.module.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_dataset, device, rank, world_size, criterion)
    if rank == 0:
        majority = max(train_kick_counts.values()) / sum(train_kick_counts.values())
        final = {"best_epoch": checkpoint["epoch"], "validation": checkpoint["validation"],
                 "test": test_metrics, "train_majority_accuracy": majority}
        with (out_dir / "metrics.json").open("w") as handle:
            json.dump(final, handle, indent=2)
        print("FINAL " + json.dumps(final), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
