"""Train on the old CSV and evaluate once on the new CSV using six lower-body joints."""
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
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from model import LOWER_BODY_EDGES, PenaltySTGCN

LABEL_NAMES = ["L", "C", "R"]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def read_csv(path):
    with path.open(newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def make_sample(rows, key, cfg):
    labels = {r["kick_direction"] for r in rows}
    if len(labels) != 1 or next(iter(labels), "") not in cfg["labels"]:
        return None, "invalid_or_inconsistent_label"
    try: frames = [int(r["frame"]) for r in rows]
    except ValueError: return None, "invalid_frame"
    if len(frames) != len(set(frames)) or any(b <= a for a, b in zip(frames, frames[1:])):
        return None, "duplicate_or_non_increasing_frame"
    touches = [int(r["frame"]) for r in rows if r["last_touch"] == "1"]
    if len(touches) != 1: return None, f"last_touch_count_{len(touches)}"
    selected = list(range(touches[0] - cfg["frames"], touches[0]))
    by_frame = {int(r["frame"]): r for r in rows}
    if any(f not in by_frame for f in selected): return None, "incomplete_input_window"
    positions = np.empty((cfg["frames"], 6, 2), dtype=np.float64)
    try:
        for t, frame in enumerate(selected):
            row = by_frame[frame]
            for v, joint in enumerate(cfg["joint_ids"]):
                positions[t, v] = [float(row[f"kp_{joint}_x"]), float(row[f"kp_{joint}_y"])]
        scales = np.asarray([float(by_frame[f]["bbox_h"]) for f in selected])
    except (ValueError, KeyError): return None, "missing_or_invalid_lower_body_value"
    scale = float(np.median(scales))
    if not np.isfinite(positions).all() or not np.isfinite(scale) or scale <= 0:
        return None, "non_finite_or_invalid_scale"
    hip_center = (positions[:, 0] + positions[:, 1]) / 2
    normalized = (positions - hip_center[:, None, :]) / scale
    velocity = np.zeros_like(normalized)
    velocity[1:] = np.diff(normalized, axis=0)
    features = np.concatenate([normalized, velocity], axis=2).transpose(2, 0, 1)[..., None]
    return {"x": features.astype(np.float32), "y": cfg["labels"][next(iter(labels))],
            "label": next(iter(labels)), "kick_id": key[0], "sequence_id": "|".join(key),
            "selected_frames": selected}, None


def build_old(path, cfg):
    groups = defaultdict(list)
    for row in read_csv(path):
        groups[(row["kick_id"], row["camera_direction"], row["track_id"])].append(row)
    return build_groups(groups, cfg)


def build_new(path, cfg):
    groups = defaultdict(list)
    for row in read_csv(path): groups[(row["kick_id"],)].append(row)
    return build_groups(groups, cfg)


def build_groups(groups, cfg):
    samples, manifest = [], []
    for key, rows in groups.items():
        sample, reason = make_sample(rows, key, cfg)
        manifest.append({"sequence_id": "|".join(key), "kick_id": key[0],
                         "included": sample is not None, "reason": reason})
        if sample is not None: samples.append(sample)
    return samples, manifest


class Samples(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        s = self.items[i]
        return torch.from_numpy(s["x"]), s["y"], s["kick_id"]


def gather(items):
    if not dist.is_initialized(): return items
    parts = [None] * dist.get_world_size(); dist.all_gather_object(parts, items)
    return [item for part in parts for item in part]


def calculate_metrics(records):
    grouped, targets = defaultdict(list), {}
    for kick, target, probs in records:
        grouped[kick].append(np.asarray(probs)); targets[kick] = target
    kicks = sorted(grouped)
    truth = np.asarray([targets[k] for k in kicks])
    pred = np.asarray([np.mean(grouped[k], axis=0).argmax() for k in kicks])
    return {"num_kicks": len(kicks), "accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, labels=[0,1,2], average="macro", zero_division=0)),
            "recall_L_C_R": recall_score(truth, pred, labels=[0,1,2], average=None, zero_division=0).tolist(),
            "confusion_matrix_L_C_R": confusion_matrix(truth, pred, labels=[0,1,2]).tolist()}


@torch.no_grad()
def evaluate(model, dataset, device, rank, world, criterion):
    model.eval(); indices = list(range(rank, len(dataset), world)); records=[]; loss_sum=0.; count=0
    for x,y,kicks in DataLoader(Subset(dataset, indices), batch_size=32):
        x,y=x.to(device),y.to(device); logits=model(x); loss=criterion(logits,y)
        loss_sum += float(loss)*len(y); count += len(y)
        records += [(k,int(t),p) for k,t,p in zip(kicks,y.cpu().tolist(),logits.softmax(1).cpu().tolist())]
    records=gather(records); totals=torch.tensor([loss_sum,count],device=device,dtype=torch.float64)
    if dist.is_initialized():dist.all_reduce(totals)
    result=calculate_metrics(records); result["loss"]=float(totals[0]/max(totals[1],1)); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="cross_dataset_6j_config.json"); parser.add_argument("--dry-run",action="store_true"); args=parser.parse_args()
    root=Path(__file__).resolve().parent; cfg=json.loads((root/args.config).read_text()); seed_all(cfg["seed"])
    distributed="RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size(); local=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local); device=torch.device("cuda",local)
    else: rank=0;world=1;local=0;device=torch.device("cpu" if args.dry_run else "cuda:0")
    seed_all(cfg["seed"]+rank)
    old, old_manifest=build_old(root/cfg["train_csv"],cfg); external, external_manifest=build_new(root/cfg["external_test_csv"],cfg)
    label_by_kick={}
    for s in old:
        if s["kick_id"] in label_by_kick and label_by_kick[s["kick_id"]]!=s["label"]: raise ValueError("old kick label conflict")
        label_by_kick[s["kick_id"]]=s["label"]
    kicks=sorted(label_by_kick,key=lambda x:int(x)); labels=[label_by_kick[k] for k in kicks]
    train_kicks,val_kicks=train_test_split(kicks,test_size=cfg["validation_fraction"],random_state=cfg["seed"],stratify=labels)
    train_set=Samples([s for s in old if s["kick_id"] in set(train_kicks)]); val_set=Samples([s for s in old if s["kick_id"] in set(val_kicks)]); test_set=Samples(external)
    out=root/cfg["output_dir"]
    if rank==0:
        out.mkdir(parents=True,exist_ok=True)
        info={"train_sequences":len(train_set),"validation_sequences":len(val_set),"external_test_sequences":len(test_set),
              "train_kicks":len(train_kicks),"validation_kicks":len(val_kicks),"external_test_kicks":len({s['kick_id'] for s in external}),
              "train_kick_labels":dict(Counter(label_by_kick[k] for k in train_kicks)),"validation_kick_labels":dict(Counter(label_by_kick[k] for k in val_kicks)),
              "external_test_labels":dict(Counter(s['label'] for s in external)),"train_csv_sha256":file_hash(root/cfg['train_csv']),"test_csv_sha256":file_hash(root/cfg['external_test_csv']),"config":cfg}
        (out/"run_info.json").write_text(json.dumps(info,indent=2)); (out/"split.json").write_text(json.dumps({"train":sorted(train_kicks),"validation":sorted(val_kicks)},indent=2)); (out/"manifest.json").write_text(json.dumps({"old":old_manifest,"external":external_manifest},indent=2)); print(json.dumps(info,indent=2),flush=True)
    model=PenaltySTGCN(cfg["dropout"],num_joints=6,edges=LOWER_BODY_EDGES,roots=(0,1)).to(device)
    if args.dry_run:
        x,_,_=next(iter(DataLoader(train_set,batch_size=2))); y=model(x.to(device)); print("DRY RUN",tuple(x.shape),tuple(y.shape),flush=True); return
    if world!=4: raise RuntimeError(f"Expected 4 GPUs, got {world}")
    model=DDP(model,device_ids=[local],output_device=local)
    sampler=DistributedSampler(train_set,num_replicas=world,rank=rank,shuffle=True,seed=cfg["seed"])
    loader=DataLoader(train_set,batch_size=cfg["per_gpu_batch_size"],sampler=sampler,pin_memory=True)
    counts=Counter(label_by_kick[k] for k in train_kicks); weights=torch.tensor([1/counts[n] for n in LABEL_NAMES],device=device); weights=weights/weights.sum()*3
    criterion=nn.CrossEntropyLoss(weight=weights); optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    best=(-1.,float("inf")); history=[]
    for epoch in range(1,cfg["epochs"]+1):
        sampler.set_epoch(epoch);model.train();local_loss=0.;local_count=0
        for x,y,_ in loader:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True);logits=model(x);loss=criterion(logits,y);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip_norm"]);optimizer.step();local_loss+=float(loss)*len(y);local_count+=len(y)
        totals=torch.tensor([local_loss,local_count],device=device,dtype=torch.float64);dist.all_reduce(totals);val=evaluate(model,val_set,device,rank,world,criterion)
        if rank==0:
            row={"epoch":epoch,"train_loss":float(totals[0]/totals[1]),"validation":val};history.append(row);print(json.dumps(row),flush=True);score=(val["macro_f1"],-val["loss"])
            if score>(best[0],-best[1]): best=(val["macro_f1"],val["loss"]);torch.save({"epoch":epoch,"model":model.module.state_dict(),"validation":val,"config":cfg},out/"best.pt")
            (out/"history.json").write_text(json.dumps(history,indent=2))
    dist.barrier();checkpoint=torch.load(out/"best.pt",map_location=device);model.module.load_state_dict(checkpoint["model"]);external_metrics=evaluate(model,test_set,device,rank,world,criterion)
    if rank==0:
        final={"best_epoch":checkpoint["epoch"],"validation":checkpoint["validation"],"external_test":external_metrics};(out/"metrics.json").write_text(json.dumps(final,indent=2));print("FINAL "+json.dumps(final),flush=True)
    dist.destroy_process_group()

if __name__=="__main__":main()
