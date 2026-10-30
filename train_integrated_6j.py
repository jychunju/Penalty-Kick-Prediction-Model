#!/usr/bin/env python3
"""Train a 3-class six-joint ST-GCN from the prepared integrated NPZ."""

from __future__ import annotations

import argparse
import csv
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

LABELS = ("L", "C", "R")


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class Samples(Dataset):
    def __init__(self, x, y, original_y, mirrored, groups, sequences, sources):
        self.x=x;self.y=y;self.original_y=original_y;self.mirrored=mirrored;self.groups=groups;self.sequences=sequences;self.sources=sources
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        return torch.from_numpy(self.x[i]),int(self.y[i]),int(self.original_y[i]),bool(self.mirrored[i]),str(self.groups[i]),str(self.sequences[i]),str(self.sources[i])


def gather(items):
    if not dist.is_initialized(): return items
    parts=[None]*dist.get_world_size();dist.all_gather_object(parts,items)
    return [x for p in parts for x in p]


def score(records):
    by_group=defaultdict(list);target={};source={}
    for group,y,prob,src,_seq in records:
        by_group[group].append(np.asarray(prob));target[group]=y;source[group]=src
    groups=sorted(by_group);truth=np.asarray([target[g] for g in groups]);pred=np.asarray([np.mean(by_group[g],axis=0).argmax() for g in groups])
    def one(mask):
        y=truth[mask];p=pred[mask]
        return {"num_kicks":int(mask.sum()),"accuracy":float(accuracy_score(y,p)),
                "macro_f1":float(f1_score(y,p,labels=[0,1,2],average="macro",zero_division=0)),
                "recall_L_C_R":recall_score(y,p,labels=[0,1,2],average=None,zero_division=0).tolist(),
                "confusion_matrix_L_C_R":confusion_matrix(y,p,labels=[0,1,2]).tolist()}
    result=one(np.ones(len(groups),bool));result["by_source"]={}
    for src in sorted(set(source.values())):result["by_source"][src]=one(np.asarray([source[g]==src for g in groups]))
    return result,groups,truth,pred


@torch.no_grad()
def evaluate(model,dataset,device,rank,world,criterion):
    model.eval();records=[];loss_sum=0.;count=0
    indices=list(range(rank,len(dataset),world))
    for x,y,original_y,mirrored,groups,sequences,sources in DataLoader(Subset(dataset,indices),batch_size=64):
        x=x.to(device);y=y.to(device);logits=model(x);loss=criterion(logits,y);probs=logits.softmax(1).cpu().numpy()
        # Convert canonical right-foot probabilities back to actual pitch
        # directions for left-foot samples before reporting metrics.
        mirrored_np=mirrored.numpy().astype(bool);probs[mirrored_np]=probs[mirrored_np][:,[2,1,0]]
        probs=probs.tolist()
        loss_sum+=float(loss)*len(y);count+=len(y)
        records += [(g,int(t),p,s,q) for g,t,p,s,q in zip(groups,original_y.tolist(),probs,sources,sequences)]
    records=gather(records);tot=torch.tensor([loss_sum,count],dtype=torch.float64,device=device)
    if dist.is_initialized():dist.all_reduce(tot)
    result,groups,truth,pred=score(records);result["loss"]=float(tot[0]/max(tot[1],1))
    return result,records,groups,truth,pred


def split_groups(y,groups,seed,test_fraction,val_fraction):
    label_by_group={}
    for yi,g in zip(y,groups):
        old=label_by_group.setdefault(str(g),int(yi))
        if old!=int(yi):raise ValueError(f"Conflicting canonical labels for {g}")
    names=sorted(label_by_group);labels=[label_by_group[g] for g in names]
    trainval,test=train_test_split(names,test_size=test_fraction,random_state=seed,stratify=labels)
    trainval_y=[label_by_group[g] for g in trainval]
    train,val=train_test_split(trainval,test_size=val_fraction,random_state=seed,stratify=trainval_y)
    return {"train":sorted(train),"validation":sorted(val),"test":sorted(test)},label_by_group


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="integrated_6j_train_config.json");p.add_argument("--dry-run",action="store_true");a=p.parse_args()
    root=Path(__file__).resolve().parent;cfg=json.loads((root/a.config).read_text());seed_all(cfg["seed"])
    distributed="RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl");rank=dist.get_rank();world=dist.get_world_size();local=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(local);device=torch.device("cuda",local)
    else:rank=0;world=1;local=0;device=torch.device("cpu" if a.dry_run else "cuda:0")
    seed_all(cfg["seed"]+rank)
    data=np.load(root/cfg["dataset"]);x=data["X"];y=data["y"];original_y=data["original_y"];mirrored=data["mirrored"];sources=data["source"].astype(str);groups=data["group_id"].astype(str);sequences=data["sequence_id"].astype(str)
    split,label_by_group=split_groups(original_y,groups,cfg["seed"],cfg["test_fraction"],cfg["validation_fraction_of_trainval"])
    membership={g:k for k,v in split.items() for g in v};parts={k:np.flatnonzero([membership[g]==k for g in groups]) for k in split}
    sets={k:Samples(x[idx],y[idx],original_y[idx],mirrored[idx],groups[idx],sequences[idx],sources[idx]) for k,idx in parts.items()};out=root/cfg["output_dir"]
    if rank==0:
        out.mkdir(parents=True,exist_ok=True)
        def summary(k):
            gs=split[k];return {"kicks":len(gs),"sequences":len(parts[k]),"labels":dict(Counter(LABELS[label_by_group[g]] for g in gs)),"sources":dict(Counter(g.split(":",1)[0] for g in gs))}
        info={"dataset":cfg["dataset"],"input_shape":list(x.shape),"all":{"kicks":len(label_by_group),"sequences":len(y),"actual_direction_labels":dict(Counter(LABELS[v] for v in label_by_group.values())),"canonical_training_sequence_labels":dict(Counter(LABELS[int(v)] for v in y)),"mirrored_sequences":int(mirrored.sum())},**{k:summary(k) for k in split},"config":cfg}
        (out/"run_info.json").write_text(json.dumps(info,indent=2));(out/"split.json").write_text(json.dumps(split,indent=2));print(json.dumps(info,indent=2),flush=True)
    model=PenaltySTGCN(cfg["dropout"],num_joints=6,edges=LOWER_BODY_EDGES,roots=(0,1),num_classes=3).to(device)
    if a.dry_run:
        xb,*_=next(iter(DataLoader(sets["train"],batch_size=2)));print("DRY_RUN",tuple(xb.shape),tuple(model(xb.to(device)).shape));return
    if world!=4:raise RuntimeError(f"Expected 4 GPUs, got {world}")
    model=DDP(model,device_ids=[local],output_device=local)
    sampler=DistributedSampler(sets["train"],num_replicas=world,rank=rank,shuffle=True,seed=cfg["seed"],drop_last=False)
    loader=DataLoader(sets["train"],batch_size=cfg["per_gpu_batch_size"],sampler=sampler,pin_memory=True)
    train_idx=parts["train"];counts=Counter(int(v) for v in y[train_idx]);weights=torch.tensor([1/counts[i] for i in range(3)],device=device);weights=weights/weights.sum()*3
    criterion=nn.CrossEntropyLoss(weight=weights);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    history=[];best=(-1.,float("inf"))
    for epoch in range(1,cfg["epochs"]+1):
        sampler.set_epoch(epoch);model.train();loss_sum=0.;count=0
        for xb,yb,*_ in loader:
            xb=xb.to(device,non_blocking=True);yb=yb.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True);logits=model(xb);loss=criterion(logits,yb);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip_norm"]);optimizer.step();loss_sum+=float(loss)*len(yb);count+=len(yb)
        tot=torch.tensor([loss_sum,count],dtype=torch.float64,device=device);dist.all_reduce(tot)
        val,*_=evaluate(model,sets["validation"],device,rank,world,criterion)
        if rank==0:
            row={"epoch":epoch,"train_loss":float(tot[0]/tot[1]),"validation":val};history.append(row);print(json.dumps(row),flush=True)
            candidate=(val["macro_f1"],-val["loss"])
            if candidate>(best[0],-best[1]):
                best=(val["macro_f1"],val["loss"]);torch.save({"epoch":epoch,"model":model.module.state_dict(),"validation":val,"config":cfg},out/"best.pt")
            (out/"history.json").write_text(json.dumps(history,indent=2))
    dist.barrier();checkpoint=torch.load(out/"best.pt",map_location=device);model.module.load_state_dict(checkpoint["model"])
    test,records,test_groups,truth,pred=evaluate(model,sets["test"],device,rank,world,criterion)
    if rank==0:
        final={"best_epoch":checkpoint["epoch"],"epochs_completed":cfg["epochs"],"selection_metric":"validation_macro_f1_then_loss","validation":checkpoint["validation"],"test":test}
        (out/"metrics.json").write_text(json.dumps(final,indent=2))
        by_group=defaultdict(list);targets={};srcs={}
        for g,t,prob,src,seq in records:by_group[g].append(np.asarray(prob));targets[g]=t;srcs[g]=src
        rows=[]
        for g in sorted(by_group):
            prob=np.mean(by_group[g],axis=0);rows.append({"group_id":g,"source":srcs[g],"actual":LABELS[targets[g]],"predicted":LABELS[int(prob.argmax())],"prob_L":prob[0],"prob_C":prob[1],"prob_R":prob[2],"num_views":len(by_group[g])})
        with (out/"test_predictions.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        import matplotlib.pyplot as plt
        cm=np.asarray(test["confusion_matrix_L_C_R"]);fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(cm,cmap="Blues");ax.set_xticks(range(3),LABELS);ax.set_yticks(range(3),LABELS);ax.set_xlabel("Predicted");ax.set_ylabel("Actual");ax.set_title("Test confusion matrix")
        for i in range(3):
            for j in range(3):ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/"confusion_matrix.png",dpi=180);plt.close(fig)
        print("FINAL "+json.dumps(final),flush=True)
    dist.destroy_process_group()


if __name__=="__main__":main()
