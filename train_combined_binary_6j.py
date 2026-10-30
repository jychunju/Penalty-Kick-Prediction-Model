"""Train/test a two-class six-joint ST-GCN on the combined CSV datasets."""
from __future__ import annotations

import argparse
import json
import os
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
from train_cross_dataset_6j import build_new, build_old, file_hash, seed_all

LABEL_NAMES = ["L", "R"]


class BinarySamples(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        item = self.items[index]
        return torch.from_numpy(item["x"]), item["y"], item["group_id"], item["source"]


def gather(items):
    if not dist.is_initialized(): return items
    parts = [None] * dist.get_world_size(); dist.all_gather_object(parts, items)
    return [item for part in parts for item in part]


def metrics(records):
    by_kick, targets, sources = defaultdict(list), {}, {}
    for kick, target, probability, source in records:
        by_kick[kick].append(np.asarray(probability)); targets[kick] = target; sources[kick] = source
    kicks = sorted(by_kick); truth=np.asarray([targets[k] for k in kicks]); pred=np.asarray([np.mean(by_kick[k],axis=0).argmax() for k in kicks])
    def score(mask):
        y, p = truth[mask], pred[mask]
        return {"num_kicks":int(mask.sum()),"accuracy":float(accuracy_score(y,p)),
                "macro_f1":float(f1_score(y,p,labels=[0,1],average="macro",zero_division=0)),
                "recall_L_R":recall_score(y,p,labels=[0,1],average=None,zero_division=0).tolist(),
                "confusion_matrix_L_R":confusion_matrix(y,p,labels=[0,1]).tolist()}
    result=score(np.ones(len(kicks),dtype=bool)); result["by_source"]={}
    for source in sorted(set(sources.values())):
        result["by_source"][source]=score(np.asarray([sources[k]==source for k in kicks]))
    return result


@torch.no_grad()
def evaluate(model,dataset,device,rank,world,criterion,export=False):
    model.eval(); records=[]; loss_sum=0.; count=0
    indices=list(range(rank,len(dataset),world))
    for x,y,kicks,sources in DataLoader(Subset(dataset,indices),batch_size=32):
        x,y=x.to(device),y.to(device); logits=model(x); loss=criterion(logits,y); probs=logits.softmax(1).cpu().tolist()
        loss_sum+=float(loss)*len(y);count+=len(y);records += [(k,int(t),p,s) for k,t,p,s in zip(kicks,y.cpu().tolist(),probs,sources)]
    records=gather(records);totals=torch.tensor([loss_sum,count],device=device,dtype=torch.float64)
    if dist.is_initialized():dist.all_reduce(totals)
    result=metrics(records);result["loss"]=float(totals[0]/max(totals[1],1));return result,records


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="combined_binary_6j_config.json");parser.add_argument("--dry-run",action="store_true");args=parser.parse_args()
    root=Path(__file__).resolve().parent;cfg=json.loads((root/args.config).read_text());seed_all(cfg["seed"])
    distributed="RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl");rank=dist.get_rank();world=dist.get_world_size();local=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(local);device=torch.device("cuda",local)
    else:rank=0;world=1;local=0;device=torch.device("cpu" if args.dry_run else "cuda:0")
    seed_all(cfg["seed"]+rank)
    old,old_manifest=build_old(root/cfg["old_csv"],cfg);new,new_manifest=build_new(root/cfg["new_csv"],cfg)
    for source,items in (("old",old),("new",new)):
        for item in items:item["source"]=source;item["group_id"]=source+":"+item["kick_id"]
    all_samples=old+new;label_by_group={}
    for item in all_samples:
        prior=label_by_group.setdefault(item["group_id"],item["label"])
        if prior!=item["label"]:raise ValueError("group label conflict: "+item["group_id"])
    groups=sorted(label_by_group);labels=[label_by_group[g] for g in groups]
    trainval,test=train_test_split(groups,test_size=cfg["test_fraction"],random_state=cfg["seed"],stratify=labels)
    trainval_labels=[label_by_group[g] for g in trainval]
    train,val=train_test_split(trainval,test_size=cfg["validation_fraction_of_trainval"],random_state=cfg["seed"],stratify=trainval_labels)
    group_split={g:name for name,gs in (("train",train),("validation",val),("test",test)) for g in gs}
    partitions={name:[s for s in all_samples if group_split[s["group_id"]]==name] for name in ("train","validation","test")}
    train_set,val_set,test_set=map(BinarySamples,(partitions["train"],partitions["validation"],partitions["test"]));out=root/cfg["output_dir"]
    if rank==0:
        out.mkdir(parents=True,exist_ok=True)
        def group_summary(gs):return {"kicks":len(gs),"labels":dict(Counter(label_by_group[g] for g in gs)),"sources":dict(Counter(g.split(':',1)[0] for g in gs))}
        info={"all":{"kicks":len(groups),"sequences":len(all_samples),"labels":dict(Counter(labels)),"sources":dict(Counter(g.split(':',1)[0] for g in groups))},"train":group_summary(train),"validation":group_summary(val),"test":group_summary(test),"old_csv_sha256":file_hash(root/cfg["old_csv"]),"new_csv_sha256":file_hash(root/cfg["new_csv"]),"config":cfg}
        (out/"run_info.json").write_text(json.dumps(info,indent=2));(out/"split.json").write_text(json.dumps({"train":sorted(train),"validation":sorted(val),"test":sorted(test)},indent=2));(out/"manifest.json").write_text(json.dumps({"old":old_manifest,"new":new_manifest},indent=2));print(json.dumps(info,indent=2),flush=True)
    model=PenaltySTGCN(cfg["dropout"],num_joints=6,edges=LOWER_BODY_EDGES,roots=(0,1),num_classes=2).to(device)
    if args.dry_run:
        x,_,_,_=next(iter(DataLoader(train_set,batch_size=2)));print("DRY RUN",tuple(x.shape),tuple(model(x.to(device)).shape),flush=True);return
    if world!=4:raise RuntimeError(f"Expected 4 GPUs, got {world}")
    model=DDP(model,device_ids=[local],output_device=local);sampler=DistributedSampler(train_set,num_replicas=world,rank=rank,shuffle=True,seed=cfg["seed"]);loader=DataLoader(train_set,batch_size=cfg["per_gpu_batch_size"],sampler=sampler,pin_memory=True)
    counts=Counter(label_by_group[g] for g in train);weights=torch.tensor([1/counts[n] for n in LABEL_NAMES],device=device);weights=weights/weights.sum()*2;criterion=nn.CrossEntropyLoss(weight=weights);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    best=(-1.,float("inf"));history=[]
    for epoch in range(1,cfg["epochs"]+1):
        sampler.set_epoch(epoch);model.train();loss_sum=0.;count=0
        for x,y,_,_ in loader:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True);logits=model(x);loss=criterion(logits,y);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip_norm"]);optimizer.step();loss_sum+=float(loss)*len(y);count+=len(y)
        totals=torch.tensor([loss_sum,count],device=device,dtype=torch.float64);dist.all_reduce(totals);validation,_=evaluate(model,val_set,device,rank,world,criterion)
        if rank==0:
            row={"epoch":epoch,"train_loss":float(totals[0]/totals[1]),"validation":validation};history.append(row);print(json.dumps(row),flush=True);candidate=(validation["macro_f1"],-validation["loss"])
            if candidate>(best[0],-best[1]):best=(validation["macro_f1"],validation["loss"]);torch.save({"epoch":epoch,"model":model.module.state_dict(),"validation":validation,"config":cfg},out/"best.pt")
            (out/"history.json").write_text(json.dumps(history,indent=2))
    dist.barrier();checkpoint=torch.load(out/"best.pt",map_location=device);model.module.load_state_dict(checkpoint["model"]);test_metrics,records=evaluate(model,test_set,device,rank,world,criterion)
    if rank==0:
        final={"best_epoch":checkpoint["epoch"],"validation":checkpoint["validation"],"test":test_metrics};(out/"metrics.json").write_text(json.dumps(final,indent=2))
        rows=[]
        for kick,target,prob,source in sorted(records):rows.append({"group_id":kick,"source":source,"actual":LABEL_NAMES[target],"predicted":LABEL_NAMES[int(np.argmax(prob))],"prob_L":prob[0],"prob_R":prob[1]})
        import csv
        with (out/"test_predictions.csv").open("w",newline="") as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print("FINAL "+json.dumps(final),flush=True)
    dist.destroy_process_group()

if __name__=="__main__":main()
