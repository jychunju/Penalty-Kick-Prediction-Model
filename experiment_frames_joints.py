#!/usr/bin/env python3
"""Frame x joint ablation on the existing C-camera development set only.

Keep the original pelvis/torso normalization, kicks, folds, stopping sets,
seeds, hidden widths and training settings. Six joints remove shoulders ONLY
from model inputs, never from the already-computed normalization. No angle
features are used. Reuse 60 verified equivalent 8-joint fits; train 180 new fits.
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
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from compare_8j20_lr_cv import summarize_runs, write_predictions
from model import STGCNBlock, make_adjacency
from train_8j20_angles_lr import ROOT, LABELS, evaluate, hash_file, save_json, score

JOINTS = {8: (5, 6, 11, 12, 13, 14, 15, 16), 6: (11, 12, 13, 14, 15, 16)}
ANATOMICAL_EDGES = ((5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))


def uniform_indices(count):
    if count not in (20,10,5,3):
        raise ValueError("Supported frame counts: 20, 10, 5, 3")
    return np.rint(np.linspace(0,19,count)).astype(int).tolist()


def coordinate_indices(joint_count):
    return [2*JOINTS[8].index(j)+axis for j in JOINTS[joint_count] for axis in (0,1)]


def joint_graph(joint_count):
    joints=JOINTS[joint_count]
    edges=tuple((joints.index(a),joints.index(b)) for a,b in ANATOMICAL_EDGES if a in joints and b in joints)
    roots=(joints.index(11),joints.index(12))
    return edges,roots


def select_inputs(x,frames,joint_count):
    # Pure subsetting: coordinates retain their existing units and normalization.
    return x[:,uniform_indices(frames),:][:,:,coordinate_indices(joint_count)]


class FlexiblePoseClassifier(nn.Module):
    """Same small model widths; adapt only input channels and graph topology.

    Initialization order and 8-joint operations match SmallPoseClassifier exactly,
    enabling validated reuse. Frame count never changes trainable parameters.
    """
    def __init__(self,architecture,joint_count,dropout=0.3):
        super().__init__()
        if architecture not in ("stgcn","cnn") or joint_count not in JOINTS:
            raise ValueError("Unsupported architecture/joint count")
        self.architecture=architecture;self.joint_count=joint_count;self.feature_count=2*joint_count
        self.coordinate_norm=nn.BatchNorm1d(self.feature_count)
        if architecture=="stgcn":
            edges,roots=joint_graph(joint_count)
            self.register_buffer("adjacency",make_adjacency(joint_count,edges,roots))
            self.blocks=nn.ModuleList([STGCNBlock(2,16,dropout=dropout),STGCNBlock(16,16,dropout=dropout),
                                       STGCNBlock(16,32,stride=2,dropout=dropout)])
        else:
            self.coordinate_cnn=nn.Sequential(
                nn.Conv1d(self.feature_count,16,5,padding=2,bias=False),nn.BatchNorm1d(16),nn.ReLU(),nn.Dropout(dropout),
                nn.Conv1d(16,32,5,padding=2,bias=False),nn.BatchNorm1d(32),nn.ReLU(),nn.Dropout(dropout),nn.AdaptiveAvgPool1d(1))
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(32,2))

    def forward(self,x):
        if x.ndim!=3 or x.shape[2]!=self.feature_count or x.shape[1]<2:
            raise ValueError("Expected (batch, frames>=2, 2*joint_count)")
        n,t,_=x.shape
        coordinates=self.coordinate_norm(x.transpose(1,2))
        if self.architecture=="stgcn":
            coordinates=coordinates.reshape(n,self.joint_count,2,t).permute(0,2,3,1)
            for block in self.blocks:
                coordinates=block(coordinates,self.adjacency)
            pooled=coordinates.mean(dim=(2,3))
        else:
            pooled=self.coordinate_cnn(coordinates).squeeze(-1)
        return self.head(pooled)


def conditions(config):
    return {f"{a}_f{t}_j{j}":{"architecture":a,"frames":t,"joints":j,
                              "frame_indices":uniform_indices(t),"coordinate_indices":coordinate_indices(j)}
            for a in config["architectures"] for t in config["frame_counts"] for j in config["joint_counts"]}


def reusable_condition(spec):
    if spec["joints"]==8 and spec["frames"] in (20,10):
        return f"{spec['architecture']}{spec['frames']}_xy"
    return None


def prepare(config):
    out,reference=ROOT/config["output_dir"],ROOT/config["reference_run"]
    if out.exists():
        raise FileExistsError("Choose a new output_dir; never overwrite previous artifacts")
    old=json.loads((reference/"run_info.json").read_text())
    assert json.loads((reference/"artifact_validation.json").read_text())["status"]=="PASS"
    for path,digest in old["protected_file_hashes"].items():
        assert hash_file(Path(path))==digest,path
    for name,digest in old["source_code_hashes"].items():
        assert hash_file(ROOT/name)==digest,name
    assert config["architectures"]==["stgcn","cnn"] and config["frame_counts"]==[20,10,5,3] and config["joint_counts"]==[8,6]
    with np.load(reference/"development_only.npz") as d:
        x=d["X"][:,:,:16].copy();y=d["y"].copy();groups=d["group_id"].copy();sources=d["dataset"].copy();kicks=d["kick_id"].copy()
    assert x.shape==(141,20,16) and np.isfinite(x).all()
    assert not(set(groups.tolist())&set(old["protected_test_groups"]))
    folds=json.loads((reference/"folds.json").read_text());specs=conditions(config)
    tasks=[];new_id=0
    for seed in old["config"]["training_seeds"]:
        for fold in folds:
            for name,spec in specs.items():
                reused=reusable_condition(spec)
                task={"condition":name,"seed":seed,"fold":fold["fold"],"reused_reference":reused is not None,
                      "task_key":f"{name}/seed_{seed}/fold_{fold['fold']}"}
                if reused:
                    directory=reference/reused/f"seed_{seed}"/f"fold_{fold['fold']}"
                    assert (directory/"best.pt").exists()
                    task["artifact_directory"]=str(directory)
                else:
                    task["worker_id"]=new_id%len(config["devices"]);new_id+=1
                    task["artifact_directory"]=str(out/task["task_key"])
                tasks.append(task)
    protected=dict(old["protected_file_hashes"])
    protected.update({str(p):hash_file(p) for p in reference.rglob("*") if p.is_file()})
    train_config=old["config"]
    info={"config":config,"training_config":train_config,"conditions":specs,"development":old["development"],
          "development_shape":list(x.shape),"protected_test_groups":old["protected_test_groups"],
          "protected_file_hashes":protected,"reference_source_code_hashes":old["source_code_hashes"],
          "source_code_hashes":{n:hash_file(ROOT/n) for n in ("experiment_frames_joints.py","model.py","compare_8j20_lr_cv.py","train_8j20_angles_lr.py")},
          "total_scored_fits":len(tasks),"new_fits":new_id,"reused_fits":len(tasks)-new_id,"test_evaluations":0,
          "notes":["Same C-camera 141 kicks, original source×label folds, inner stopping sets and seeds.",
                   "No interpolation, renormalization, rotation, label modification or camera filtering changes.",
                   "Six joints remove shoulders from inputs only; the original shoulder-derived torso scale remains fixed.",
                   "All temporal selections are uniform sampled indices including start/contact, not fixed seconds.",
                   "XY-only throughout; no angle features. Hidden widths and training hyperparameters are held fixed.",
                   "Frame count changes effective temporal receptive field and padding fraction despite fixed parameters.",
                   "Train–validation gap is a diagnostic, not proof of overfitting; fold/seed SD is not a confidence interval."]}
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/"run_info.json",info);save_json(out/"folds.json",folds);save_json(out/"tasks.json",tasks)
    np.savez_compressed(out/"development_only.npz",X=x,y=y,group_id=groups,dataset=sources,kick_id=kicks)
    return info


def load_data(out):
    with np.load(out/"development_only.npz") as d:
        x,y=torch.from_numpy(d["X"].copy()),torch.from_numpy(d["y"].copy())
        samples=[{"group_id":g,"dataset":s,"kick_id":k,"label":LABELS[int(v)]}
                 for g,s,k,v in zip(d["group_id"].tolist(),d["dataset"].tolist(),d["kick_id"].tolist(),d["y"].tolist())]
    return x,y,samples


def fit_task(info,task,fold,x,y,samples,device):
    config=info["training_config"];spec=info["conditions"][task["condition"]]
    fit_seed=task["seed"]+1000*task["fold"]
    random.seed(fit_seed);np.random.seed(fit_seed);torch.manual_seed(fit_seed);torch.cuda.manual_seed_all(fit_seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    model=FlexiblePoseClassifier(spec["architecture"],spec["joints"],config["dropout"]).to(device)
    x=select_inputs(x,spec["frames"],spec["joints"])
    train,stop,outer=[fold["indices"][k] for k in ("inner_train","early_stop","validation")]
    counts=torch.bincount(y[train],minlength=2).float();weights=counts.sum()/(2*counts)
    criterion=nn.CrossEntropyLoss(weight=weights.to(device))
    loader=DataLoader(TensorDataset(x[train],y[train]),batch_size=config["batch_size"],shuffle=True,
                      generator=torch.Generator().manual_seed(fit_seed),num_workers=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=8,min_lr=1e-5)
    directory=Path(task["artifact_directory"]);directory.mkdir(parents=True,exist_ok=False)
    best,best_epoch,history=(-1.,-float("inf")),0,[];start=time.perf_counter()
    for epoch in range(1,config["epochs"]+1):
        model.train();total_loss,count=0.,0
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device);optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(xb),yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),config["gradient_clip_norm"]);optimizer.step()
            total_loss+=float(loss.detach())*len(yb);count+=len(yb)
        stopping,_=evaluate(model,x,y,stop,samples,device)
        candidate=(stopping["macro_f1"],-stopping["loss"])
        history.append({"epoch":epoch,"train_loss":total_loss/count,"early_stop":stopping,"lr":optimizer.param_groups[0]["lr"]})
        if candidate>best:
            best,best_epoch=candidate,epoch
            torch.save({"model":model.state_dict(),"epoch":epoch,"early_stop":stopping,"condition":task["condition"],
                        "architecture":spec["architecture"],"joint_ids":list(JOINTS[spec["joints"]]),
                        "feature_columns":[f"kp_{j}_{axis}n" for j in JOINTS[spec["joints"]] for axis in "xy"],
                        "frame_indices":spec["frame_indices"],"training_indices":train,"seed":task["seed"],"fit_seed":fit_seed,
                        "fold":task["fold"],"config":config,"class_weights":weights.tolist()},directory/"best.pt")
        scheduler.step(stopping["loss"])
        if epoch-best_epoch>=config["early_stopping_patience"]:
            break
    checkpoint=torch.load(directory/"best.pt",map_location=device);model.load_state_dict(checkpoint["model"])
    validation,probabilities=evaluate(model,x,y,outer,samples,device)
    training,_=evaluate(model,x,y,train,samples,device)
    baseline=np.zeros((len(outer),2));baseline[:,int(counts.argmax())]=1
    baseline=score(y[outer].numpy(),baseline,np.array([samples[i]["dataset"] for i in outer]))
    metrics={**task,**spec,"fit_seed":fit_seed,"parameters":sum(p.numel() for p in model.parameters()),
             "training_kicks":len(train),"best_epoch":best_epoch,"epochs_completed":len(history),
             "train":training,"early_stop":checkpoint["early_stop"],"validation":validation,"majority_baseline":baseline,
             "elapsed_seconds":time.perf_counter()-start,"outer_validation_evaluations":1,"test_evaluations":0}
    save_json(directory/"metrics.json",metrics);save_json(directory/"history.json",history)
    write_predictions(directory/"validation_predictions.csv",samples,outer,probabilities,task["seed"],task["fold"])
    return metrics


def worker(config,worker_id):
    out=ROOT/config["output_dir"];info=json.loads((out/"run_info.json").read_text())
    tasks=json.loads((out/"tasks.json").read_text());folds=json.loads((out/"folds.json").read_text())
    torch.set_num_threads(info["training_config"]["torch_threads_per_worker"])
    device=torch.device(config["devices"][worker_id])
    if device.type=="cuda":torch.cuda.set_device(device)
    x,y,samples=load_data(out);assigned=[t for t in tasks if not t["reused_reference"] and t["worker_id"]==worker_id]
    for done,task in enumerate(assigned,1):
        fold=next(f for f in folds if f["fold"]==task["fold"])
        r=fit_task(info,task,fold,x,y,samples,device)
        save_json(out/f"worker_{worker_id}_progress.json",{"completed_fits":done,"total_fits":len(assigned),"last_task":task["task_key"]})
        print(json.dumps({"task":task["task_key"],"accuracy":r["validation"]["accuracy"],"train_accuracy":r["train"]["accuracy"],"best_epoch":r["best_epoch"]}),flush=True)


def delta_summary(result,reference):
    delta=np.array([a["validation"]["accuracy"]-b["validation"]["accuracy"] for a,b in zip(result["runs"],reference["runs"])])
    gap=np.array([(a["train"]["accuracy"]-a["validation"]["accuracy"])-(b["train"]["accuracy"]-b["validation"]["accuracy"]) for a,b in zip(result["runs"],reference["runs"])])
    return {"accuracy_delta_mean":float(delta.mean()),"gap_delta_mean":float(gap.mean()),
            "wins":int((delta>0).sum()),"ties":int((delta==0).sum()),"losses":int((delta<0).sum())}


def aggregate(config):
    out=ROOT/config["output_dir"];info=json.loads((out/"run_info.json").read_text())
    folds=json.loads((out/"folds.json").read_text());tasks=json.loads((out/"tasks.json").read_text())
    _,y,samples=load_data(out);sources=np.array([s["dataset"] for s in samples]);results={}
    for name,spec in info["conditions"].items():
        runs=[];oof_scores={};(out/name).mkdir(exist_ok=True)
        for seed in info["training_config"]["training_seeds"]:
            oof=np.full((len(samples),2),np.nan,dtype=np.float32)
            for fold in folds:
                task=next(t for t in tasks if (t["condition"],t["seed"],t["fold"])==(name,seed,fold["fold"]))
                directory=Path(task["artifact_directory"]);r=json.loads((directory/"metrics.json").read_text())
                r={**r,**task,**spec};runs.append(r)
                with (directory/"validation_predictions.csv").open() as f:rows=list(csv.DictReader(f))
                assert [row["group_id"] for row in rows]==fold["group_ids"]["validation"]
                oof[fold["indices"]["validation"]]=[[float(row["prob_L"]),float(row["prob_R"])] for row in rows]
            assert np.isfinite(oof).all()
            oof_scores[str(seed)]=score(y.numpy(),oof,sources)
            write_predictions(out/name/f"oof_seed_{seed}.csv",samples,list(range(len(samples))),oof,seed,"see folds.json")
        gaps=[r["train"]["accuracy"]-r["validation"]["accuracy"] for r in runs]
        f1gaps=[r["train"]["macro_f1"]-r["validation"]["macro_f1"] for r in runs]
        results[name]={"spec":spec,"parameters":runs[0]["parameters"],"cv":summarize_runs(runs),
                       "accuracy_gap":{"mean":float(np.mean(gaps)),"std":float(np.std(gaps,ddof=1))},
                       "macro_f1_gap":{"mean":float(np.mean(f1gaps)),"std":float(np.std(f1gaps,ddof=1))},
                       "per_seed_oof":oof_scores,"runs":runs}
    for name,r in results.items():
        a,t,j=[r["spec"][k] for k in ("architecture","frames","joints")]
        r["paired_vs_20_frames_same_joints"]=delta_summary(r,results[f"{a}_f20_j{j}"])
        r["paired_vs_8_joints_same_frames"]=delta_summary(r,results[f"{a}_f{t}_j8"])
    ref=results["stgcn_f20_j8"];baseline=float(np.mean([r["majority_baseline"]["accuracy"] for r in ref["runs"]]))
    comparison={"results":results,"selected_by_cv_macro_f1":max(results,key=lambda k:results[k]["cv"]["macro_f1"]["mean"]),
                "new_fits":info["new_fits"],"reused_fits":info["reused_fits"],"total_scored_fits":len(tasks),
                "majority_accuracy_mean":baseline,"test_evaluations":0}
    save_json(out/"comparison.json",comparison)
    with (out/"comparison.csv").open("x",newline="") as f:
        columns=["condition","architecture","frames","joints","parameters","train_accuracy","validation_accuracy","validation_accuracy_std","macro_f1","train_validation_gap"]
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for name,r in results.items():
            w.writerow({"condition":name,**{k:r["spec"][k] for k in ("architecture","frames","joints")},"parameters":r["parameters"],
                        "train_accuracy":r["cv"]["training_accuracy"]["mean"],"validation_accuracy":r["cv"]["accuracy"]["mean"],
                        "validation_accuracy_std":r["cv"]["accuracy"]["std"],"macro_f1":r["cv"]["macro_f1"]["mean"],"train_validation_gap":r["accuracy_gap"]["mean"]})
    lines=["# Frames × joints ablation (C camera, L/R prediction)","",
           "141 development kicks (CLT 18, EPL 21, Women 102); same 5 folds × 3 seeds and inner stopping sets as the C-camera experiment.",
           "Existing Test is excluded. Original pelvis/torso normalization is fixed; 6 joints remove shoulders from inputs only.",
           "XY-only, same hidden widths/dropout/optimizer. Uniformly select existing sample indices including first/contact.","",
           "| Model | Frames | Joints | Parameters | Train accuracy | Validation mean ± SD | Macro F1 | Train−validation gap |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,r in results.items():
        cv=r["cv"];s=r["spec"]
        lines.append(f"| {s['architecture']} | {s['frames']} | {s['joints']} | {r['parameters']} | {100*cv['training_accuracy']['mean']:.2f}% | {100*cv['accuracy']['mean']:.2f}% ± {100*cv['accuracy']['std']:.2f} pp | {cv['macro_f1']['mean']:.4f} | {100*r['accuracy_gap']['mean']:.2f} pp |")
    lines += ["",f"Majority baseline accuracy: {100*baseline:.2f}%. Selected by CV macro F1: {comparison['selected_by_cv_macro_f1']}.",
              "",f"{info['new_fits']} new fits; {info['reused_fits']} verified existing fits; {len(tasks)} scored fits.",
              "", "## Interpretation", "", "A larger train−validation gap is a diagnostic, not proof of overfitting.",
              "Both scores declining may reflect information loss, optimization, or altered temporal receptive fields.",
              "At 3/5 frames, temporal kernels remain width 5; padding/receptive-field changes are part of this fixed-architecture ablation.",
              "Frame reduction does not reduce kick count. Joint reduction slightly reduces ST-GCN BN parameters and reduces CNN input weights.",
              "Folds/seeds share data; SD is descriptive, not a confidence interval. No independent Test evaluation or final all-development fit.",
              "This is kick-level, not player/match-level generalization. All source data/prior artifacts remain unchanged.",
              "", "![Train and validation accuracy](train_validation_curves.png)"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    plot_results(out,results,baseline)
    return comparison


def plot_results(out,results,baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharey=True)
    for ax,a in zip(axes,("stgcn","cnn")):
        for j,color in ((8,"tab:blue"),(6,"tab:orange")):
            frames=[3,5,10,20];rs=[results[f"{a}_f{t}_j{j}"] for t in frames]
            ax.plot(frames,[100*r["cv"]["training_accuracy"]["mean"] for r in rs],"--o",color=color,label=f"{j} joints - train")
            ax.errorbar(frames,[100*r["cv"]["accuracy"]["mean"] for r in rs],yerr=[100*r["cv"]["accuracy"]["std"] for r in rs],
                        fmt="-o",color=color,capsize=3,label=f"{j} joints - validation")
        ax.axhline(100*baseline,color="gray",linestyle=":",label="Majority baseline")
        ax.set(title=a.upper(),xlabel="Selected frames",xticks=[3,5,10,20],ylim=(30,100));ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("C camera | 141 kicks | 5 folds x 3 seeds | Validation bars: descriptive SD")
    fig.tight_layout();fig.savefig(out/"train_validation_curves.png",dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=ROOT/"experiment_frames_joints_config.json")
    p.add_argument("--worker",type=int);args=p.parse_args();config=json.loads(args.config.read_text());out=ROOT/config["output_dir"]
    if args.worker is not None:
        worker(config,args.worker);return
    info=prepare(config);print(json.dumps({"new_fits":info["new_fits"],"reused_fits":info["reused_fits"],"development":info["development"]}),flush=True)
    processes=[]
    try:
        for i in range(len(config["devices"])):
            log=(out/f"worker_{i}.log").open("x")
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--config",str(args.config.resolve()),"--worker",str(i)],stdout=log,stderr=subprocess.STDOUT)
            processes.append((process,log))
        while any(p.poll() is None for p,_ in processes):
            if any(p.poll() not in (None,0) for p,_ in processes):raise RuntimeError("Worker failed; inspect logs")
            time.sleep(5)
        if any(p.returncode for p,_ in processes):raise RuntimeError("Worker failed")
        aggregate(config);print("COMPLETED: "+str(out/"REPORT.md"),flush=True)
    finally:
        for process,log in processes:
            if process.poll() is None:process.terminate();process.wait()
            log.close()


if __name__=="__main__":main()
