#!/usr/bin/env python3
"""Four ST-GCN conditions using raw consecutive frames just before contact.

Keep C-camera development kicks, folds, RNG seeds, normalization and model
hyperparameters fixed. Use ORIGINAL full-sequence rows, not the tail of 20
uniform samples. By default exclude the contact frame. No Test evaluation.
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

from compare_8j20_lr_cv import summarize_runs,write_predictions
from experiment_frames_joints import FlexiblePoseClassifier,JOINTS,coordinate_indices,delta_summary
from extract_uniform_20frames import assess_candidate,candidate_sequences,frame_number,is_touch
from train_8j20_angles_lr import ROOT,LABELS,evaluate,hash_file,save_json,score

XY_COLUMNS=[f"kp_{j}_{a}n" for j in JOINTS[8] for a in "xy"]


def desired_frames(contact,count,include_contact=False):
    if count not in (5,3) or int(contact)!=contact:
        raise ValueError("Need an integer contact frame and 5 or 3 frames")
    end=int(contact)+(1 if include_contact else 0)
    return list(range(end-count,end))


def frame_rows(candidate,rows,contact):
    """Match old sampler's duplicate policy: first row, explicit contact preferred."""
    result={}
    for i in sorted(candidate["indices"]):
        t=frame_number(rows[i])
        if t is None:raise ValueError("Invalid frame in selected sequence")
        t=int(t)
        if t not in result or (t==contact and is_touch(rows[i]) and not is_touch(rows[result[t]])):
            result[t]=i
    return result


def extract_windows(rows,records,groups,reference_x,include_contact):
    """Trace the SAME camera/track sequence via prior sampler's exact CSV lines."""
    candidates={(c["key"],min(c["indices"])+2):c for c in candidate_sequences(rows)}
    records={f"{r['dataset']}:{r['kick_id']}":r for r in records}
    x5=[];audits=[];kept_rows={5:[],3:[]}
    for position,group in enumerate(groups):
        r=records[group];key=(r["dataset"],r["kick_id"],r["camera_direction"])
        assert r["dataset"]=="CLT" or r["camera_direction"]=="C"
        candidate=candidates[(key,r["first_source_csv_line"])]
        _,recomputed=assess_candidate(candidate,rows)
        assert recomputed["sequence_id"]==r["sequence_id"]
        assert recomputed["selected_source_csv_lines"]==r["selected_source_csv_lines"]
        prior=np.array([[float(rows[line-2][column]) for column in XY_COLUMNS] for line in r["selected_source_csv_lines"]],dtype=np.float32)
        np.testing.assert_array_equal(prior,reference_x[position])
        contact=int(r["last_touch_frame"]);byframe=frame_rows(candidate,rows,contact)
        assert is_touch(rows[byframe[contact]])
        frames=desired_frames(contact,5,include_contact)
        missing=[f for f in frames if f not in byframe]
        if missing:raise ValueError(f"Missing consecutive precontact frames in {group}: {missing}; no silent substitution")
        indices=[byframe[f] for f in frames]
        features=np.array([[float(rows[i][column]) for column in XY_COLUMNS] for i in indices],dtype=np.float32)
        assert np.isfinite(features).all()
        x5.append(features)
        audits.append({"group_id":group,"dataset":r["dataset"],"kick_id":r["kick_id"],"sequence_id":r["sequence_id"],
                       "original_camera_direction":r["camera_direction"],"effective_camera_direction":"C", "contact_frame":contact,
                       "frames_5":frames,"frames_3":frames[-3:],"source_csv_lines_5":[i+2 for i in indices],
                       "source_csv_lines_3":[i+2 for i in indices[-3:]]})
        for count in (5,3):
            for step,i in enumerate(indices[-count:]):
                kept_rows[count].append({**rows[i],"sequence_id":r["sequence_id"],"precontact_idx":step,
                                         "contact_frame":contact,"frame_offset_from_contact":int(frame_number(rows[i]))-contact,
                                         "effective_camera_direction":"C"})
    x5=np.stack(x5);return x5,x5[:,-3:,:].copy(),audits,kept_rows


def prepare(config):
    reference,out=ROOT/config["reference_run"],ROOT/config["output_dir"]
    if out.exists():raise FileExistsError("Choose a new output directory; outputs are never overwritten")
    old=json.loads((reference/"run_info.json").read_text())
    assert json.loads((reference/"artifact_validation.json").read_text())["status"]=="PASS"
    for path,digest in old["protected_file_hashes"].items():assert hash_file(Path(path))==digest,path
    for name,digest in {**old["reference_source_code_hashes"],**old["source_code_hashes"]}.items():assert hash_file(ROOT/name)==digest,name
    report_path,source=ROOT/config["sampling_report"],ROOT/config["source_csv"]
    report=json.loads(report_path.read_text());assert hash_file(source)==report["input_sha256"]
    with source.open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    with np.load(reference/"development_only.npz") as d:
        ref_x=d["X"].copy();groups=d["group_id"].tolist();y=d["y"].copy();sources=d["dataset"].copy();kicks=d["kick_id"].copy()
    assert len(groups)==141 and not(set(groups)&set(old["protected_test_groups"]))
    x5,x3,audits,exported=extract_windows(rows,report["selected_sequences"],groups,ref_x,config["include_contact"])
    assert x5.shape==(141,5,16) and x3.shape==(141,3,16)
    folds=json.loads((reference/"folds.json").read_text())
    specs={f"stgcn_pre{t}_j{j}":{"frames":t,"joints":j,"architecture":"stgcn","coordinate_indices":coordinate_indices(j)}
           for t in config["frame_counts"] for j in config["joint_counts"]}
    assert len(specs)==4 and config["frame_counts"]==[5,3] and config["joint_counts"]==[6,8]
    tasks=[]
    for seed in old["training_config"]["training_seeds"]:
        for fold in folds:
            for name in specs:
                key=f"{name}/seed_{seed}/fold_{fold['fold']}"
                tasks.append({"condition":name,"seed":seed,"fold":fold["fold"],"worker_id":len(tasks)%len(config["devices"]),
                              "task_key":key,"artifact_directory":str(out/key)})
    protected=dict(old["protected_file_hashes"])
    protected.update({str(p):hash_file(p) for p in reference.rglob("*") if p.is_file()})
    test_report=ROOT/"runs/main_data_camera_c_stgcn_f3_j6_test"
    protected.update({str(p):hash_file(p) for p in test_report.rglob("*") if p.is_file()})
    protected.update({str(source):hash_file(source),str(report_path):hash_file(report_path)})
    info={"config":config,"training_config":old["training_config"],"conditions":specs,"development":old["development"],
          "protected_test_groups":old["protected_test_groups"],"protected_file_hashes":protected,
          "source_code_hashes":{n:hash_file(ROOT/n) for n in ("experiment_precontact.py","experiment_frames_joints.py","extract_uniform_20frames.py","model.py","train_8j20_angles_lr.py")},
          "total_fits":len(tasks),"test_evaluations":0,"prior_test_already_exposed":test_report.exists(),
          "notes":["Four ST-GCN settings only; original consecutive precontact frames, not last 5/3 of 20 sampled poses.",
                   "Same C-camera 141 kicks, source×label folds, inner stopping sets and three RNG seeds as the reference.",
                   "Existing normalized xy retained; no rotation, reflection, interpolation or torso-scale recomputation.",
                   "XY-only: 6 joints are hips/knees/ankles; 8 joints additionally include shoulders.",
                   "Test has already been reported for the previous selected model; this exploratory comparison does not reevaluate it."]}
    out.mkdir(parents=True,exist_ok=False);save_json(out/"run_info.json",info);save_json(out/"folds.json",folds)
    save_json(out/"tasks.json",tasks);save_json(out/"extraction_audit.json",audits)
    np.savez_compressed(out/"development_only.npz",X5=x5,X3=x3,y=y,group_id=np.array(groups),dataset=sources,kick_id=kicks,
                        source_lines_5=np.array([a["source_csv_lines_5"] for a in audits]),frames_5=np.array([a["frames_5"] for a in audits]),
                        contact_frames=np.array([a["contact_frame"] for a in audits]))
    for count in (5,3):
        with (out/f"precontact_{count}frames.csv").open("x",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=fields+["sequence_id","precontact_idx","contact_frame","frame_offset_from_contact","effective_camera_direction"])
            writer.writeheader();writer.writerows(exported[count])
    return info


def load_data(out):
    with np.load(out/"development_only.npz") as d:
        windows={t:torch.from_numpy(d[f"X{t}"].copy()) for t in (5,3)};y=torch.from_numpy(d["y"].copy())
        samples=[{"group_id":g,"dataset":s,"kick_id":k,"label":LABELS[int(v)]}
                 for g,s,k,v in zip(d["group_id"].tolist(),d["dataset"].tolist(),d["kick_id"].tolist(),d["y"].tolist())]
    return windows,y,samples


def fit_task(info,task,fold,windows,y,samples,device):
    config=info["training_config"];spec=info["conditions"][task["condition"]];fit_seed=task["seed"]+1000*task["fold"]
    random.seed(fit_seed);np.random.seed(fit_seed);torch.manual_seed(fit_seed);torch.cuda.manual_seed_all(fit_seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    model=FlexiblePoseClassifier("stgcn",spec["joints"],config["dropout"]).to(device)
    x=windows[spec["frames"]][:,:,spec["coordinate_indices"]]
    train,stop,outer=[fold["indices"][k] for k in ("inner_train","early_stop","validation")]
    counts=torch.bincount(y[train],minlength=2).float();weights=counts.sum()/(2*counts)
    criterion=nn.CrossEntropyLoss(weight=weights.to(device))
    loader=DataLoader(TensorDataset(x[train],y[train]),batch_size=config["batch_size"],shuffle=True,
                      generator=torch.Generator().manual_seed(fit_seed),num_workers=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.5,patience=8,min_lr=1e-5)
    directory=Path(task["artifact_directory"]);directory.mkdir(parents=True,exist_ok=False)
    best,best_epoch,history=(-1.,-float("inf")),0,[];started=time.perf_counter()
    for epoch in range(1,config["epochs"]+1):
        model.train();total_loss,count=0.,0
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device);optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(xb),yb)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),config["gradient_clip_norm"]);optimizer.step()
            total_loss+=float(loss.detach())*len(yb);count+=len(yb)
        stopping,_=evaluate(model,x,y,stop,samples,device);candidate=(stopping["macro_f1"],-stopping["loss"])
        history.append({"epoch":epoch,"train_loss":total_loss/count,"early_stop":stopping,"lr":optimizer.param_groups[0]["lr"]})
        if candidate>best:
            best,best_epoch=candidate,epoch
            torch.save({"model":model.state_dict(),"epoch":epoch,"early_stop":stopping,"condition":task["condition"],
                        "architecture":"stgcn","frames":spec["frames"],"joint_ids":list(JOINTS[spec["joints"]]),
                        "feature_columns":[f"kp_{j}_{a}n" for j in JOINTS[spec["joints"]] for a in "xy"],
                        "include_contact":info["config"]["include_contact"],"training_indices":train,"seed":task["seed"],
                        "fit_seed":fit_seed,"fold":task["fold"],"config":config,"class_weights":weights.tolist()},directory/"best.pt")
        scheduler.step(stopping["loss"])
        if epoch-best_epoch>=config["early_stopping_patience"]:break
    ckpt=torch.load(directory/"best.pt",map_location=device);model.load_state_dict(ckpt["model"])
    validation,probabilities=evaluate(model,x,y,outer,samples,device);training,_=evaluate(model,x,y,train,samples,device)
    baseline=np.zeros((len(outer),2));baseline[:,int(counts.argmax())]=1
    baseline=score(y[outer].numpy(),baseline,np.array([samples[i]["dataset"] for i in outer]))
    metrics={**task,**spec,"fit_seed":fit_seed,"parameters":sum(p.numel() for p in model.parameters()),"training_kicks":len(train),
             "best_epoch":best_epoch,"epochs_completed":len(history),"elapsed_seconds":time.perf_counter()-started,
             "train":training,"early_stop":ckpt["early_stop"],"validation":validation,"majority_baseline":baseline,
             "outer_validation_evaluations":1,"test_evaluations":0}
    save_json(directory/"metrics.json",metrics);save_json(directory/"history.json",history)
    write_predictions(directory/"validation_predictions.csv",samples,outer,probabilities,task["seed"],task["fold"])
    return metrics


def worker(config,worker_id):
    out=ROOT/config["output_dir"];info=json.loads((out/"run_info.json").read_text());tasks=json.loads((out/"tasks.json").read_text())
    folds=json.loads((out/"folds.json").read_text());torch.set_num_threads(info["training_config"]["torch_threads_per_worker"])
    device=torch.device(config["devices"][worker_id])
    if device.type=="cuda":torch.cuda.set_device(device)
    windows,y,samples=load_data(out);assigned=[t for t in tasks if t["worker_id"]==worker_id]
    for done,task in enumerate(assigned,1):
        r=fit_task(info,task,next(f for f in folds if f["fold"]==task["fold"]),windows,y,samples,device)
        save_json(out/f"worker_{worker_id}_progress.json",{"completed_fits":done,"total_fits":len(assigned),"last_task":task["task_key"]})
        print(json.dumps({"task":task["task_key"],"accuracy":r["validation"]["accuracy"],"train_accuracy":r["train"]["accuracy"]}),flush=True)


def aggregate(config):
    out=ROOT/config["output_dir"];info=json.loads((out/"run_info.json").read_text());folds=json.loads((out/"folds.json").read_text())
    tasks=json.loads((out/"tasks.json").read_text());_,y,samples=load_data(out);sources=np.array([s["dataset"] for s in samples]);results={}
    prior=json.loads((ROOT/config["reference_run"]/"comparison.json").read_text())["results"]
    for name,spec in info["conditions"].items():
        runs=[];oof_scores={};(out/name).mkdir(exist_ok=True)
        for seed in info["training_config"]["training_seeds"]:
            oof=np.full((len(samples),2),np.nan,dtype=np.float32)
            for fold in folds:
                task=next(t for t in tasks if (t["condition"],t["seed"],t["fold"])==(name,seed,fold["fold"]))
                directory=Path(task["artifact_directory"]);runs.append(json.loads((directory/"metrics.json").read_text()))
                with (directory/"validation_predictions.csv").open() as f:rows=list(csv.DictReader(f))
                assert [r["group_id"] for r in rows]==fold["group_ids"]["validation"]
                oof[fold["indices"]["validation"]]=[[float(r["prob_L"]),float(r["prob_R"])] for r in rows]
            assert np.isfinite(oof).all();oof_scores[str(seed)]=score(y.numpy(),oof,sources)
            write_predictions(out/name/f"oof_seed_{seed}.csv",samples,list(range(len(samples))),oof,seed,"see folds.json")
        gaps=[r["train"]["accuracy"]-r["validation"]["accuracy"] for r in runs]
        r={"spec":spec,"cv":summarize_runs(runs),"parameters":runs[0]["parameters"],"per_seed_oof":oof_scores,"runs":runs,
           "accuracy_gap":{"mean":float(np.mean(gaps)),"std":float(np.std(gaps,ddof=1))}}
        baseline=prior[f"stgcn_f{spec['frames']}_j{spec['joints']}"]
        r["paired_vs_uniform_same_frame_joint_count"]=delta_summary(r,baseline)
        r["uniform_reference_accuracy"]=baseline["cv"]["accuracy"]
        r["by_dataset"]={s:{"accuracy_mean":float(np.mean([v["by_dataset"][s]["accuracy"] for v in oof_scores.values()])),
                                "macro_f1_mean":float(np.mean([v["by_dataset"][s]["macro_f1"] for v in oof_scores.values()]))} for s in sorted(set(sources))}
        results[name]=r
    baseline=float(np.mean([r["majority_baseline"]["accuracy"] for r in next(iter(results.values()))["runs"]]))
    comparison={"results":results,"selected_by_cv_macro_f1":max(results,key=lambda k:results[k]["cv"]["macro_f1"]["mean"]),
                "total_fits":len(tasks),"majority_accuracy_mean":baseline,"test_evaluations":0}
    save_json(out/"comparison.json",comparison)
    with (out/"comparison.csv").open("x",newline="") as f:
        cols=["condition","frames","joints","train_accuracy","validation_accuracy","validation_accuracy_std","macro_f1","gap","uniform_reference_accuracy","delta_vs_uniform"]
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for name,r in results.items():w.writerow({"condition":name,"frames":r["spec"]["frames"],"joints":r["spec"]["joints"],
            "train_accuracy":r["cv"]["training_accuracy"]["mean"],"validation_accuracy":r["cv"]["accuracy"]["mean"],
            "validation_accuracy_std":r["cv"]["accuracy"]["std"],"macro_f1":r["cv"]["macro_f1"]["mean"],"gap":r["accuracy_gap"]["mean"],
            "uniform_reference_accuracy":r["uniform_reference_accuracy"]["mean"],"delta_vs_uniform":r["paired_vs_uniform_same_frame_joint_count"]["accuracy_delta_mean"]})
    mode="INCLUDING" if config["include_contact"] else "EXCLUDING"
    lines=["# ST-GCN: consecutive precontact 5/3 frames × 6/8 joints","",f"Original full-sequence consecutive frames, {mode} the contact frame.",
           "Same C-camera 141 kicks, 5 folds × 3 seeds, same inner stopping sets and hyperparameters; 60 new fits.",
           "No interpolation or new normalization. Six joints exclude shoulder inputs only. No Test reevaluation.","",
           "| Frames | Joints | Train accuracy | Validation mean ± SD | Macro F1 | Gap | Earlier uniform validation |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results.values():
        c=r["cv"];s=r["spec"]
        lines.append(f"| {s['frames']} | {s['joints']} | {100*c['training_accuracy']['mean']:.2f}% | {100*c['accuracy']['mean']:.2f}% ± {100*c['accuracy']['std']:.2f} pp | {c['macro_f1']['mean']:.4f} | {100*r['accuracy_gap']['mean']:.2f} pp | {100*r['uniform_reference_accuracy']['mean']:.2f}% |")
    lines += ["",f"Majority baseline: {100*baseline:.2f}%. Best by CV macro F1: {comparison['selected_by_cv_macro_f1']}.","",
              "## Dataset OOF accuracy (mean over seeds)","","| Frames | Joints | CLT | EPL | Women |","|---|---:|---:|---:|---:|"]
    for r in results.values():
        values=[f"{100*r['by_dataset'][d]['accuracy_mean']:.2f}%" for d in ("CLT","EPL","Women")]
        lines.append(f"| {r['spec']['frames']} | {r['spec']['joints']} | "+" | ".join(values)+" |")
    lines += ["","Earlier uniform windows covered first-to-contact progress; these windows cover consecutive raw frames immediately before contact.",
              "Both temporal coverage and (when excluded) endpoint availability change; differences cannot be attributed to frame count alone.",
              "FPS differs/may be absent, so 5/3 raw frames are not an identical physical duration across videos.",
              "Fold/seed SD is descriptive, not a confidence interval. Kicks, not players/matches, are split.",
              "The old Test has already been exposed; these are exploratory CV results, not a new independent Test confirmation."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=ROOT/"experiment_precontact_config.json");p.add_argument("--worker",type=int)
    a=p.parse_args();config=json.loads(a.config.read_text());out=ROOT/config["output_dir"]
    if a.worker is not None:worker(config,a.worker);return
    info=prepare(config);print(json.dumps({"development":info["development"],"fits":info["total_fits"],"include_contact":config["include_contact"]}),flush=True)
    processes=[]
    try:
        for i in range(len(config["devices"])):
            log=(out/f"worker_{i}.log").open("x")
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--config",str(a.config.resolve()),"--worker",str(i)],stdout=log,stderr=subprocess.STDOUT)
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
