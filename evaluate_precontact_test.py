#!/usr/bin/env python3
"""Evaluate all FOUR frozen precontact settings on the already-exposed Test.

No refit, epoch selection, threshold tuning, or model selection by Test results.
Each setting uses all 15 pre-existing checkpoints, and an equal-probability
ensemble fixed before scoring. Repeated Test use is explicitly exploratory.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

from evaluate_best_camera_c_test import mean_probabilities,write_predictions
from experiment_camera_c import effective_camera
from experiment_frames_joints import FlexiblePoseClassifier,JOINTS,coordinate_indices
from experiment_precontact import XY_COLUMNS,desired_frames,extract_windows
from train_8j20_angles_lr import ROOT,LABELS,hash_file,load_samples,save_json,score


def verify_score(y,probabilities,metrics):
    predicted=probabilities.argmax(-1)
    assert float(accuracy_score(y,predicted))==metrics["accuracy"]
    assert float(f1_score(y,predicted,labels=[0,1],average="macro",zero_division=0))==metrics["macro_f1"]
    assert confusion_matrix(y,predicted,labels=[0,1]).tolist()==metrics["confusion_matrix_L_R"]


def execute(reference,out,device_name):
    if out.exists():raise FileExistsError("Do not overwrite Test results; use saved predictions for review")
    info=json.loads((reference/"run_info.json").read_text());cv=json.loads((reference/"comparison.json").read_text())
    assert json.loads((reference/"artifact_validation.json").read_text())["status"]=="PASS"
    assert info["config"]["include_contact"] is False
    assert set(info["conditions"])=={"stgcn_pre5_j6","stgcn_pre5_j8","stgcn_pre3_j6","stgcn_pre3_j8"}
    for path,digest in info["protected_file_hashes"].items():assert hash_file(Path(path))==digest,path
    for name,digest in info["source_code_hashes"].items():assert hash_file(ROOT/name)==digest,name
    protected=dict(info["protected_file_hashes"])
    protected.update({str(p):hash_file(p) for p in reference.rglob("*") if p.is_file()})
    folds=json.loads((reference/"folds.json").read_text())
    with np.load(reference/"development_only.npz") as d:
        development_groups=d["group_id"].tolist();development_y=d["y"].copy()
    protected_groups=set(info["protected_test_groups"])
    assert not(set(development_groups)&protected_groups)
    baseline_label=int(np.bincount(development_y,minlength=2).argmax())
    plans={}
    for name,spec in info["conditions"].items():
        runs=sorted(cv["results"][name]["runs"],key=lambda r:(r["seed"],r["fold"]))
        assert len(runs)==15 and {(r["seed"],r["fold"]) for r in runs}=={(s,f) for s in (42,43,44) for f in range(1,6)}
        plans[name]={"spec":spec,"checkpoints":[{"seed":r["seed"],"fold":r["fold"],
                     "path":str(Path(r["artifact_directory"])/"best.pt"),
                     "sha256":hash_file(Path(r["artifact_directory"])/"best.pt"),"best_epoch":r["best_epoch"]} for r in runs]}
    test_path=ROOT/info["training_config"]["data_dir"]/"test.csv"
    # Freeze all four settings and all model paths before scoring any Test kick.
    protocol={"plans":plans,"cv_selected_condition":cv["selected_by_cv_macro_f1"],
              "window_rule":"Original consecutive [t-5,...,t-1] or [t-3,...,t-1]; contact excluded",
              "camera_rule":"CLT=C by user confirmation; EPL/Women original camera_direction=C",
              "ensemble_rule":"Equal softmax-probability mean of all 15 checkpoints per setting; argmax L/R",
              "single_model_summary":"Mean and descriptive SD over 15 models on the same 26 Test kicks",
              "baseline_label_from_development":LABELS[baseline_label],"test_path":str(test_path),"test_sha256":hash_file(test_path),
              "protected_file_hashes":protected,"evaluator_sha256":hash_file(Path(__file__)),
              "new_training":False,"test_based_checkpoint_or_epoch_selection":False,"test_already_exposed":True,
              "interpretation":"Exploratory follow-up on an already-exposed Test, not a fresh independent confirmation"}
    out.mkdir(parents=True,exist_ok=False);save_json(out/"protocol.json",protocol)
    _,all_test,excluded=load_samples(test_path)
    assert not excluded and {s["group_id"] for s in all_test}==protected_groups
    samples=sorted([s for s in all_test if effective_camera(s)=="C"],key=lambda s:s["group_id"])
    assert len(samples)==26 and len(all_test)==32
    groups=[s["group_id"] for s in samples];y=np.array([LABELS.index(s["label"]) for s in samples])
    sources=np.array([s["dataset"] for s in samples])
    source_path=ROOT/info["config"]["source_csv"];sampling_path=ROOT/info["config"]["sampling_report"]
    with source_path.open(newline="",encoding="utf-8-sig") as f:source=list(csv.DictReader(f))
    sampling=json.loads(sampling_path.read_text());assert hash_file(source_path)==sampling["input_sha256"]
    x5,x3,audit,_=extract_windows(source,sampling["selected_sequences"],groups,np.stack([s["x"][:,:16] for s in samples]),False)
    windows={5:torch.from_numpy(x5),3:torch.from_numpy(x3)}
    assert x5.shape==(26,5,16) and x3.shape==(26,3,16)
    # Independently reconcile exact source rows and temporal offsets.
    for i,a in enumerate(audit):
        for count in (5,3):
            rows=[source[line-2] for line in a[f"source_csv_lines_{count}"]]
            assert [int(float(r["frame"])) for r in rows]==desired_frames(a["contact_frame"],count,False)
            assert all(float(r["last_touch"])==0 and r["kick_direction"]==samples[i]["label"] for r in rows)
            np.testing.assert_array_equal(windows[count][i].numpy(),np.array([[float(r[c]) for c in XY_COLUMNS] for r in rows],dtype=np.float32))
    save_json(out/"test_extraction_audit.json",audit)
    np.savez_compressed(out/"test_windows.npz",X5=x5,X3=x3,y=y,group_id=np.array(groups),dataset=sources,
                        frames_5=np.array([a["frames_5"] for a in audit]),contact_frames=np.array([a["contact_frame"] for a in audit]))
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True);device=torch.device(device_name)
    if device.type=="cuda":torch.cuda.set_device(device)
    baseline=np.zeros((26,2));baseline[:,baseline_label]=1;baseline_metrics=score(y,baseline,sources)
    results={};prediction_checks=0
    for name,plan in plans.items():
        spec=plan["spec"];x=windows[spec["frames"]][:,:,coordinate_indices(spec["joints"])];directory=out/name
        directory.mkdir(exist_ok=False);individual=[];probabilities=[]
        for p in plan["checkpoints"]:
            checkpoint=torch.load(p["path"],map_location="cpu")
            assert checkpoint["condition"]==name and checkpoint["epoch"]==p["best_epoch"]
            assert checkpoint["frames"]==spec["frames"] and checkpoint["joint_ids"]==list(JOINTS[spec["joints"]])
            assert checkpoint["include_contact"] is False and checkpoint["seed"]==p["seed"] and checkpoint["fold"]==p["fold"]
            fold=next(f for f in folds if f["fold"]==p["fold"])
            assert checkpoint["training_indices"]==fold["indices"]["inner_train"]
            assert not({development_groups[i] for i in checkpoint["training_indices"]}&protected_groups)
            model=FlexiblePoseClassifier("stgcn",spec["joints"],info["training_config"]["dropout"])
            model.load_state_dict(checkpoint["model"],strict=True);model.to(device);model.eval()
            with torch.no_grad():prediction=model(x.to(device)).cpu().softmax(-1).numpy()
            metrics=score(y,prediction,sources);verify_score(y,prediction,metrics);prediction_checks+=26
            probabilities.append(prediction);individual.append({"seed":p["seed"],"fold":p["fold"],"test":metrics})
            write_predictions(directory/f"predictions_seed_{p['seed']}_fold_{p['fold']}.csv",samples,prediction)
        probabilities=np.stack(probabilities);ensemble=mean_probabilities(probabilities)
        ensemble_metrics=score(y,ensemble,sources);verify_score(y,ensemble,ensemble_metrics)
        summary={m:{"mean":float(np.mean([r["test"][m] for r in individual])),
                    "std":float(np.std([r["test"][m] for r in individual],ddof=1))} for m in ("accuracy","macro_f1")}
        result={"spec":spec,"test_kicks":26,"individual_models":individual,"individual_model_summary":summary,
                "ensemble":ensemble_metrics,"ensemble_correct":int((ensemble.argmax(-1)==y).sum()),
                "reference_cv":{"train":cv["results"][name]["cv"]["training_accuracy"],
                                "validation":cv["results"][name]["cv"]["accuracy"],"macro_f1":cv["results"][name]["cv"]["macro_f1"]}}
        results[name]=result;save_json(directory/"metrics.json",result)
        write_predictions(directory/"ensemble_predictions.csv",samples,ensemble)
        np.savez_compressed(directory/"predictions.npz",probabilities=probabilities,ensemble=ensemble,y=y,group_id=np.array(groups))
        with (directory/"ensemble_predictions.csv").open() as f:saved=list(csv.DictReader(f))
        assert len(saved)==26 and sum(r["correct"]=="1" for r in saved)==result["ensemble_correct"]
    comparison={"results":results,"cv_selected_condition":protocol["cv_selected_condition"],"test_kicks":26,
                "dataset_counts":dict(Counter(s["dataset"] for s in samples)),"label_counts":dict(Counter(s["label"] for s in samples)),
                "excluded_non_c_test_kicks":6,"majority_baseline":baseline_metrics,"new_training":False,
                "fixed_checkpoint_evaluations":60,"test_already_exposed":True,"fresh_independent_confirmation":False}
    save_json(out/"comparison.json",comparison)
    with (out/"comparison.csv").open("x",newline="") as f:
        columns=["condition","frames","joints","train_accuracy","cv_validation_accuracy","test_model_mean","test_model_std","test_ensemble_accuracy","test_ensemble_correct","test_ensemble_macro_f1"]
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for name,r in results.items():w.writerow({"condition":name,"frames":r["spec"]["frames"],"joints":r["spec"]["joints"],
            "train_accuracy":r["reference_cv"]["train"]["mean"],"cv_validation_accuracy":r["reference_cv"]["validation"]["mean"],
            "test_model_mean":r["individual_model_summary"]["accuracy"]["mean"],"test_model_std":r["individual_model_summary"]["accuracy"]["std"],
            "test_ensemble_accuracy":r["ensemble"]["accuracy"],"test_ensemble_correct":r["ensemble_correct"],"test_ensemble_macro_f1":r["ensemble"]["macro_f1"]})
    for path,digest in protected.items():assert hash_file(Path(path))==digest,path
    validation={"status":"PASS","settings":4,"fixed_checkpoints":60,"test_kicks_per_setting":26,
                "individual_predictions_checked":prediction_checks,"ensemble_predictions_checked":4*26,
                "exact_contact_excluded_raw_consecutive_windows":True,"train_test_disjoint":True,
                "source_training_checkpoints_and_prior_test_artifacts_unchanged":True,"new_training":False,
                "test_based_checkpoint_or_epoch_selection":False}
    save_json(out/"artifact_validation.json",validation)
    lines=["# Four precontact ST-GCN settings: Test follow-up","",
           "Original consecutive frames immediately before contact, excluding contact; C-camera Test 26 kicks (CLT 3, EPL 5, Women 18).",
           "All 60 existing checkpoints and equal-probability ensembles fixed before scoring. No refit or Test-based checkpoint selection.",
           "IMPORTANT: This Test was already exposed. Results are exploratory follow-up, not fresh independent confirmation.","",
           "| Frames | Joints | Train mean | CV validation mean | Test single-model mean ± SD | Test ensemble | Ensemble F1 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results.values():
        s=r["spec"];m=r["individual_model_summary"]["accuracy"]
        lines.append(f"| {s['frames']} | {s['joints']} | {100*r['reference_cv']['train']['mean']:.2f}% | {100*r['reference_cv']['validation']['mean']:.2f}% | {100*m['mean']:.2f}% ± {100*m['std']:.2f} pp | {100*r['ensemble']['accuracy']:.2f}% ({r['ensemble_correct']}/26) | {r['ensemble']['macro_f1']:.4f} |")
    lines += ["",f"Development-majority baseline: {100*baseline_metrics['accuracy']:.2f}% (L only). CV-selected setting remains {protocol['cv_selected_condition']}; no Test winner is selected.",
              "","## Ensemble by dataset","","| Frames | Joints | CLT (3) | EPL (5) | Women (18) |","|---|---:|---:|---:|---:|"]
    for r in results.values():
        values=[f"{100*r['ensemble']['by_dataset'][d]['accuracy']:.2f}%" for d in ("CLT","EPL","Women")]
        lines.append(f"| {r['spec']['frames']} | {r['spec']['joints']} | "+" | ".join(values)+" |")
    lines += ["","Train/CV validation are single-model means, not validation scores for the 15-model Test ensemble.",
              "SD is model variability on the same Test set, not a confidence interval. One Test kick changes accuracy by 3.85 pp.",
              "This evaluates 95–96-kick CV-trained models, not a new all-141 development refit.",
              "Comparing four settings on an already-exposed Test cannot establish an unbiased winner; further tuning needs fresh holdout data."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"report":str(out/"REPORT.md"),"validation":validation,"results":{
        k:{"test_mean":r["individual_model_summary"]["accuracy"],"ensemble":r["ensemble"],"correct":r["ensemble_correct"]} for k,r in results.items()}},ensure_ascii=False,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--reference",type=Path,default=ROOT/"runs/main_data_camera_c_precontact")
    p.add_argument("--output",type=Path,default=ROOT/"runs/main_data_camera_c_precontact_test");p.add_argument("--device",default="cuda:0")
    a=p.parse_args();execute(a.reference,a.output,a.device)
