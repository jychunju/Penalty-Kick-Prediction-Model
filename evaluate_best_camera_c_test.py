#!/usr/bin/env python3
"""Frozen Test evaluation of the CV-selected ST-GCN / 3 frames / 6 joints.

No training or checkpoint selection occurs here. The protocol, 15 checkpoints
and equal-weight probability ensemble are recorded before Test features load.
Individual-model mean and ensemble scores are different estimands and are
reported separately. Only camera-C L/R kicks from the original Test are scored.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiment_camera_c import effective_camera
from experiment_frames_joints import FlexiblePoseClassifier, select_inputs
from train_8j20_angles_lr import ROOT, LABELS, hash_file, load_samples, save_json, score

CONDITION = "stgcn_f3_j6"


def mean_probabilities(probabilities):
    p=np.asarray(probabilities)
    if p.ndim!=3 or p.shape[0]<1 or p.shape[2]!=2:
        raise ValueError("Expected (models, kicks, 2) probabilities")
    if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):
        raise ValueError("Invalid probability values")
    if not np.allclose(p.sum(-1),1,atol=1e-6,rtol=0):
        raise ValueError("Probabilities do not sum to one")
    return np.mean(p,axis=0,dtype=np.float64)


def write_predictions(path,samples,probabilities):
    with path.open("x",newline="",encoding="utf-8") as f:
        columns=["dataset","kick_id","group_id","actual","predicted","prob_L","prob_R","correct"]
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for s,p in zip(samples,probabilities):
            predicted=LABELS[int(p.argmax())]
            w.writerow({**{k:s[k] for k in columns[:3]},"actual":s["label"],"predicted":predicted,
                        "prob_L":float(p[0]),"prob_R":float(p[1]),"correct":int(predicted==s["label"])})


def execute(reference,out,device_name):
    if out.exists():raise FileExistsError("Test results already exist; do not overwrite or rerun to select models")
    info=json.loads((reference/"run_info.json").read_text());comparison=json.loads((reference/"comparison.json").read_text())
    validation=json.loads((reference/"artifact_validation.json").read_text());folds=json.loads((reference/"folds.json").read_text())
    assert validation["status"]=="PASS" and comparison["selected_by_cv_macro_f1"]==CONDITION
    for path,digest in info["protected_file_hashes"].items():assert hash_file(Path(path))==digest,path
    for name,digest in {**info["reference_source_code_hashes"],**info["source_code_hashes"]}.items():assert hash_file(ROOT/name)==digest,name
    result=comparison["results"][CONDITION];spec=result["spec"]
    assert (spec["architecture"],spec["frames"],spec["joints"])==("stgcn",3,6)
    runs=sorted(result["runs"],key=lambda r:(r["seed"],r["fold"]))
    assert len(runs)==15 and {(r["seed"],r["fold"]) for r in runs}=={(s,f) for s in (42,43,44) for f in range(1,6)}
    with np.load(reference/"development_only.npz") as d:
        development_groups=d["group_id"].tolist();development_y=d["y"].copy()
    protected=set(info["protected_test_groups"])
    assert not(set(development_groups)&protected)
    baseline_label=int(np.bincount(development_y,minlength=2).argmax())
    train_config=info["training_config"]
    test_path=ROOT/train_config["data_dir"]/"test.csv"
    protected_hashes=dict(info["protected_file_hashes"])
    protected_hashes.update({str(p):hash_file(p) for p in reference.rglob("*") if p.is_file()})
    checkpoints=[]
    for r in runs:
        p=Path(r["artifact_directory"])/"best.pt"
        checkpoints.append({"seed":r["seed"],"fold":r["fold"],"path":str(p),"sha256":hash_file(p)})
    # This document is written BEFORE any Test features/labels are parsed.
    protocol={"condition":CONDITION,"selection":"Previously fixed mean CV macro F1; no Test-based selection",
              "architecture":"stgcn","frames":3,"joint_ids":[11,12,13,14,15,16],"frame_indices":[0,10,19],
              "primary_prediction_rule":"Equal average of softmax probabilities from all 15 fixed CV models; argmax L/R",
              "individual_summary_rule":"Arithmetic mean and descriptive sample SD over 15 models on the SAME Test kicks",
              "camera_rule":"CLT=C by user confirmation; EPL/Women camera_direction=C; kick_direction L/R only",
              "majority_baseline_label_from_development":LABELS[baseline_label],"checkpoints":checkpoints,
              "protected_test_groups":sorted(protected),"test_csv":str(test_path),"test_csv_sha256":hash_file(test_path),
              "protected_file_hashes":protected_hashes,"evaluator_sha256":hash_file(Path(__file__)),
              "new_training":False,"test_based_tuning":False}
    out.mkdir(parents=True,exist_ok=False);save_json(out/"protocol.json",protocol)
    fields,all_test,excluded=load_samples(test_path)
    assert not excluded and {s["group_id"] for s in all_test}==protected
    samples=sorted([s for s in all_test if effective_camera(s)=="C"],key=lambda s:s["group_id"])
    rejected=[{"group_id":s["group_id"],"camera_direction":effective_camera(s)} for s in all_test if effective_camera(s)!="C"]
    assert len(samples)==26 and len(rejected)==6
    x=torch.from_numpy(np.stack([s["x"][:,:16] for s in samples]))
    x=select_inputs(x,3,6);assert tuple(x.shape)==(26,3,12) and torch.isfinite(x).all()
    y=np.array([LABELS.index(s["label"]) for s in samples]);sources=np.array([s["dataset"] for s in samples])
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    device=torch.device(device_name)
    if device.type=="cuda":torch.cuda.set_device(device)
    model_probabilities=[];individual=[]
    for r,p in zip(runs,checkpoints):
        ckpt=torch.load(p["path"],map_location="cpu")
        assert ckpt["condition"]==CONDITION and ckpt["frame_indices"]==[0,10,19] and ckpt["joint_ids"]==[11,12,13,14,15,16]
        fold=next(f for f in folds if f["fold"]==r["fold"])
        assert ckpt["training_indices"]==fold["indices"]["inner_train"]
        assert not({development_groups[i] for i in ckpt["training_indices"]}&protected)
        assert ckpt["epoch"]==r["best_epoch"] and ckpt["seed"]==r["seed"] and ckpt["fold"]==r["fold"]
        model=FlexiblePoseClassifier("stgcn",6,train_config["dropout"])
        model.load_state_dict(ckpt["model"],strict=True);model.to(device);model.eval()
        with torch.no_grad():probabilities=model(x.to(device)).cpu().softmax(-1).numpy()
        model_probabilities.append(probabilities)
        metrics=score(y,probabilities,sources)
        individual.append({"seed":r["seed"],"fold":r["fold"],"checkpoint":p["path"],"test":metrics})
        write_predictions(out/f"predictions_seed_{r['seed']}_fold_{r['fold']}.csv",samples,probabilities)
    model_probabilities=np.stack(model_probabilities);ensemble=mean_probabilities(model_probabilities)
    ensemble_metrics=score(y,ensemble,sources)
    baseline=np.zeros((len(samples),2));baseline[:,baseline_label]=1;baseline_metrics=score(y,baseline,sources)
    summary={metric:{"mean":float(np.mean([r["test"][metric] for r in individual])),
                     "std":float(np.std([r["test"][metric] for r in individual],ddof=1))} for metric in ("accuracy","macro_f1")}
    metrics={"condition":CONDITION,"input_shape":list(x.shape),"test_kicks":len(samples),"test_labels":dict(Counter(s["label"] for s in samples)),
             "test_dataset_counts":dict(Counter(s["dataset"] for s in samples)),"excluded_non_c_test":rejected,
             "ensemble":ensemble_metrics,"ensemble_correct":int((ensemble.argmax(-1)==y).sum()),
             "individual_model_summary":summary,"individual_models":individual,"majority_baseline":baseline_metrics,
             "reference_cv":{"training_accuracy":result["cv"]["training_accuracy"],"validation_accuracy":result["cv"]["accuracy"],"macro_f1":result["cv"]["macro_f1"]},
             "test_forward_passes":len(runs),"test_based_tuning":False,"new_training":False,
             "notes":["Test ensemble uses 15 models; the prior CV accuracy is mean single-model accuracy, not the same ensemble estimator.",
                      "Individual-model SD reflects model variability on one shared Test set, not uncertainty over independent Test sets.",
                      "Each CV model trained on 95–96 development kicks; no final all-141 refit has been performed.",
                      "Original Test was previously evaluated by the initial baseline; this selected setting was not tuned on its Test performance.",
                      "26 kicks is a small Test set; one prediction changes accuracy by 3.85 percentage points."]}
    save_json(out/"metrics.json",metrics);write_predictions(out/"ensemble_predictions.csv",samples,ensemble)
    np.savez_compressed(out/"test_predictions.npz",X=x.numpy(),y=y,probabilities=model_probabilities,ensemble=ensemble,
                        group_id=np.array([s["group_id"] for s in samples]),dataset=sources,
                        model_seed=np.array([r["seed"] for r in runs]),model_fold=np.array([r["fold"] for r in runs]))
    with (out/"test_camera_c.csv").open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for s in samples:writer.writerows(dict(row,camera_direction="C") for row in s["rows"])
    # Reconcile stored predictions independently; no model/settings changes.
    from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
    for p,m in [(ensemble,ensemble_metrics),*[(p,r["test"]) for p,r in zip(model_probabilities,individual)]]:
        predicted=p.argmax(-1)
        assert float(accuracy_score(y,predicted))==m["accuracy"]
        assert float(f1_score(y,predicted,labels=[0,1],average="macro",zero_division=0))==m["macro_f1"]
        assert confusion_matrix(y,predicted,labels=[0,1]).tolist()==m["confusion_matrix_L_R"]
    for path,digest in protected_hashes.items():assert hash_file(Path(path))==digest,path
    save_json(out/"artifact_validation.json",{"status":"PASS","fixed_models":15,"test_kicks":26,
              "individual_predictions_checked":15*26,"ensemble_predictions_checked":26,
              "checkpoint_train_test_disjointness":True,"protected_files_unchanged":True,"test_based_tuning":False})
    report=["# Fixed ST-GCN / 3 frames / 6 joints: C-camera Test evaluation","",
            "Protocol and checkpoint list frozen before Test features were parsed. No training or Test-based selection.",
            f"Test: {len(samples)} kicks; {metrics['test_dataset_counts']}; labels {metrics['test_labels']}. Six non-C Test kicks excluded.","",
            "| Result | Accuracy | Macro F1 |","|---|---:|---:|",
            f"| Mean of 15 individual CV models | {100*summary['accuracy']['mean']:.2f}% ± {100*summary['accuracy']['std']:.2f} pp | {summary['macro_f1']['mean']:.4f} |",
            f"| Equal probability ensemble of all 15 | {100*ensemble_metrics['accuracy']:.2f}% ({metrics['ensemble_correct']}/26) | {ensemble_metrics['macro_f1']:.4f} |",
            f"| Development-majority baseline ({LABELS[baseline_label]}) | {100*baseline_metrics['accuracy']:.2f}% | {baseline_metrics['macro_f1']:.4f} |","",
            "## Ensemble by dataset","","| Dataset | Kicks | Accuracy | Macro F1 |","|---|---:|---:|---:|"]
    for ds,m in ensemble_metrics["by_dataset"].items():report.append(f"| {ds} | {m['num_kicks']} | {100*m['accuracy']:.2f}% | {m['macro_f1']:.4f} |")
    report += ["",f"Ensemble confusion matrix (rows actual L/R, columns predicted L/R): {ensemble_metrics['confusion_matrix_L_R']}.","",
               "The prior CV accuracy (75.02%) is a mean over single fold models, not a validation score for this 15-model ensemble.",
               "All models remain frozen; none was selected by Test accuracy. No final all-141 development refit.",
               "Model SD is descriptive; 26 shared Test kicks are not 390 independent samples. One kick = 3.85 pp.",
               "This Test was previously exposed for the initial baseline. Further tuning after these results needs a new independent holdout."]
    (out/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({"ensemble":ensemble_metrics,"ensemble_correct":metrics["ensemble_correct"],"individual_model_summary":summary,
                      "majority_baseline":baseline_metrics,"report":str(out/"REPORT.md")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--reference",type=Path,default=ROOT/"runs/main_data_camera_c_frames_joints")
    p.add_argument("--output",type=Path,default=ROOT/"runs/main_data_camera_c_stgcn_f3_j6_test")
    p.add_argument("--device",default="cuda:0");a=p.parse_args();execute(a.reference,a.output,a.device)
