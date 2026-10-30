#!/usr/bin/env python3
"""Reconcile all frame/joint fits, including reference reuse and input identity."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from experiment_frames_joints import (
    JOINTS, FlexiblePoseClassifier, coordinate_indices, delta_summary, load_data, select_inputs, uniform_indices,
)
from train_8j20_angles_lr import ROOT, evaluate, hash_file, save_json
from validate_8j20_lr_cv import check_predictions


def read_rows(path):
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def validate(out):
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    info=json.loads((out/"run_info.json").read_text());config=info["config"]
    comparison=json.loads((out/"comparison.json").read_text());tasks=json.loads((out/"tasks.json").read_text())
    folds=json.loads((out/"folds.json").read_text());reference=ROOT/config["reference_run"]
    old_info=json.loads((reference/"run_info.json").read_text());old_tasks=json.loads((reference/"tasks.json").read_text())
    for path,digest in info["protected_file_hashes"].items():assert hash_file(Path(path))==digest,path
    for name,digest in {**info["reference_source_code_hashes"],**info["source_code_hashes"]}.items():assert hash_file(ROOT/name)==digest,name
    assert folds==json.loads((reference/"folds.json").read_text())
    x,y,samples=load_data(out);groups=[s["group_id"] for s in samples]
    with np.load(reference/"development_only.npz") as d:
        np.testing.assert_array_equal(x.numpy(),d["X"][:,:,:16]);np.testing.assert_array_equal(y.numpy(),d["y"])
        assert groups==d["group_id"].tolist()
    assert tuple(x.shape)==(141,20,16) and len(set(groups))==141
    assert not(set(groups)&set(info["protected_test_groups"]))
    covered=[]
    for fold in folds:
        for a,b in itertools.combinations(fold["group_ids"],2):assert not(set(fold["group_ids"][a])&set(fold["group_ids"][b]))
        assert set.union(*(set(v) for v in fold["group_ids"].values()))==set(groups)
        for phase,indices in fold["indices"].items():assert fold["group_ids"][phase]==[groups[i] for i in indices]
        covered.extend(fold["group_ids"]["validation"])
    assert len(covered)==len(set(covered))==141
    assert len(tasks)==240 and len({t["task_key"] for t in tasks})==240
    counts={True:0,False:0};prediction_count=0;max_probability_error=0.0
    for name,result in comparison["results"].items():
        spec=info["conditions"][name];runs=[]
        assert spec["frame_indices"]==uniform_indices(spec["frames"])
        assert spec["coordinate_indices"]==coordinate_indices(spec["joints"])
        selected=select_inputs(x,spec["frames"],spec["joints"])
        np.testing.assert_array_equal(selected.numpy(),x.numpy()[:,spec["frame_indices"],:][:,:,spec["coordinate_indices"]])
        assert tuple(selected.shape)==(141,spec["frames"],2*spec["joints"])
        for seed in info["training_config"]["training_seeds"]:
            oof=read_rows(out/name/f"oof_seed_{seed}.csv")
            calculated=check_predictions(oof,groups)
            for k,v in calculated.items():assert v==result["per_seed_oof"][str(seed)][k]
            oof_by_group={r["group_id"]:r for r in oof}
            for fold in folds:
                task=next(t for t in tasks if (t["condition"],t["seed"],t["fold"])==(name,seed,fold["fold"]))
                directory=Path(task["artifact_directory"])
                stored=json.loads((directory/"metrics.json").read_text());history=json.loads((directory/"history.json").read_text())
                run=next(r for r in result["runs"] if r["seed"]==seed and r["fold"]==fold["fold"])
                assert run=={**stored,**task,**spec}
                best=max(history,key=lambda h:(h["early_stop"]["macro_f1"],-h["early_stop"]["loss"]))
                assert best["epoch"]==stored["best_epoch"] and best["early_stop"]==stored["early_stop"]
                assert all("validation" not in h and "test" not in h for h in history)
                assert stored["test_evaluations"]==0 and stored["outer_validation_evaluations"]==1
                assert len(history)==stored["epochs_completed"]<=info["training_config"]["epochs"]
                rows=read_rows(directory/"validation_predictions.csv")
                calculated=check_predictions(rows,fold["group_ids"]["validation"])
                for k,v in calculated.items():assert v==stored["validation"][k]
                for row in rows:
                    i=groups.index(row["group_id"])
                    assert row["actual"]==samples[i]["label"] and row["dataset"]==samples[i]["dataset"]
                    for column in ("prob_L","prob_R"):assert abs(float(row[column])-float(oof_by_group[row["group_id"]][column]))<1e-7
                checkpoint=torch.load(directory/"best.pt",map_location="cpu")
                assert checkpoint["epoch"]==stored["best_epoch"]
                assert checkpoint["training_indices"]==fold["indices"]["inner_train"]
                assert checkpoint["frame_indices"]==spec["frame_indices"]
                expected_features=[f"kp_{j}_{a}n" for j in JOINTS[spec["joints"]] for a in "xy"]
                assert checkpoint["feature_columns"]==expected_features
                model=FlexiblePoseClassifier(spec["architecture"],spec["joints"],info["training_config"]["dropout"])
                model.load_state_dict(checkpoint["model"],strict=True)
                assert sum(p.numel() for p in model.parameters())==stored["parameters"]==result["parameters"]
                if task["reused_reference"]:
                    assert spec["joints"]==8 and spec["frames"] in (20,10)
                    expected_condition=f"{spec['architecture']}{spec['frames']}_xy"
                    assert directory==reference/expected_condition/f"seed_{seed}"/f"fold_{fold['fold']}"
                    old_index=next(i for i,t in enumerate(old_tasks) if (t["condition"],t["seed"],t["fold"])==(expected_condition,seed,fold["fold"]))
                    device_name=old_info["config"]["devices"][old_index%len(old_info["config"]["devices"])]
                else:
                    assert checkpoint["joint_ids"]==list(JOINTS[spec["joints"]])
                    assert stored["training_kicks"]==len(fold["indices"]["inner_train"])
                    device_name=config["devices"][task["worker_id"]]
                device=torch.device(device_name)
                if device.type=="cuda":torch.cuda.set_device(device)
                model.to(device)
                validation,probabilities=evaluate(model,selected,y,fold["indices"]["validation"],samples,device)
                train,_=evaluate(model,selected,y,fold["indices"]["inner_train"],samples,device)
                recorded=np.array([[float(r["prob_L"]),float(r["prob_R"])] for r in rows])
                max_probability_error=max(max_probability_error,float(np.abs(probabilities-recorded).max()))
                np.testing.assert_allclose(probabilities,recorded,atol=2e-6,rtol=2e-6)
                assert np.array_equal(probabilities.argmax(-1),recorded.argmax(-1))
                for metric in ("accuracy","macro_f1"):
                    assert validation[metric]==stored["validation"][metric]
                    assert train[metric]==stored["train"][metric]
                counts[task["reused_reference"]]+=1;prediction_count+=len(rows);runs.append(run)
        assert len(runs)==15
        for metric in ("accuracy","macro_f1"):
            values=[r["validation"][metric] for r in runs]
            assert abs(np.mean(values)-result["cv"][metric]["mean"])<1e-12
            assert abs(np.std(values,ddof=1)-result["cv"][metric]["std"])<1e-12
        gaps=[r["train"]["accuracy"]-r["validation"]["accuracy"] for r in runs]
        assert abs(np.mean(gaps)-result["accuracy_gap"]["mean"])<1e-12
        assert abs(np.mean([r["train"]["accuracy"] for r in runs])-result["cv"]["training_accuracy"]["mean"])<1e-12
        assert result["paired_vs_8_joints_same_frames"]==delta_summary(result,comparison["results"][f"{spec['architecture']}_f{spec['frames']}_j8"])
        assert result["paired_vs_20_frames_same_joints"]==delta_summary(result,comparison["results"][f"{spec['architecture']}_f20_j{spec['joints']}"])
    assert counts=={True:60,False:180}
    assert comparison["total_scored_fits"]==240 and comparison["test_evaluations"]==0
    assert (out/"train_validation_curves.png").stat().st_size>0
    result={"status":"PASS","development_kicks":141,"new_fits_checked":180,"reused_fits_checked":60,"checkpoints_loaded":240,
            "validation_predictions_checked":prediction_count,"train_and_validation_metrics_reproduced":True,
            "max_reproduced_probability_error":max_probability_error,"fixed_kicks_folds_stopping_sets_and_seeds":True,
            "pure_input_subsetting_no_renormalization":True,"protected_files_unchanged":True,"test_evaluations":0}
    save_json(out/"artifact_validation.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--directory",type=Path,default=ROOT/"runs/main_data_camera_c_frames_joints")
    validate(p.parse_args().directory)
