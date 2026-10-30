#!/usr/bin/env python3
"""Verify exact full-source precontact windows, 60 fits and unchanged Test."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from experiment_precontact import XY_COLUMNS,desired_frames,load_data
from experiment_frames_joints import FlexiblePoseClassifier,JOINTS,coordinate_indices,delta_summary
from train_8j20_angles_lr import ROOT,evaluate,hash_file,save_json
from validate_8j20_lr_cv import check_predictions


def read_rows(path):
    with path.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))


def validate(out):
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    info=json.loads((out/"run_info.json").read_text());config=info["config"]
    comparison=json.loads((out/"comparison.json").read_text());folds=json.loads((out/"folds.json").read_text())
    tasks=json.loads((out/"tasks.json").read_text());audit=json.loads((out/"extraction_audit.json").read_text())
    reference=ROOT/config["reference_run"]
    for path,digest in info["protected_file_hashes"].items():assert hash_file(Path(path))==digest,path
    for name,digest in info["source_code_hashes"].items():assert hash_file(ROOT/name)==digest,name
    assert folds==json.loads((reference/"folds.json").read_text())
    windows,y,samples=load_data(out);groups=[s["group_id"] for s in samples]
    with np.load(reference/"development_only.npz") as d:
        np.testing.assert_array_equal(y.numpy(),d["y"]);assert groups==d["group_id"].tolist()
    assert len(groups)==len(set(groups))==141 and not(set(groups)&set(info["protected_test_groups"]))
    source=read_rows(ROOT/config["source_csv"])
    np.testing.assert_array_equal(windows[3].numpy(),windows[5][:,-3:,:].numpy())
    assert tuple(windows[5].shape)==(141,5,16) and tuple(windows[3].shape)==(141,3,16)
    for count in (5,3):
        exported=read_rows(out/f"precontact_{count}frames.csv");assert len(exported)==141*count
        for i,a in enumerate(audit):
            assert a["group_id"]==groups[i]
            frames=desired_frames(a["contact_frame"],count,config["include_contact"])
            assert a[f"frames_{count}"]==frames and all(b-a==1 for a,b in zip(frames,frames[1:]))
            rows=[source[line-2] for line in a[f"source_csv_lines_{count}"]]
            assert [int(float(r["frame"])) for r in rows]==frames
            assert all(f"{r['dataset']}:{r['kick_id']}"==groups[i] for r in rows)
            assert all(r["kick_direction"]==samples[i]["label"] for r in rows)
            assert all(r["dataset"]=="CLT" or r["camera_direction"]=="C" for r in rows)
            if not config["include_contact"]:assert all(float(r["last_touch"])==0 for r in rows)
            expected=np.array([[float(r[c]) for c in XY_COLUMNS] for r in rows],dtype=np.float32)
            np.testing.assert_array_equal(windows[count][i].numpy(),expected)
            for step,(old,new) in enumerate(zip(rows,exported[count*i:count*(i+1)])):
                assert {k:new[k] for k in old}==old
                assert new["sequence_id"]==a["sequence_id"] and int(new["precontact_idx"])==step
                assert int(new["frame_offset_from_contact"])==frames[step]-a["contact_frame"]
                assert new["effective_camera_direction"]=="C"
    covered=[]
    for fold in folds:
        for a,b in itertools.combinations(fold["group_ids"],2):assert not(set(fold["group_ids"][a])&set(fold["group_ids"][b]))
        assert set.union(*(set(v) for v in fold["group_ids"].values()))==set(groups)
        for phase,indices in fold["indices"].items():assert fold["group_ids"][phase]==[groups[i] for i in indices]
        covered.extend(fold["group_ids"]["validation"])
    assert len(covered)==len(set(covered))==141 and len(tasks)==60 and len(info["conditions"])==4
    count=prediction_count=0;max_probability_error=0.0
    prior=json.loads((reference/"comparison.json").read_text())["results"]
    for name,result in comparison["results"].items():
        spec=result["spec"];assert spec==info["conditions"][name]
        assert spec["coordinate_indices"]==coordinate_indices(spec["joints"])
        x=windows[spec["frames"]][:,:,spec["coordinate_indices"]];runs=[]
        for seed in info["training_config"]["training_seeds"]:
            oof=read_rows(out/name/f"oof_seed_{seed}.csv");oof_by_group={r["group_id"]:r for r in oof}
            calculated=check_predictions(oof,groups)
            for k,v in calculated.items():assert v==result["per_seed_oof"][str(seed)][k]
            for fold in folds:
                task=next(t for t in tasks if (t["condition"],t["seed"],t["fold"])==(name,seed,fold["fold"]))
                directory=Path(task["artifact_directory"]);stored=json.loads((directory/"metrics.json").read_text())
                history=json.loads((directory/"history.json").read_text())
                best=max(history,key=lambda h:(h["early_stop"]["macro_f1"],-h["early_stop"]["loss"]))
                assert best["epoch"]==stored["best_epoch"] and best["early_stop"]==stored["early_stop"]
                assert len(history)==stored["epochs_completed"]<=info["training_config"]["epochs"]
                assert all("validation" not in h and "test" not in h for h in history)
                assert stored["test_evaluations"]==0 and stored["outer_validation_evaluations"]==1
                rows=read_rows(directory/"validation_predictions.csv")
                calculated=check_predictions(rows,fold["group_ids"]["validation"])
                for k,v in calculated.items():assert v==stored["validation"][k]
                for row in rows:
                    assert row["actual"]==samples[groups.index(row["group_id"])]["label"]
                    for col in ("prob_L","prob_R"):assert abs(float(row[col])-float(oof_by_group[row["group_id"]][col]))<1e-7
                ckpt=torch.load(directory/"best.pt",map_location="cpu")
                assert ckpt["training_indices"]==fold["indices"]["inner_train"]
                assert ckpt["epoch"]==stored["best_epoch"] and ckpt["frames"]==spec["frames"]
                assert ckpt["include_contact"]==config["include_contact"]
                assert ckpt["joint_ids"]==list(JOINTS[spec["joints"]])
                assert ckpt["feature_columns"]==[f"kp_{j}_{a}n" for j in JOINTS[spec["joints"]] for a in "xy"]
                model=FlexiblePoseClassifier("stgcn",spec["joints"],info["training_config"]["dropout"])
                model.load_state_dict(ckpt["model"],strict=True)
                assert sum(p.numel() for p in model.parameters())==stored["parameters"]==result["parameters"]
                device=torch.device(config["devices"][task["worker_id"]])
                if device.type=="cuda":torch.cuda.set_device(device)
                model.to(device)
                validation,probabilities=evaluate(model,x,y,fold["indices"]["validation"],samples,device)
                training,_=evaluate(model,x,y,fold["indices"]["inner_train"],samples,device)
                recorded=np.array([[float(r["prob_L"]),float(r["prob_R"])] for r in rows])
                max_probability_error=max(max_probability_error,float(np.abs(probabilities-recorded).max()))
                np.testing.assert_allclose(probabilities,recorded,atol=2e-6,rtol=2e-6)
                assert np.array_equal(probabilities.argmax(-1),recorded.argmax(-1))
                for metric in ("accuracy","macro_f1"):
                    assert training[metric]==stored["train"][metric] and validation[metric]==stored["validation"][metric]
                runs.append(stored);count+=1;prediction_count+=len(rows)
        assert len(runs)==15
        for metric in ("accuracy","macro_f1"):
            values=[r["validation"][metric] for r in runs]
            assert abs(np.mean(values)-result["cv"][metric]["mean"])<1e-12
            assert abs(np.std(values,ddof=1)-result["cv"][metric]["std"])<1e-12
        assert abs(np.mean([r["train"]["accuracy"]-r["validation"]["accuracy"] for r in runs])-result["accuracy_gap"]["mean"])<1e-12
        assert result["paired_vs_uniform_same_frame_joint_count"]==delta_summary(result,prior[f"stgcn_f{spec['frames']}_j{spec['joints']}"])
    assert count==60 and comparison["total_fits"]==60 and comparison["test_evaluations"]==0
    result={"status":"PASS","development_kicks":141,"conditions":4,"checkpoints_loaded":60,
            "validation_predictions_checked":prediction_count,"raw_consecutive_precontact_frames_verified":True,
            "original_coordinate_metadata_labels_and_normalization_preserved":True,"same_folds_stopping_sets_and_seeds":True,
            "train_and_validation_metrics_reproduced":True,"max_reproduced_probability_error":max_probability_error,
            "source_and_prior_artifacts_including_test_unchanged":True,"test_evaluations":0}
    save_json(out/"artifact_validation.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--directory",type=Path,default=ROOT/"runs/main_data_camera_c_precontact")
    validate(p.parse_args().directory)
