#!/usr/bin/env python3
"""Independently check C-camera selection, folds, artifacts and predictions."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from experiment_camera_c import CONDITIONS, DynamicPoseClassifier, effective_camera, selected_frames
from train_8j20_angles_lr import ROOT, hash_file, load_samples, save_json, score
from validate_8j20_lr_cv import check_predictions


def read_rows(path):
    with path.open(newline="",encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def validate(out):
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    info=json.loads((out/"run_info.json").read_text());config=info["config"]
    comparison=json.loads((out/"comparison.json").read_text());folds=json.loads((out/"folds.json").read_text())
    tasks=json.loads((out/"tasks.json").read_text())
    for path,digest in info["protected_file_hashes"].items():
        assert hash_file(Path(path))==digest,path
    for name,digest in info["source_code_hashes"].items():
        assert hash_file(ROOT/name)==digest,name
    protected=set(info["protected_test_groups"])
    originals={}
    for phase in ("train","validation"):
        _,samples,excluded=load_samples(ROOT/config["data_dir"]/f"{phase}.csv")
        assert not excluded
        originals.update({s["group_id"]:s for s in samples})
    expected={g:s for g,s in originals.items() if effective_camera(s)=="C"}
    with np.load(out/"development_only.npz") as data:
        x,y=data["X"].copy(),data["y"].copy();groups=data["group_id"].tolist();sources=data["dataset"].copy()
    assert x.shape==(len(expected),20,20) and np.isfinite(x).all()
    assert len(groups)==len(set(groups)) and set(groups)==set(expected) and not(set(groups)&protected)
    for i,g in enumerate(groups):
        np.testing.assert_array_equal(x[i],expected[g]["x"])
        assert ("L","R")[y[i]]==expected[g]["label"]
    filtered=read_rows(out/"camera_c_development.csv")
    assert len(filtered)==20*len(groups) and all(r["camera_direction"]=="C" for r in filtered)
    for i,g in enumerate(groups):
        rows=filtered[20*i:20*i+20]
        for old,new in zip(expected[g]["rows"],rows):
            assert new==dict(old,camera_direction="C")
    covered=[]
    for fold in folds:
        for a,b in itertools.combinations(fold["group_ids"],2):
            assert not(set(fold["group_ids"][a])&set(fold["group_ids"][b]))
        assert set.union(*(set(v) for v in fold["group_ids"].values()))==set(groups)
        for phase,indices in fold["indices"].items():
            assert fold["group_ids"][phase]==[groups[i] for i in indices]
            assert len({(sources[i],y[i]) for i in indices})==6
        covered.extend(fold["group_ids"]["validation"])
    assert len(covered)==len(set(covered))==len(groups)
    count,predictions=0,0
    max_probability_error=0.0
    assert len(tasks)==len(CONDITIONS)*len(folds)*len(config["training_seeds"])
    for result in comparison["results"]:
        name=result["condition"];spec=CONDITIONS[name];runs=[]
        for seed in config["training_seeds"]:
            oof=read_rows(out/name/f"oof_seed_{seed}.csv")
            calculated=check_predictions(oof,groups)
            for k,v in calculated.items():
                assert v==result["per_seed_oof"][str(seed)][k]
            oof_by_group={r["group_id"]:r for r in oof}
            for fold in folds:
                directory=out/name/f"seed_{seed}"/f"fold_{fold['fold']}"
                metrics=json.loads((directory/"metrics.json").read_text());history=json.loads((directory/"history.json").read_text())
                best=max(history,key=lambda r:(r["early_stop"]["macro_f1"],-r["early_stop"]["loss"]))
                assert best["epoch"]==metrics["best_epoch"] and best["early_stop"]==metrics["early_stop"]
                assert len(history)==metrics["epochs_completed"]<=config["epochs"]
                assert all("validation" not in h and "test" not in h for h in history)
                assert metrics["test_evaluations"]==0 and metrics["outer_validation_evaluations"]==1
                rows=read_rows(directory/"validation_predictions.csv")
                calculated=check_predictions(rows,fold["group_ids"]["validation"])
                for k,v in calculated.items():
                    assert v==metrics["validation"][k]
                for row in rows:
                    assert row["actual"]==expected[row["group_id"]]["label"]
                    assert row["dataset"]==expected[row["group_id"]]["dataset"]
                    assert row["group_id"] not in protected
                    for column in ("prob_L","prob_R"):
                        assert abs(float(row[column])-float(oof_by_group[row["group_id"]][column]))<1e-7
                checkpoint=torch.load(directory/"best.pt",map_location="cpu")
                assert checkpoint["epoch"]==metrics["best_epoch"]
                assert checkpoint["training_indices"]==fold["indices"]["inner_train"]
                frames=selected_frames(spec["window"])
                assert checkpoint["frame_indices"]==frames==metrics["frame_indices"]
                assert checkpoint["config"]["dropout"]==spec.get("dropout",config["dropout"])
                assert checkpoint["config"]["weight_decay"]==spec.get("weight_decay",config["weight_decay"])
                model=DynamicPoseClassifier(spec["variant"],checkpoint["config"]["dropout"])
                model.load_state_dict(checkpoint["model"],strict=True);model.eval()
                assert sum(p.numel() for p in model.parameters())==metrics["parameters"]
                assert len(checkpoint["feature_columns"])==model.feature_count==metrics["feature_count"]
                assert metrics["frames"]==len(frames)
                # Match the original GPU backend: CPU and CUDA/TF32 convolution
                # may differ slightly. evaluate() applies softmax on CPU logits.
                task_index=next(i for i,t in enumerate(tasks) if (t["condition"],t["seed"],t["fold"])==(name,seed,fold["fold"]))
                device=torch.device(config["devices"][task_index%len(config["devices"])])
                if device.type=="cuda":
                    torch.cuda.set_device(device)
                model.to(device)
                with torch.no_grad():
                    logits=model(torch.from_numpy(x[fold["indices"]["validation"]][:,frames,:model.feature_count]).to(device)).cpu()
                    probabilities=logits.softmax(-1).numpy()
                recorded=np.array([[float(r["prob_L"]),float(r["prob_R"])] for r in rows])
                max_probability_error=max(max_probability_error,float(np.abs(probabilities-recorded).max()))
                np.testing.assert_allclose(probabilities,recorded,atol=2e-6,rtol=2e-6)
                assert np.array_equal(probabilities.argmax(-1),recorded.argmax(-1))
                predictions+=len(rows);count+=1;runs.append(metrics)
        for metric in ("accuracy","macro_f1"):
            values=[r["validation"][metric] for r in runs]
            assert abs(np.mean(values)-result["cv"][metric]["mean"])<1e-12
            assert abs(np.std(values,ddof=1)-result["cv"][metric]["std"])<1e-12
        assert len(runs)==len(config["training_seeds"])*len(folds)
    assert count==len(tasks)==comparison["total_fits"]
    assert comparison["test_evaluations"]==0
    result={"status":"PASS","development_kicks":len(groups),"fits_checked":count,"checkpoints_loaded":count,
            "validation_predictions_checked":predictions,"validation_inference_reproduced_on_original_backend":True,
            "max_reproduced_probability_error":max_probability_error,
            "camera_filter_and_clt_override_verified":True,"all_six_source_label_strata_present_in_each_split":True,
            "original_features_and_labels_unchanged":True,"protected_files_unchanged":True,"test_evaluations":0}
    save_json(out/"artifact_validation.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--directory",type=Path,default=ROOT/"runs/main_data_camera_c_lr_comparison")
    validate(parser.parse_args().directory)
