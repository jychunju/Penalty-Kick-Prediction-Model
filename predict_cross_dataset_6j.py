"""Export per-kick predictions for the six-joint external test set."""
import csv
import json
from collections import Counter
from pathlib import Path

import torch

from model import LOWER_BODY_EDGES, PenaltySTGCN
from train_cross_dataset_6j import build_new


def kick_key(value):
    return tuple(int(part) if part.isdigit() else part for part in value.split("-"))


def main():
    root = Path(__file__).resolve().parent
    config = json.loads((root / "cross_dataset_6j_config.json").read_text())
    run_dir = root / config["output_dir"]
    samples, _ = build_new(root / config["external_test_csv"], config)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
    model = PenaltySTGCN(config["dropout"], num_joints=6,
                         edges=LOWER_BODY_EDGES, roots=(0, 1))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    id_to_label = {value: key for key, value in config["labels"].items()}
    results = []
    with torch.inference_mode():
        for sample in samples:
            x = torch.from_numpy(sample["x"]).unsqueeze(0)
            probabilities = model(x).softmax(dim=1)[0].tolist()
            predicted_index = max(range(3), key=probabilities.__getitem__)
            results.append({
                "kick_id": sample["kick_id"],
                "actual": sample["label"],
                "predicted": id_to_label[predicted_index],
                "correct": sample["label"] == id_to_label[predicted_index],
                "prob_L": probabilities[0],
                "prob_C": probabilities[1],
                "prob_R": probabilities[2],
                "confidence": probabilities[predicted_index],
                "frame_start": sample["selected_frames"][0],
                "frame_end": sample["selected_frames"][-1],
            })
    results.sort(key=lambda row: kick_key(row["kick_id"]))
    columns = list(results[0])
    with (run_dir / "external_test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(results)
    summary = {
        "checkpoint_epoch": checkpoint["epoch"],
        "num_predictions": len(results),
        "actual_counts": dict(Counter(row["actual"] for row in results)),
        "predicted_counts": dict(Counter(row["predicted"] for row in results)),
        "correct": sum(row["correct"] for row in results),
        "incorrect": sum(not row["correct"] for row in results),
    }
    (run_dir / "external_test_prediction_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
