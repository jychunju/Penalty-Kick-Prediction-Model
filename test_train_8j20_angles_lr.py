"""Split, feature-isolation, and model-shape checks for the L/R baseline."""
import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from train_8j20_angles_lr import FEATURES, PoseAngleSTGCN, load_samples, stratified_kick_split


def sample_records():
    samples = []
    for dataset, left, right in (("CLT", 10, 11), ("EPL", 29, 35), ("Women", 71, 56)):
        for label, count in (("L", left), ("R", right)):
            for i in range(count):
                samples.append({"dataset": dataset, "kick_id": f"{label}{i:03d}",
                                "label": label, "group_id": f"{dataset}:{label}{i:03d}"})
    return samples


class TrainingTests(unittest.TestCase):
    def test_no_overlap_all_samples_and_reproducible(self):
        samples = sample_records()
        parts = stratified_kick_split(samples, [0.7, 0.15, 0.15], 42)
        self.assertEqual([len(v) for v in parts.values()], [148, 32, 32])
        self.assertEqual(parts, stratified_kick_split(samples, [0.7, 0.15, 0.15], 42))
        self.assertNotEqual(parts, stratified_kick_split(samples, [0.7, 0.15, 0.15], 43))
        flattened = sum(parts.values(), [])
        self.assertEqual(sorted(flattened), list(range(212)))
        self.assertEqual(len(set(flattened)), 212)

    def test_source_ratios_and_both_labels_in_every_split(self):
        samples = sample_records()
        parts = stratified_kick_split(samples, [0.7, 0.15, 0.15], 42)
        overall = Counter(sample["dataset"] for sample in samples)
        for indices in parts.values():
            counts = Counter(samples[i]["dataset"] for i in indices)
            for dataset, total in overall.items():
                self.assertLess(abs(counts[dataset]/len(indices)-total/212), 0.011)
                self.assertEqual({samples[i]["label"] for i in indices if samples[i]["dataset"] == dataset}, {"L", "R"})
        self.assertEqual(Counter(samples[i]["dataset"] for i in parts["validation"]),
                         Counter(samples[i]["dataset"] for i in parts["test"]))

    def write_fixture(self, path, invalid=False):
        fields = ["dataset", "kick_id", "kick_direction", "frame", "sample_20_idx", "sequence_id", "last_touch", "goal", *FEATURES]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for kick, label in (("1", "L"), ("2", "R"), ("3", "C")):
                for step in range(20):
                    row = {"dataset": "EPL", "kick_id": kick, "kick_direction": label,
                           "frame": str(step*2), "sample_20_idx": str(step), "sequence_id": f"EPL|{kick}|C|1",
                           "last_touch": "1" if step == 19 else "0", "goal": "" if step == 19 else "1"}
                    row.update({column: "0.5" for column in FEATURES})
                    if invalid and label == "L" and step == 10:
                        row[FEATURES[0]] = "nan"
                    writer.writerow(row)

    def test_c_exclusion_twenty_features_and_goal_not_used(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"fixture.csv"
            self.write_fixture(path)
            fields, samples, excluded = load_samples(path)
            self.assertEqual(len(samples), 2)
            self.assertEqual(len(excluded), 1)
            self.assertEqual(excluded[0]["label"], "C")
            self.assertEqual([s["label"] for s in samples], ["L", "R"])
            self.assertNotIn("goal", FEATURES)
            self.assertEqual(len(FEATURES), 20)
            for sample in samples:
                self.assertEqual(sample["x"].shape, (20, 20))
                self.assertEqual(sample["x"].dtype, np.float32)
                np.testing.assert_array_equal(sample["x"], np.full((20, 20), 0.5))
            self.write_fixture(path, invalid=True)
            with self.assertRaisesRegex(ValueError, "Invalid features"):
                load_samples(path)

    def test_model_forward_backward_and_shape_guard(self):
        torch.set_num_threads(2)
        model = PoseAngleSTGCN()
        x = torch.randn(4, 20, 20)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (4, 2))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.angle_branch[1].weight.grad)
        self.assertIsNotNone(model.blocks[0].spatial.weight.grad)
        with self.assertRaises(ValueError):
            model(torch.randn(4, 16, 20))


if __name__ == "__main__":
    unittest.main()
