"""Regression checks for paired, protected-test cross-validation."""
import unittest
from collections import Counter

import numpy as np
import torch

from compare_8j20_lr_cv import SmallPoseClassifier, VARIANTS, make_folds


def development_samples():
    samples = []
    for source, left, right in (("CLT", 9, 9), ("EPL", 24, 30), ("Women", 60, 48)):
        for label, count in (("L", left), ("R", right)):
            for i in range(count):
                kick = f"{label}_{i:03d}"
                samples.append({"dataset": source, "kick_id": kick, "group_id": f"{source}:{kick}", "label": label})
    return sorted(samples, key=lambda s: s["group_id"])


class CrossValidationTests(unittest.TestCase):
    def test_fold_disjointness_protected_test_and_validation_coverage(self):
        samples = development_samples()
        folds = make_folds(samples, ["EPL:protected"])
        self.assertEqual(len(folds), 5)
        coverage = []
        for fold in folds:
            indices = fold["indices"]
            self.assertEqual([len(indices[k]) for k in ("inner_train", "early_stop", "validation")], [122, 22, 36])
            flattened = sum(indices.values(), [])
            self.assertEqual(sorted(flattened), list(range(180)))
            self.assertEqual(len(set(flattened)), 180)
            coverage.extend(indices["validation"])
            for partition in indices.values():
                strata = {(samples[i]["dataset"], samples[i]["label"]) for i in partition}
                self.assertEqual(len(strata), 6)
            counts = Counter(samples[i]["dataset"] for i in indices["validation"])
            self.assertIn(counts["CLT"], (3, 4))
            self.assertIn(counts["EPL"], (10, 11))
            self.assertIn(counts["Women"], (21, 22))
        self.assertEqual(sorted(coverage), list(range(180)))
        self.assertEqual(folds, make_folds(samples, ["EPL:protected"]))

    def test_duplicate_or_protected_group_is_rejected(self):
        samples = development_samples()
        with self.assertRaises(ValueError):
            make_folds(samples, [samples[0]["group_id"]])
        with self.assertRaises(ValueError):
            make_folds(samples+[samples[0]], [])

    def test_four_models_forward_backward_feature_guards(self):
        torch.set_num_threads(2)
        for variant in VARIANTS:
            model = SmallPoseClassifier(variant)
            count = 20 if variant.endswith("_angles") else 16
            x = torch.randn(4, 20, count)
            logits = model(x)
            self.assertEqual(tuple(logits.shape), (4, 2))
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1]))
            loss.backward()
            self.assertTrue(np.isfinite(float(loss.detach())))
            self.assertIsNotNone(model.head[-1].weight.grad)
            if model.use_angles:
                self.assertIsNotNone(model.angle_branch[1].weight.grad)
            self.assertLess(sum(p.numel() for p in model.parameters()), 20000)
            with self.assertRaises(ValueError):
                model(torch.randn(4, 20, 16 if count == 20 else 20))
        with self.assertRaises(ValueError):
            SmallPoseClassifier("unknown")


if __name__ == "__main__":
    unittest.main()
