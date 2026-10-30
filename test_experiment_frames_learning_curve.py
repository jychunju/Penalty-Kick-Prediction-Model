"""Tests for temporal selection, nested source balance and model equivalence."""
import unittest
from collections import Counter

import numpy as np
import torch

from compare_8j20_lr_cv import SmallPoseClassifier
from experiment_frames_learning_curve import (
    ALL_CONDITIONS, VariableLengthSTGCN, frame_indices, nested_training_subsets,
)


class FrameLearningTests(unittest.TestCase):
    def test_selection_existing_samples_and_endpoints(self):
        self.assertEqual(frame_indices("full20_train100"), list(range(20)))
        self.assertEqual(frame_indices("uniform10_train100"), [0, 2, 4, 6, 8, 11, 13, 15, 17, 19])
        self.assertEqual(frame_indices("last10_train100"), list(range(10, 20)))
        original = np.arange(2*20*16).reshape(2, 20, 16)
        for condition in ALL_CONDITIONS:
            indices = frame_indices(condition)
            self.assertEqual(indices, sorted(set(indices)))
            self.assertEqual(indices[-1], 19)
            chosen = original[:, indices, :]
            np.testing.assert_array_equal(chosen[:, -1], original[:, -1])
        with self.assertRaises(ValueError):
            frame_indices("unknown")

    def test_nested_source_label_balanced_subsets_reproducible(self):
        samples = []
        for source, left, right in (("CLT", 6, 6), ("EPL", 16, 20), ("Women", 41, 33)):
            for label, count in (("L", left), ("R", right)):
                samples += [{"dataset": source, "label": label} for _ in range(count)]
        train = list(range(122))
        fractions = [.25, .5, .75, 1.]
        subsets = nested_training_subsets(samples, train, fractions, seed=2027)
        self.assertEqual([len(subsets[str(f)]) for f in fractions], [31, 61, 92, 122])
        self.assertEqual(subsets, nested_training_subsets(samples, train, fractions, seed=2027))
        self.assertNotEqual(subsets, nested_training_subsets(samples, train, fractions, seed=2028))
        self.assertEqual(subsets["1.0"], train)
        totals = Counter((s["dataset"], s["label"]) for s in samples)
        for i, fraction in enumerate(fractions):
            chosen = subsets[str(fraction)]
            counts = Counter((samples[j]["dataset"], samples[j]["label"]) for j in chosen)
            self.assertEqual(set(counts), set(totals))
            for key, total in totals.items():
                self.assertLessEqual(abs(counts[key]-len(chosen)*total/122), 1.01)
            if i:
                self.assertTrue(set(subsets[str(fractions[i-1])]).issubset(chosen))

    def test_model_identical_for_twenty_and_valid_for_ten(self):
        torch.set_num_threads(2)
        reference = SmallPoseClassifier("small_stgcn_xy")
        dynamic = VariableLengthSTGCN()
        dynamic.load_state_dict(reference.state_dict(), strict=True)
        reference.eval(); dynamic.eval()
        x = torch.randn(3, 20, 16)
        with torch.no_grad():
            self.assertTrue(torch.equal(reference(x), dynamic(x)))
        self.assertEqual(sum(p.numel() for p in dynamic.parameters()), 11074)
        dynamic.train()
        for frames in (10, 20):
            output = dynamic(torch.randn(4, frames, 16))
            self.assertEqual(tuple(output.shape), (4, 2))
            loss = torch.nn.functional.cross_entropy(output, torch.tensor([0, 1, 0, 1]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):
            dynamic(torch.randn(4, 10, 20))


if __name__ == "__main__":
    unittest.main()
