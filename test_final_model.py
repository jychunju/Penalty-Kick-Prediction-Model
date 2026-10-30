"""Checkpoint compatibility and final-model input-contract checks."""
import unittest
import numpy as np
import torch
from model import PenaltySTGCN
from predict import load_model, predict


class FinalModelTests(unittest.TestCase):
    def test_checkpoint_and_backward(self):
        torch.set_num_threads(2)
        model = load_model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 11066)
        x = torch.randn(2, 3, 12, requires_grad=True)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 2))
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_input_contract_and_single_batch_equivalence(self):
        x = np.zeros((3, 12), dtype=np.float32)
        self.assertEqual(predict(x), predict(x[None]))
        for bad in [np.zeros((5, 12)), np.zeros((3, 16)), np.zeros((0, 3, 12)),
                    np.full((3, 12), np.nan)]:
            with self.assertRaises(ValueError):
                predict(bad)
        with self.assertRaises(ValueError):
            PenaltySTGCN()(torch.zeros(2, 5, 12))


if __name__ == '__main__':
    unittest.main()
