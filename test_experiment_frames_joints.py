"""Frame/joint subsetting, anatomical graph and reference equivalence tests."""
import unittest

import numpy as np
import torch

from compare_8j20_lr_cv import SmallPoseClassifier
from experiment_camera_c import DynamicPoseClassifier
from experiment_frames_joints import (
    JOINTS, FlexiblePoseClassifier, coordinate_indices, joint_graph, select_inputs, uniform_indices,
)


class FrameJointTests(unittest.TestCase):
    def test_uniform_selection_preserves_endpoints_order(self):
        expected={20:list(range(20)),10:[0,2,4,6,8,11,13,15,17,19],5:[0,5,10,14,19],3:[0,10,19]}
        for n,indices in expected.items():
            self.assertEqual(uniform_indices(n),indices)
            self.assertEqual(len(indices),n);self.assertEqual(indices,sorted(set(indices)))
            self.assertEqual(indices[0],0);self.assertEqual(indices[-1],19)
        with self.assertRaises(ValueError):uniform_indices(2)

    def test_coordinate_subsets_exact_no_renormalization(self):
        original=torch.arange(2*20*16,dtype=torch.float32).reshape(2,20,16)
        copy=original.clone()
        self.assertEqual(coordinate_indices(8),list(range(16)))
        self.assertEqual(coordinate_indices(6),list(range(4,16)))
        for j in (8,6):
            for t in (20,10,5,3):
                selected=select_inputs(original,t,j)
                self.assertEqual(tuple(selected.shape),(2,t,2*j))
                np.testing.assert_array_equal(selected.numpy(),original.numpy()[:,uniform_indices(t),:][:,:,coordinate_indices(j)])
        self.assertTrue(torch.equal(original,copy))

    def test_six_joint_graph_remaps_hips_and_limb_edges(self):
        edges,roots=joint_graph(6)
        self.assertEqual(JOINTS[6],(11,12,13,14,15,16))
        self.assertEqual(roots,(0,1))
        self.assertEqual(set(edges),{(0,1),(0,2),(2,4),(1,3),(3,5)})
        model=FlexiblePoseClassifier("stgcn",6)
        self.assertEqual(tuple(model.adjacency.shape),(3,6,6))
        self.assertTrue(torch.isfinite(model.adjacency).all())

    def test_eight_joint_initialization_and_outputs_match_reference(self):
        torch.set_num_threads(2)
        for architecture in ("stgcn","cnn"):
            torch.manual_seed(2026);old=SmallPoseClassifier(f"small_{architecture}_xy").eval()
            torch.manual_seed(2026);new=FlexiblePoseClassifier(architecture,8).eval()
            for name,value in old.state_dict().items():
                self.assertTrue(torch.equal(value,new.state_dict()[name]),name)
            dynamic=DynamicPoseClassifier(f"small_{architecture}_xy").eval();dynamic.load_state_dict(old.state_dict())
            with torch.no_grad():
                x=torch.randn(4,20,16)
                self.assertTrue(torch.equal(old(x),new(x)))
                self.assertTrue(torch.equal(dynamic(x[:,:10]),new(x[:,:10])))

    def test_all_input_shapes_gradients_and_parameter_counts(self):
        torch.set_num_threads(2)
        counts={('stgcn',8):11074,('stgcn',6):11066,('cnn',8):4034,('cnn',6):3706}
        for architecture in ("stgcn","cnn"):
            for joints in (8,6):
                model=FlexiblePoseClassifier(architecture,joints)
                self.assertEqual(sum(p.numel() for p in model.parameters()),counts[(architecture,joints)])
                for frames in (20,10,5,3):
                    model.zero_grad(set_to_none=True)
                    output=model(torch.randn(4,frames,2*joints))
                    self.assertEqual(tuple(output.shape),(4,2))
                    loss=torch.nn.functional.cross_entropy(output,torch.tensor([0,1,0,1]));loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        with self.assertRaises(ValueError):FlexiblePoseClassifier("cnn",4)
        with self.assertRaises(ValueError):FlexiblePoseClassifier("cnn",6)(torch.randn(3,3,16))


if __name__=="__main__":unittest.main()
