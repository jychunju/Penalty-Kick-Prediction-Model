"""C camera filtering, dynamic model equivalence and temporal selection tests."""
import unittest

import numpy as np
import torch

from compare_8j20_lr_cv import SmallPoseClassifier, VARIANTS, make_folds
from experiment_camera_c import CONDITIONS, DynamicPoseClassifier, effective_camera, select_c_samples, selected_frames


def sample(dataset, camera, kick="1", label="L"):
    return {"dataset":dataset,"kick_id":kick,"group_id":f"{dataset}:{kick}","label":label,
            "rows":[{"camera_direction":camera,"kick_direction":label} for _ in range(20)],
            "x":np.zeros((20,20),dtype=np.float32)}


class CameraTests(unittest.TestCase):
    def test_explicit_clt_camera_override_not_label_swap(self):
        old=sample("CLT","",kick="010-R-R.mp4",label="R")
        kept,excluded=select_c_samples([old],set())
        self.assertEqual(effective_camera(old),"C")
        self.assertEqual(kept[0]["rows"][0]["camera_direction"],"C")
        self.assertEqual(old["rows"][0]["camera_direction"],"")
        self.assertEqual(kept[0]["label"],"R")
        self.assertEqual(kept[0]["rows"][0]["kick_direction"],"R")
        self.assertEqual(excluded,[])
        np.testing.assert_array_equal(old["x"],kept[0]["x"])

    def test_camera_filter_not_direction_filter_and_test_exclusion(self):
        kept,excluded=select_c_samples([sample("EPL","C",label="R"),sample("Women","R")],set())
        self.assertEqual([s["group_id"] for s in kept],["EPL:1"])
        self.assertEqual(len(excluded),1)
        with self.assertRaises(ValueError):
            select_c_samples([sample("EPL","C")],{"EPL:1"})
        with self.assertRaises(ValueError):
            effective_camera(sample("EPL",""))
        mixed=sample("Women","C");mixed["rows"][0]["camera_direction"]="L"
        with self.assertRaises(ValueError):
            effective_camera(mixed)

    def test_existing_ordered_samples_and_contact_endpoint(self):
        self.assertEqual(selected_frames("uniform10"),[0,2,4,6,8,11,13,15,17,19])
        self.assertEqual(selected_frames("last10"),list(range(10,20)))
        for spec in CONDITIONS.values():
            frames=selected_frames(spec["window"])
            self.assertEqual(frames,sorted(set(frames)))
            self.assertEqual(frames[-1],19)
        with self.assertRaises(ValueError):
            selected_frames("unknown")

    def test_original_model_equivalence_and_ten_frame_gradients(self):
        torch.set_num_threads(2)
        for variant in VARIANTS:
            original=SmallPoseClassifier(variant).eval();dynamic=DynamicPoseClassifier(variant).eval()
            dynamic.load_state_dict(original.state_dict(),strict=True)
            x=torch.randn(3,20,dynamic.feature_count)
            with torch.no_grad():
                self.assertTrue(torch.equal(original(x),dynamic(x)),variant)
            self.assertEqual(sum(p.numel() for p in original.parameters()),sum(p.numel() for p in dynamic.parameters()))
            dynamic.train();out=dynamic(torch.randn(4,10,dynamic.feature_count))
            self.assertEqual(tuple(out.shape),(4,2))
            loss=torch.nn.functional.cross_entropy(out,torch.tensor([0,1,0,1]));loss.backward()
            self.assertTrue(torch.isfinite(loss))
            with self.assertRaises(ValueError):
                dynamic(torch.randn(4,10,17))

    def test_folds_on_actual_c_development_strata(self):
        samples=[]
        for ds,left,right in (("CLT",9,9),("EPL",12,9),("Women",57,45)):
            for label,count in (("L",left),("R",right)):
                samples.extend(sample(ds,"C",f"{label}{i}",label) for i in range(count))
        folds=make_folds(samples,["EPL:protected"])
        validation=[g for f in folds for g in f["group_ids"]["validation"]]
        self.assertEqual(len(validation),141);self.assertEqual(len(set(validation)),141)
        for fold in folds:
            for phase,indices in fold["indices"].items():
                strata={(samples[i]["dataset"],samples[i]["label"]) for i in indices}
                self.assertEqual(len(strata),6,phase)


if __name__=="__main__":
    unittest.main()
