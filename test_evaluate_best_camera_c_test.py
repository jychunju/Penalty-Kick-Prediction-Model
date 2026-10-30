import unittest
import numpy as np
from evaluate_best_camera_c_test import mean_probabilities


class FrozenEnsembleTests(unittest.TestCase):
    def test_equal_probability_average_not_best_model_selection(self):
        p=np.array([[[.9,.1],[.3,.7]],[[.2,.8],[.7,.3]],[[.4,.6],[.5,.5]]])
        result=mean_probabilities(p)
        np.testing.assert_allclose(result,[[.5,.5],[.5,.5]])
        np.testing.assert_allclose(result,mean_probabilities(p[::-1]))

    def test_invalid_probabilities_rejected(self):
        for p in (np.zeros((2,2)),np.full((2,2,2),np.nan),np.full((2,2,2),.7),np.array([[[1.1,-.1]]])):
            with self.assertRaises(ValueError):mean_probabilities(p)


if __name__=="__main__":unittest.main()
