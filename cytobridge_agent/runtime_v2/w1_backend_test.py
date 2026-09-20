import unittest
from unittest import mock

import numpy as np

from CytoBridge.tl import w1_backend


class W1BackendPolicyTest(unittest.TestCase):
    def test_auto_backend_is_selected_for_whole_panel(self):
        with mock.patch.object(w1_backend, "_geomloss_online_available", return_value=True):
            policy = w1_backend.select_w1_backend_policy(
                [(100, 100), (5000, 5000)],
                pair_threshold=16_000_000,
            )
        self.assertEqual(policy["w1_backend"], "geomloss_sinkhorn_online")
        self.assertEqual(policy["w1_backend_selection_scope"], "evaluation_panel")
        self.assertEqual(policy["w1_backend_max_pair_count"], 25_000_000)
        self.assertEqual(policy["w1_backend_pair_counts"], [10_000, 25_000_000])

    def test_auto_backend_keeps_small_panel_exact(self):
        policy = w1_backend.select_w1_backend_policy(
            [(100, 100), (4000, 4000)],
            pair_threshold=16_000_000,
        )
        self.assertEqual(policy["w1_backend"], "exact")
        self.assertTrue(policy["w1_backend_exact"])

    def test_exact_w1_computation_still_available(self):
        observed = np.array([[0.0], [2.0]])
        predicted = np.array([[1.0], [3.0]])
        policy = w1_backend.select_w1_backend_policy(
            [(observed.shape[0], predicted.shape[0])],
            requested_backend="exact",
        )
        value = w1_backend.compute_w1_distance(observed, predicted, backend_policy=policy)
        self.assertAlmostEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
