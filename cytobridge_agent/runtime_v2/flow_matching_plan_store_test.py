from __future__ import annotations

import unittest

import numpy as np
import torch

from CytoBridge.tl.flow_matching_backends import (
    CouplingPlanStore,
    PairwiseCost,
    TransportPlanBuilder,
    TransportSolverConfig,
)


class FlowMatchingPlanStoreTest(unittest.TestCase):
    def test_sparse_edge_store_samples_pairs_and_row_mass(self) -> None:
        X = [
            np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
            np.array([[10.0], [11.0]], dtype=np.float32),
        ]
        store = CouplingPlanStore.from_edges(
            source_n=3,
            target_n=2,
            edge_src=np.array([0, 0, 1, 2], dtype=np.int64),
            edge_tgt=np.array([0, 1, 1, 0], dtype=np.int64),
            edge_weight=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        )

        self.assertTrue(store.is_sparse_edges)
        self.assertTrue(store.has_terminal_mass_payload)
        batch = store.sample_pairs(
            X=X,
            time_idx=0,
            batch_size=64,
            device=torch.device("cpu"),
        )
        self.assertIsNotNone(batch)
        assert batch is not None
        observed = set(zip(batch.idx0.tolist(), batch.idx1.tolist()))
        self.assertLessEqual(observed, {(0, 0), (0, 1), (1, 1), (2, 0)})
        np.testing.assert_allclose(
            batch.sampled_terminal_mass.reshape(-1),
            store.edge_row_mass[batch.idx0],
        )

    def test_sparse_edge_store_accepts_edge_terminal_mass(self) -> None:
        store = CouplingPlanStore.from_edges(
            source_n=2,
            target_n=2,
            edge_src=[0, 1],
            edge_tgt=[1, 0],
            edge_weight=[0.25, 0.75],
            terminal_mass_edges=[2.0, 3.0],
        )
        X = [
            np.array([[0.0], [1.0]], dtype=np.float32),
            np.array([[10.0], [11.0]], dtype=np.float32),
        ]
        batch = store.sample_pairs(X=X, time_idx=0, batch_size=32, device=torch.device("cpu"))
        self.assertIsNotNone(batch)
        assert batch is not None
        expected = np.where(batch.idx0 == 0, 2.0, 3.0).astype(np.float32)
        np.testing.assert_allclose(batch.sampled_terminal_mass.reshape(-1), expected)

    def test_transport_plan_builder_solves_cost_to_plan_store(self) -> None:
        builder = TransportPlanBuilder(
            TransportSolverConfig(
                solver_mode="balanced",
                use_mini_batch=False,
                balanced_method="exact",
            )
        )
        store = builder.solve_cost_to_store(
            PairwiseCost(cost_matrix=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)),
            device=torch.device("cpu"),
        )
        self.assertTrue(store.is_dense)
        self.assertEqual(store.dense_plan.shape, (2, 2))
        self.assertGreater(float(store.dense_plan.sum()), 0.0)

    def test_transport_plan_builder_supports_wfr_oet_solver(self) -> None:
        builder = TransportPlanBuilder(
            TransportSolverConfig(
                solver_mode="wfr",
                use_mini_batch=False,
            )
        )
        store = builder.solve_cost_to_store(
            PairwiseCost(cost_matrix=np.array([[0.0, 0.2], [0.3, 0.0]], dtype=np.float32)),
            device=torch.device("cpu"),
        )
        self.assertTrue(store.is_chunked)
        self.assertTrue(store.has_terminal_mass_payload)
        X = [
            np.array([[0.0], [1.0]], dtype=np.float32),
            np.array([[10.0], [11.0]], dtype=np.float32),
        ]
        batch = store.sample_pairs(X=X, time_idx=0, batch_size=8, device=torch.device("cpu"))
        self.assertIsNotNone(batch)
        assert batch is not None
        terminal_mass = store.terminal_mass_for_batch(batch)
        self.assertEqual(terminal_mass.shape, (8, 1))
        self.assertTrue(np.isfinite(terminal_mass).all())


if __name__ == "__main__":
    unittest.main()
