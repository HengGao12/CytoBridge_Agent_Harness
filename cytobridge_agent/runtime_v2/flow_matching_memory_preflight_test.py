from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from CytoBridge.tl.flow_matching_backends import CostBasedPairwiseOTCouplingStrategy, PairwiseCost
from cytobridge_agent.tools.training_tools import flow_matching_memory_preflight


class DenseCustomCoupling(CostBasedPairwiseOTCouplingStrategy):
    def build_pairwise_cost(self, x0, x1, *, time_idx, t0, t1, device):  # noqa: ANN001
        del time_idx, t0, t1, device
        return PairwiseCost(cost_matrix=np.zeros((x0.shape[0], x1.shape[0]), dtype=np.float32))


class FlowMatchingMemoryPreflightTests(unittest.TestCase):
    def test_warns_large_custom_dense_pairwise_coupling_without_blocking(self) -> None:
        training_data = SimpleNamespace(
            latent_by_time=[
                torch.zeros(5000, 2),
                torch.zeros(12000, 2),
            ]
        )
        target = SimpleNamespace(training_mode="custom")
        backend = SimpleNamespace(coupling=DenseCustomCoupling(chunk_size=1000, use_mini_batch=True))

        with patch.dict(
            "os.environ",
            {
                "CYTOBRIDGE_CUSTOM_FM_PREFLIGHT_MAX_DENSE_MB": "512",
                "CYTOBRIDGE_CUSTOM_FM_PREFLIGHT_MAX_PAIR_ENTRIES": "10000000",
            },
        ):
            message = flow_matching_memory_preflight(
                training_data=training_data,
                target=target,
                backend=backend,
            )

        self.assertIsNotNone(message)
        self.assertIn("Memory/scalability warning", str(message))
        self.assertIn("60,000,000 pair entries", str(message))


if __name__ == "__main__":
    unittest.main()
