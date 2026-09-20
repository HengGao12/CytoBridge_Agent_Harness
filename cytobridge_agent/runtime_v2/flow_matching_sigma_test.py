from __future__ import annotations

import unittest

import torch

from CytoBridge.tl.flow_matching_backends import (
    RegularizedUnbalancedConditionalPath,
    SchrodingerBridgeConditionalPath,
    WFRFlowMatchingBackend,
    WFROETCouplingStrategy,
    WFRTerminalMassStrategy,
    WFRTravelingGaussianPath,
    default_flow_matching_backend_builder,
)
from CytoBridge.tl.training_algorithm import FlowMatchingBuildContext, TrainingDataBundle


class FlowMatchingSigmaTests(unittest.TestCase):
    def test_regularized_unbalanced_path_allows_zero_sigma_as_deterministic_limit(self) -> None:
        path = RegularizedUnbalancedConditionalPath(sigma=0.0)
        x0 = torch.zeros(4, 2)
        x1 = torch.ones(4, 2)
        t = torch.full((4,), 0.5)
        eps = path.sample_noise_like(x0)

        xt = path.sample_xt(x0, x1, t, eps)
        ut = path.compute_conditional_flow(x0, x1, t, xt)

        self.assertTrue(torch.allclose(xt, torch.full_like(xt, 0.5)))
        self.assertTrue(torch.allclose(ut, torch.ones_like(ut)))

    def test_schrodinger_bridge_path_still_requires_positive_sigma(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires sigma > 0"):
            SchrodingerBridgeConditionalPath(sigma=0.0)

    def test_wfrfm_backend_allows_zero_sigma_and_samples_finite_batch(self) -> None:
        backend = WFRFlowMatchingBackend(
            path=WFRTravelingGaussianPath(delta=2.0, sigma=0.0),
            coupling=WFROETCouplingStrategy(delta=2.0, use_mini_batch=False),
            mass=WFRTerminalMassStrategy(),
        )
        X = [
            torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]], dtype=torch.float32).numpy(),
            torch.tensor([[0.1, 0.1], [0.3, 0.1], [0.1, 0.3], [0.4, 0.2]], dtype=torch.float32).numpy(),
        ]
        t_train = torch.tensor([0.0, 1.0], dtype=torch.float32)

        backend.prepare(X, t_train, torch.device("cpu"))
        batch = backend.sample_batch(X, t_train, batch_size=5, device=torch.device("cpu"))

        self.assertEqual(batch.xt.shape, (5, 2))
        self.assertEqual(batch.gt.shape, (5, 1))
        self.assertTrue(torch.isfinite(batch.xt).all())
        self.assertTrue(torch.isfinite(batch.ut).all())
        self.assertTrue(torch.isfinite(batch.gt).all())
        self.assertTrue(torch.isfinite(batch.loss_weights).all())

    def test_wfrfm_auto_delta_uses_adjacent_latent_distance_scale(self) -> None:
        stage_params = {
            "mode": "flow_matching",
            "flow_matching": {
                "backend": "wfrfm",
                "coupling": {
                    "delta": "auto",
                    "delta_quantile": 0.9,
                    "delta_target_angle": 1.0,
                    "delta_sample_size": 8,
                    "delta_pair_samples": 16,
                },
                "path": {"delta": "auto", "sigma": 0.0},
                "mass": {"kind": "wfr_semicoupling"},
            },
        }
        training_data = TrainingDataBundle(
            adata=object(),
            time_points=[0.0, 1.0],
            latent_by_time=[
                torch.tensor([[0.0, 0.0]], dtype=torch.float32),
                torch.tensor([[4.0, 0.0]], dtype=torch.float32),
            ],
            obs_indices_by_time=[[], []],
        )

        backend = default_flow_matching_backend_builder(
            FlowMatchingBuildContext(
                stage_params=stage_params,
                training_data=training_data,
                device=torch.device("cpu"),
                regress_v=True,
                regress_g=True,
                regress_score=False,
            )
        )

        self.assertIsInstance(backend, WFRFlowMatchingBackend)
        self.assertAlmostEqual(backend.coupling.delta, 2.0)
        self.assertAlmostEqual(backend.path.delta, 2.0)
        self.assertAlmostEqual(stage_params["delta"], 2.0)
        self.assertEqual(stage_params["wfr_delta_metadata"]["strategy"], "auto_adjacent_distance_q")
        self.assertAlmostEqual(stage_params["wfr_delta_metadata"]["distance_quantile_value"], 4.0)

    def test_wfrfm_fixed_delta_still_overrides_auto_default(self) -> None:
        stage_params = {
            "mode": "flow_matching",
            "delta": 7.0,
            "flow_matching": {
                "backend": "wfrfm",
                "coupling": {"delta": 7.0},
                "path": {"delta": 7.0, "sigma": 0.0},
                "mass": {"kind": "wfr_semicoupling"},
            },
        }
        training_data = TrainingDataBundle(
            adata=object(),
            time_points=[0.0, 1.0],
            latent_by_time=[
                torch.tensor([[0.0, 0.0]], dtype=torch.float32),
                torch.tensor([[100.0, 0.0]], dtype=torch.float32),
            ],
            obs_indices_by_time=[[], []],
        )

        backend = default_flow_matching_backend_builder(
            FlowMatchingBuildContext(
                stage_params=stage_params,
                training_data=training_data,
                device=torch.device("cpu"),
                regress_v=True,
                regress_g=True,
                regress_score=False,
            )
        )

        self.assertAlmostEqual(backend.coupling.delta, 7.0)
        self.assertAlmostEqual(backend.path.delta, 7.0)
        self.assertEqual(stage_params["wfr_delta_metadata"]["strategy"], "fixed")

    def test_wfrfm_missing_delta_defaults_to_auto(self) -> None:
        stage_params = {
            "mode": "flow_matching",
            "flow_matching": {
                "backend": "wfrfm",
                "coupling": {
                    "delta_sample_size": 8,
                    "delta_pair_samples": 16,
                },
                "path": {"sigma": 0.0},
                "mass": {"kind": "wfr_semicoupling"},
            },
        }
        training_data = TrainingDataBundle(
            adata=object(),
            time_points=[0.0, 1.0],
            latent_by_time=[
                torch.tensor([[0.0, 0.0]], dtype=torch.float32),
                torch.tensor([[6.0, 0.0]], dtype=torch.float32),
            ],
            obs_indices_by_time=[[], []],
        )

        backend = default_flow_matching_backend_builder(
            FlowMatchingBuildContext(
                stage_params=stage_params,
                training_data=training_data,
                device=torch.device("cpu"),
                regress_v=True,
                regress_g=True,
                regress_score=False,
            )
        )

        self.assertAlmostEqual(backend.coupling.delta, 3.0)
        self.assertAlmostEqual(backend.path.delta, 3.0)
        self.assertEqual(stage_params["wfr_delta_metadata"]["strategy"], "auto_adjacent_distance_q")


if __name__ == "__main__":
    unittest.main()
