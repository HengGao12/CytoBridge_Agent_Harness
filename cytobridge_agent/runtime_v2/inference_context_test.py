from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from CytoBridge.tl.trainer import TrainingPipeline, _clone_state_dict_tensors
from CytoBridge.tl.analysis import simulate_trajectory
from CytoBridge.tl.training_algorithm import (
    InferenceContext,
    SimulationResult,
    TrainingDataBundle,
)
from cytobridge_agent.tools.claim_metric_evaluator import load_evaluation_trajectory_artifact


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))


class _TrainableModuleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.velocity_net = torch.nn.Linear(3, 2)
        self.source_head = torch.nn.Linear(3, 1)
        self.cytobridge_component_modules = {"growth": ["source_head"]}


class _TinyDynamicsModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.components = ["velocity", "growth", "score"]
        self.velocity_net = torch.nn.Linear(3, 2)
        self.growth_net = torch.nn.Linear(3, 1)
        self.score_net = torch.nn.Linear(3, 1)

    def compute_score(self, t, x, create_graph=True):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_expanded = t.expand(x.size(0), 1)
        x = x.requires_grad_(True)
        out_score = self.score_net(torch.cat([x, t_expanded], dim=1))
        gradient = torch.autograd.grad(
            outputs=out_score,
            inputs=x,
            grad_outputs=torch.ones_like(out_score),
            create_graph=create_graph,
        )[0]
        return out_score, gradient


class InferenceContextTests(unittest.TestCase):
    def test_stage_setup_can_train_custom_component_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [torch.zeros(2, 2), torch.ones(2, 2)]
            adata = ad.AnnData(np.zeros((4, 2), dtype=np.float32))
            adata.obs["time_point_processed"] = [0.0, 0.0, 1.0, 1.0]
            adata.obsm["X_latent"] = np.zeros((4, 2), dtype=np.float32)
            bundle = TrainingDataBundle(
                adata=adata,
                time_points=[0.0, 1.0],
                latent_by_time=data,
                obs_indices_by_time=[np.array([0, 1]), np.array([2, 3])],
            )
            model = _TrainableModuleModel()
            trainer = TrainingPipeline(
                model,
                {
                    "model": {"components": ["velocity", "growth"]},
                    "training": {"defaults": {"sigma": 0.0}, "plan": []},
                    "ckpt_dir": tmpdir,
                },
                batch_size=2,
                device=torch.device("cpu"),
                training_data=bundle,
            )

            trainer._setup_stage(
                {
                    "name": "custom",
                    "mode": "flow_matching",
                    "lr": 0.001,
                    "train_strategy": "g",
                }
            )

            self.assertFalse(any(p.requires_grad for p in model.velocity_net.parameters()))
            self.assertTrue(all(p.requires_grad for p in model.source_head.parameters()))
            optimizer_param_ids = {
                id(param)
                for group in trainer.optimizer.param_groups
                for param in group["params"]
            }
            for param in model.source_head.parameters():
                self.assertIn(id(param), optimizer_param_ids)

    def test_stage_setup_supports_stage_trainable_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [torch.zeros(2, 2), torch.ones(2, 2)]
            adata = ad.AnnData(np.zeros((4, 2), dtype=np.float32))
            adata.obs["time_point_processed"] = [0.0, 0.0, 1.0, 1.0]
            adata.obsm["X_latent"] = np.zeros((4, 2), dtype=np.float32)
            bundle = TrainingDataBundle(
                adata=adata,
                time_points=[0.0, 1.0],
                latent_by_time=data,
                obs_indices_by_time=[np.array([0, 1]), np.array([2, 3])],
            )
            model = _TrainableModuleModel()
            model.cytobridge_component_modules = {}
            trainer = TrainingPipeline(
                model,
                {
                    "model": {"components": ["velocity", "growth"]},
                    "training": {"defaults": {"sigma": 0.0}, "plan": []},
                    "ckpt_dir": tmpdir,
                },
                batch_size=2,
                device=torch.device("cpu"),
                training_data=bundle,
            )

            trainer._setup_stage(
                {
                    "name": "custom",
                    "mode": "flow_matching",
                    "lr": 0.001,
                    "train_strategy": "",
                    "trainable_modules": ["source_head"],
                }
            )

            self.assertTrue(all(p.requires_grad for p in model.source_head.parameters()))

    def test_clone_state_dict_tensors_is_deep_copy(self) -> None:
        model = torch.nn.Linear(2, 1)
        cloned = _clone_state_dict_tensors(model.state_dict())
        original_weight = cloned["weight"].clone()
        with torch.no_grad():
            model.weight.add_(10.0)
        self.assertTrue(torch.equal(cloned["weight"], original_weight))
        self.assertFalse(torch.equal(cloned["weight"], model.state_dict()["weight"]))

    def test_builtin_simulate_trajectory_streams_without_retaining_graph(self) -> None:
        model = _TinyDynamicsModel()
        model.train()
        x0 = torch.zeros(4, 2)

        points, weights = simulate_trajectory(
            adata=None,
            model=model,
            x0=x0,
            sigma=0.0,
            time=[0.0, 0.5, 1.0],
            dt=0.25,
            device=torch.device("cpu"),
        )

        self.assertTrue(model.training)
        self.assertEqual(points.shape, (3, 4, 2))
        self.assertEqual(weights.shape, (3, 4, 1))
        self.assertFalse(x0.requires_grad)
        self.assertTrue(np.isfinite(points).all())
        self.assertTrue(np.isfinite(weights).all())
        for param in model.parameters():
            self.assertIsNone(param.grad)

    def test_training_time_budget_uses_continuous_adjacent_gap_bottleneck_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            counts = [4638, 14985, 29679]
            adata = ad.AnnData(np.zeros((sum(counts), 1), dtype=np.float32))
            adata.obs["time_point_processed"] = np.repeat([0.0, 1.0, 2.0], counts)
            adata.obsm["X_latent"] = np.zeros((sum(counts), 2), dtype=np.float32)
            data = [torch.zeros(count, 2) for count in counts]
            offsets = np.cumsum([0, *counts])
            bundle = TrainingDataBundle(
                adata=adata,
                time_points=[0.0, 1.0, 2.0],
                latent_by_time=data,
                obs_indices_by_time=[
                    np.arange(offsets[i], offsets[i + 1])
                    for i in range(len(counts))
                ],
            )

            trainer = TrainingPipeline(
                _TinyModel(),
                {
                    "model": {"components": []},
                    "training": {"defaults": {"sigma": 0.0}, "plan": []},
                    "ckpt_dir": tmpdir,
                },
                batch_size=2,
                device=torch.device("cpu"),
                training_data=bundle,
            )
            budget = trainer.training_time_budget
            self.assertEqual(budget["budget_rule"], "continuous_adjacent_gap_bottleneck_cells")
            self.assertEqual(budget["cell_count"], 49302)
            self.assertEqual(budget["timepoint_cell_counts"], counts)
            self.assertEqual(budget["gap_effective_cell_counts"], [4638, 14985])
            self.assertEqual(budget["effective_cell_count"], 19623)
            self.assertEqual(budget["gap_count"], 2)
            self.assertEqual(budget["seconds_per_cell_block"], 90)
            self.assertEqual(budget["rounding_seconds"], 30)
            self.assertEqual(budget["budget_sec"], 780)

    def test_training_time_budget_keeps_small_multitimepoint_simulation_short(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            counts = [400, 442, 530, 690, 969]
            adata = ad.AnnData(np.zeros((sum(counts), 1), dtype=np.float32))
            adata.obs["time_point_processed"] = np.repeat([0.0, 1.0, 2.0, 3.0, 4.0], counts)
            adata.obsm["X_latent"] = np.zeros((sum(counts), 2), dtype=np.float32)
            data = [torch.zeros(count, 2) for count in counts]
            offsets = np.cumsum([0, *counts])
            bundle = TrainingDataBundle(
                adata=adata,
                time_points=[0.0, 1.0, 2.0, 3.0, 4.0],
                latent_by_time=data,
                obs_indices_by_time=[
                    np.arange(offsets[i], offsets[i + 1])
                    for i in range(len(counts))
                ],
            )

            trainer = TrainingPipeline(
                _TinyModel(),
                {
                    "model": {"components": []},
                    "training": {"defaults": {"sigma": 0.0}, "plan": []},
                    "ckpt_dir": tmpdir,
                },
                batch_size=2,
                device=torch.device("cpu"),
                training_data=bundle,
            )
            budget = trainer.training_time_budget
            self.assertEqual(budget["gap_effective_cell_counts"], [400, 442, 530, 690])
            self.assertEqual(budget["effective_cell_count"], 2062)
            self.assertEqual(budget["budget_sec"], 240)
            self.assertLessEqual(budget["budget_sec"], 240)

    def test_inference_time_budget_caps_small_simulation_at_thirty_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            counts = [400, 442, 530, 690, 969]
            adata = ad.AnnData(np.zeros((sum(counts), 1), dtype=np.float32))
            adata.obs["time_point_processed"] = np.repeat([0.0, 1.0, 2.0, 3.0, 4.0], counts)
            adata.obsm["X_latent"] = np.zeros((sum(counts), 2), dtype=np.float32)
            data = [torch.zeros(count, 2) for count in counts]
            offsets = np.cumsum([0, *counts])
            bundle = TrainingDataBundle(
                adata=adata,
                time_points=[0.0, 1.0, 2.0, 3.0, 4.0],
                latent_by_time=data,
                obs_indices_by_time=[
                    np.arange(offsets[i], offsets[i + 1])
                    for i in range(len(counts))
                ],
            )

            trainer = TrainingPipeline(
                _TinyModel(),
                {
                    "model": {"components": []},
                    "training": {"defaults": {"sigma": 0.0}, "plan": []},
                    "ckpt_dir": tmpdir,
                },
                batch_size=2,
                device=torch.device("cpu"),
                training_data=bundle,
            )
            budget = trainer._build_inference_time_budget(data=data)
            self.assertEqual(budget["budget_rule"], "strict_adjacent_gap_bottleneck_cells")
            self.assertEqual(budget["gap_effective_cell_counts"], [400, 442, 530, 690])
            self.assertEqual(budget["effective_cell_count"], 2062)
            self.assertLessEqual(budget["budget_sec"], 30)

    def test_inference_time_budget_gives_weinreb_scale_small_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            counts = [4638, 14985, 29679]
            adata = ad.AnnData(np.zeros((sum(counts), 1), dtype=np.float32))
            adata.obs["time_point_processed"] = np.repeat([0.0, 1.0, 2.0], counts)
            adata.obsm["X_latent"] = np.zeros((sum(counts), 2), dtype=np.float32)
            data = [torch.zeros(count, 2) for count in counts]
            offsets = np.cumsum([0, *counts])
            bundle = TrainingDataBundle(
                adata=adata,
                time_points=[0.0, 1.0, 2.0],
                latent_by_time=data,
                obs_indices_by_time=[
                    np.arange(offsets[i], offsets[i + 1])
                    for i in range(len(counts))
                ],
            )

            trainer = TrainingPipeline(
                _TinyModel(),
                {
                    "model": {"components": []},
                    "training": {"defaults": {"sigma": 0.0}, "plan": []},
                    "ckpt_dir": tmpdir,
                },
                batch_size=2,
                device=torch.device("cpu"),
                training_data=bundle,
            )
            budget = trainer._build_inference_time_budget(data=data)
            self.assertEqual(budget["gap_effective_cell_counts"], [4638, 14985])
            self.assertEqual(budget["effective_cell_count"], 19623)
            self.assertEqual(budget["budget_sec"], 140)
            self.assertLessEqual(budget["budget_sec"], 150)

    def test_evaluation_sigma_falls_back_to_config_when_not_trained_in_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adata = ad.AnnData(np.zeros((4, 2), dtype=np.float32))
            adata.obs["time_point_processed"] = [0.0, 0.0, 1.0, 1.0]
            adata.obsm["X_latent"] = np.zeros((4, 2), dtype=np.float32)
            data = [torch.zeros(2, 2), torch.ones(2, 2)]
            bundle = TrainingDataBundle(
                adata=adata,
                time_points=[0.0, 1.0],
                latent_by_time=data,
                obs_indices_by_time=[np.array([0, 1]), np.array([2, 3])],
            )
            trainer = TrainingPipeline(
                _TinyModel(),
                {
                    "model": {"components": []},
                    "training": {
                        "defaults": {"sigma": 0.01},
                        "plan": [
                            {"name": "Train_FM", "mode": "flow_matching", "sigma": 0.05}
                        ],
                    },
                    "ckpt_dir": tmpdir,
                },
                batch_size=2,
                device=torch.device("cpu"),
                training_data=bundle,
            )

            self.assertEqual(trainer._evaluation_sigma(), 0.05)
            trainer.sigma = 0.0
            self.assertEqual(trainer._evaluation_sigma(), 0.0)

    def test_inference_context_builder_scopes_simulation_context_to_t0(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adata = ad.AnnData(np.zeros((5, 2), dtype=np.float32))
            adata.obs["time_point_processed"] = [0.0, 0.0, 1.0, 1.0, 1.0]
            adata.obsm["X_latent"] = np.zeros((5, 2), dtype=np.float32)
            data = [
                torch.zeros(2, 2),
                torch.ones(3, 2),
            ]
            bundle = TrainingDataBundle(
                adata=adata,
                time_points=[0.0, 1.0],
                latent_by_time=data,
                obs_indices_by_time=[np.array([0, 1]), np.array([2, 3, 4])],
            )
            calls = {"builder": False, "sim": False}

            def build_context(ctx):
                calls["builder"] = True
                self.assertEqual(ctx.t0_adata.n_obs, 2)
                self.assertEqual(tuple(ctx.initial_data.shape), (2, 2))
                return InferenceContext(
                    payload={"seed_count": int(ctx.initial_data.shape[0])},
                    visibility={"seed_count": "t0_inference"},
                    provenance={"future_rows_used": False},
                )

            def simulate(ctx):
                calls["sim"] = True
                self.assertIsNotNone(ctx.inference_context)
                self.assertEqual(ctx.inference_context.payload["seed_count"], 2)
                self.assertEqual(ctx.adata.n_obs, 2)
                self.assertEqual(len(ctx.training_data.latent_by_time), 1)
                self.assertEqual(ctx.observed_time_points, [0.0, 1.0])
                self.assertEqual(ctx.trajectory_time_points[0], 0.0)
                self.assertEqual(ctx.trajectory_time_points[-1], 1.0)
                self.assertGreater(len(ctx.trajectory_time_points), len(ctx.observed_time_points))
                return SimulationResult(
                    predicted_points_by_time=[
                        data[0].detach().cpu().numpy() + float(t)
                        for t in ctx.trajectory_time_points
                    ],
                    predicted_weights_by_time=[
                        np.ones(2, dtype=np.float32)
                        for _ in ctx.trajectory_time_points
                    ],
                    trajectory_time_points=list(ctx.trajectory_time_points),
                )

            trainer = TrainingPipeline(
                _TinyModel(),
                {
                    "model": {"components": []},
                    "training": {"defaults": {"sigma": 0.0}, "plan": []},
                    "ckpt_dir": tmpdir,
                },
                batch_size=2,
                device=torch.device("cpu"),
                training_data=bundle,
                inference_context_builder=build_context,
                simulation_hook=simulate,
            )
            trainer._predict_for_evaluation(adata=adata, data=data, time_points=[0.0, 1.0])
            self.assertTrue(calls["builder"])
            self.assertTrue(calls["sim"])

    def test_evaluate_uses_full_trajectory_for_builtin_and_custom_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adata = ad.AnnData(np.zeros((4, 2), dtype=np.float32))
            adata.obs["time_point_processed"] = [0.0, 0.0, 1.0, 1.0]
            adata.obsm["X_latent"] = np.zeros((4, 2), dtype=np.float32)
            data = [
                torch.zeros(2, 2),
                torch.ones(2, 2),
            ]
            bundle = TrainingDataBundle(
                adata=adata,
                time_points=[0.0, 1.0],
                latent_by_time=data,
                obs_indices_by_time=[np.array([0, 1]), np.array([2, 3])],
            )

            def simulate(ctx):
                return SimulationResult(
                    predicted_points_by_time=[
                        data[0].detach().cpu().numpy() + float(t)
                        for t in ctx.trajectory_time_points
                    ],
                    predicted_weights_by_time=[
                        np.ones(2, dtype=np.float32)
                        for _ in ctx.trajectory_time_points
                    ],
                    trajectory_time_points=list(ctx.trajectory_time_points),
                )

            def custom_metric(ctx):
                self.assertIsNotNone(ctx.evaluation_trajectory)
                self.assertEqual(ctx.observed_time_indices, [0, 10])
                self.assertEqual(len(ctx.simulated_points_by_time), 2)
                self.assertEqual(len(ctx.trajectory_points_by_time), 11)
                self.assertEqual(ctx.metadata["prediction_contract"], "full_t0_to_final_trajectory")
                return {"claim_metric": float(len(ctx.trajectory_time_points))}

            trainer = TrainingPipeline(
                _TinyModel(),
                {
                    "model": {"components": []},
                    "training": {"defaults": {"sigma": 0.0}, "plan": []},
                    "evaluation": {"trajectory_step": 0.1},
                    "ckpt_dir": tmpdir,
                },
                batch_size=2,
                device=torch.device("cpu"),
                training_data=bundle,
                simulation_hook=simulate,
                evaluation_metrics_hook=custom_metric,
            )
            metrics = trainer.evaluate(adata=adata, data=data, time_points=[0.0, 1.0])

            self.assertEqual(metrics["evaluation_prediction_contract"], "full_t0_to_final_trajectory")
            self.assertEqual(metrics["evaluation_trajectory_source"], "simulation_hook")
            self.assertEqual(metrics["evaluation_observed_time_indices"], [0, 10])
            self.assertEqual(len(metrics["evaluation_trajectory_time_points"]), 11)
            self.assertTrue(Path(metrics["evaluation_trajectory_path"]).exists())
            saved_trajectory = load_evaluation_trajectory_artifact(metrics["evaluation_trajectory_path"])
            self.assertEqual(saved_trajectory.observed_time_indices, [0, 10])
            self.assertEqual(len(saved_trajectory.points_by_time), 11)
            self.assertEqual(metrics["custom_metrics"]["claim_metric"], 11.0)

    def test_unit_weight_branching_trajectory_uses_relative_initial_mass_for_tmv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adata = ad.AnnData(np.zeros((5, 2), dtype=np.float32))
            adata.obs["time_point_processed"] = [0.0, 0.0, 1.0, 1.0, 1.0]
            adata.obsm["X_latent"] = np.zeros((5, 2), dtype=np.float32)
            data = [
                torch.zeros(2, 2),
                torch.ones(3, 2),
            ]
            bundle = TrainingDataBundle(
                adata=adata,
                time_points=[0.0, 1.0],
                latent_by_time=data,
                obs_indices_by_time=[np.array([0, 1]), np.array([2, 3, 4])],
            )

            def simulate(ctx):
                points = []
                weights = []
                for t in ctx.trajectory_time_points:
                    if abs(float(t) - 1.0) < 1e-8:
                        points.append(np.ones((3, 2), dtype=np.float32))
                        weights.append(np.ones(3, dtype=np.float32))
                    else:
                        points.append(np.zeros((2, 2), dtype=np.float32))
                        weights.append(np.ones(2, dtype=np.float32))
                return SimulationResult(
                    predicted_points_by_time=points,
                    predicted_weights_by_time=weights,
                    trajectory_time_points=list(ctx.trajectory_time_points),
                )

            trainer = TrainingPipeline(
                _TinyModel(),
                {
                    "model": {"components": []},
                    "training": {"defaults": {"sigma": 0.0}, "plan": []},
                    "evaluation": {"trajectory_step": 0.1},
                    "ckpt_dir": tmpdir,
                },
                batch_size=2,
                device=torch.device("cpu"),
                training_data=bundle,
                simulation_hook=simulate,
            )

            metrics = trainer.evaluate(adata=adata, data=data, time_points=[0.0, 1.0])
            self.assertAlmostEqual(metrics["tmv_scores"][0], 0.0)

    def test_simulation_hook_must_return_full_evaluation_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adata = ad.AnnData(np.zeros((4, 2), dtype=np.float32))
            adata.obs["time_point_processed"] = [0.0, 0.0, 1.0, 1.0]
            adata.obsm["X_latent"] = np.zeros((4, 2), dtype=np.float32)
            data = [
                torch.zeros(2, 2),
                torch.ones(2, 2),
            ]
            bundle = TrainingDataBundle(
                adata=adata,
                time_points=[0.0, 1.0],
                latent_by_time=data,
                obs_indices_by_time=[np.array([0, 1]), np.array([2, 3])],
            )

            def simulate_observed_only(ctx):
                del ctx
                return SimulationResult(
                    predicted_points_by_time=[
                        data[0].detach().cpu().numpy(),
                        data[1].detach().cpu().numpy(),
                    ],
                    predicted_weights_by_time=[
                        np.ones(2, dtype=np.float32),
                        np.ones(2, dtype=np.float32),
                    ],
                )

            trainer = TrainingPipeline(
                _TinyModel(),
                {
                    "model": {"components": []},
                    "training": {"defaults": {"sigma": 0.0}, "plan": []},
                    "evaluation": {"trajectory_step": 0.1},
                    "ckpt_dir": tmpdir,
                },
                batch_size=2,
                device=torch.device("cpu"),
                training_data=bundle,
                simulation_hook=simulate_observed_only,
            )

            with self.assertRaisesRegex(ValueError, "full trajectory"):
                trainer.evaluate(adata=adata, data=data, time_points=[0.0, 1.0])

    def test_evaluate_stops_slow_simulation_hook_on_inference_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adata = ad.AnnData(np.zeros((5, 2), dtype=np.float32))
            adata.obs["time_point_processed"] = [0.0, 0.0, 1.0, 1.0, 1.0]
            adata.obsm["X_latent"] = np.zeros((5, 2), dtype=np.float32)
            data = [
                torch.zeros(2, 2),
                torch.ones(3, 2),
            ]
            bundle = TrainingDataBundle(
                adata=adata,
                time_points=[0.0, 1.0],
                latent_by_time=data,
                obs_indices_by_time=[np.array([0, 1]), np.array([2, 3, 4])],
            )

            def simulate(ctx):
                del ctx
                time.sleep(0.2)
                trajectory_time_points = list(ctx.trajectory_time_points)
                return SimulationResult(
                    predicted_points_by_time=[
                        data[0].detach().cpu().numpy()
                        for _ in trajectory_time_points
                    ],
                    predicted_weights_by_time=[
                        np.ones(2, dtype=np.float32)
                        for _ in trajectory_time_points
                    ],
                    trajectory_time_points=trajectory_time_points,
                )

            trainer = TrainingPipeline(
                _TinyModel(),
                {
                    "model": {"components": []},
                    "training": {"defaults": {"sigma": 0.0}, "plan": []},
                    "evaluation": {"inference_timeout_sec": 0.05},
                    "ckpt_dir": tmpdir,
                },
                batch_size=2,
                device=torch.device("cpu"),
                training_data=bundle,
                simulation_hook=simulate,
            )
            metrics = trainer.evaluate(adata=adata, data=data, time_points=[0.0, 1.0])
            self.assertTrue(metrics["inference_timed_out"])
            self.assertEqual(metrics["w1_scores"], [])
            self.assertIn("Inference time budget reached", metrics["inference_timeout_message"])


if __name__ == "__main__":
    unittest.main()
