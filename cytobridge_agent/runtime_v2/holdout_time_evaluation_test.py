from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np

from cytobridge_agent.runtime_v2.proposal_versioning_test import DummyLLM
from cytobridge_agent.runtime_v2.state import ensure_runtime_v2_state
from cytobridge_agent.runtime_v2.tool_registry import SingleAgentTools
from cytobridge_agent.tools.claim_metric_evaluator import load_campaign_claim_metric_evaluator


class HoldoutTimeEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.state = ensure_runtime_v2_state(
            {
                "session_id": "holdout-time-evaluation",
                "output_dir": str(self.root),
                "input_path": str(self.root / "input.h5ad"),
            }
        )
        self.tools = SingleAgentTools(DummyLLM(), self.state, agent_role="planner", agent_id="planner")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_normalize_holdout_modes(self) -> None:
        normalized = self.tools._normalize_holdout_time_eval_spec(
            {
                "enabled": True,
                "mode": "sequential",
                "time_points": [1, "2", 1.0, "bad"],
                "use_as_claim_metric": True,
            }
        )
        self.assertTrue(normalized["enabled"])
        self.assertEqual(normalized["mode"], "sequential")
        self.assertEqual(normalized["time_points"], [1.0, 2.0])
        self.assertEqual(normalized["groups"], [[1.0], [2.0]])
        self.assertTrue(normalized["attach_to_custom_metrics"])

    def test_builtin_holdout_claim_metric_spec_produces_evaluator_identity(self) -> None:
        spec = {
            "name": "holdout_time_w1",
            "direction": "lower",
            "evaluator_path": "builtin:holdout_time_w1",
            "holdout_time_evaluation": {
                "enabled": True,
                "mode": "single",
                "time_points": [1.0],
            },
        }
        hook, info = load_campaign_claim_metric_evaluator(spec)
        self.assertTrue(callable(hook))
        self.assertEqual(info["source"], "builtin:holdout_time_w1")
        self.assertEqual(info["function_name"], "evaluate_holdout_time_w1")

        holdout_spec = self.tools._holdout_time_eval_spec_from_claim_metric(spec)
        normalized = self.tools._normalize_holdout_time_eval_spec(holdout_spec)
        self.assertTrue(normalized["enabled"])
        self.assertTrue(normalized["attach_to_custom_metrics"])
        self.assertEqual(normalized["metric_name"], "holdout_time_w1")
        self.assertEqual(normalized["claim_metric_evaluator"]["source"], "builtin:holdout_time_w1")

        class Context:
            builtin_metrics = {
                "holdout_time_evaluation": {
                    "mean_w1": 0.25,
                }
            }

        self.assertEqual(hook(Context())["holdout_time_w1"], 0.25)

    def test_compute_holdout_w1_from_saved_trajectory(self) -> None:
        try:
            import ot  # noqa: F401
        except Exception as exc:  # pragma: no cover - dependency guard
            self.skipTest(f"POT is unavailable: {exc}")

        adata_path = self.root / "full.h5ad"
        adata = ad.AnnData(X=np.array([[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]], dtype=float))
        adata.obs["time_point_processed"] = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        adata.obsm["X_latent"] = np.asarray(adata.X, dtype=float)
        adata.write_h5ad(adata_path)

        trajectory_path = self.root / "evaluation_trajectory.npz"
        manifest = {
            "format": "cytobridge_evaluation_trajectory_npz",
            "point_keys": ["points_0", "points_1", "points_2"],
            "weight_keys": ["weights_0", "weights_1", "weights_2"],
            "time_points": [0.0, 1.0, 2.0],
            "observed_time_points": [0.0, 2.0],
            "observed_time_indices": [0, 2],
            "source": "test",
            "step": 1.0,
            "metadata": {},
        }
        np.savez_compressed(
            trajectory_path,
            manifest_json=np.asarray(json.dumps(manifest)),
            time_points=np.asarray([0.0, 1.0, 2.0]),
            observed_time_points=np.asarray([0.0, 2.0]),
            observed_time_indices=np.asarray([0, 2]),
            points_0=np.asarray([[0.0], [0.0]], dtype=float),
            points_1=np.asarray([[1.0], [1.0]], dtype=float),
            points_2=np.asarray([[2.0], [2.0]], dtype=float),
            weights_0=np.asarray([0.5, 0.5], dtype=float),
            weights_1=np.asarray([0.5, 0.5], dtype=float),
            weights_2=np.asarray([0.5, 0.5], dtype=float),
        )

        report = self.tools._compute_holdout_w1_from_trajectory(
            trajectory_path=str(trajectory_path),
            full_adata_path=str(adata_path),
            heldout_times=[1.0],
            time_key="time_point_processed",
            latent_key="X_latent",
        )

        self.assertEqual(report["status"], "ok")
        self.assertAlmostEqual(float(report["mean_w1"]), 0.0, places=8)
        self.assertEqual(report["timepoint_results"][0]["observed_cells"], 2)

    def test_holdout_evaluation_recovers_standard_artifact_path_when_metrics_omit_it(self) -> None:
        try:
            import ot  # noqa: F401
        except Exception as exc:  # pragma: no cover - dependency guard
            self.skipTest(f"POT is unavailable: {exc}")

        adata_path = self.root / "full.h5ad"
        adata = ad.AnnData(X=np.array([[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]], dtype=float))
        adata.obs["time_point_processed"] = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        adata.obsm["X_latent"] = np.asarray(adata.X, dtype=float)
        adata.write_h5ad(adata_path)

        run_id = "baseline_aux_missing_path"

        def fake_run_training_tool_impl(**kwargs):
            del kwargs
            run_dir = self.root / "training_runs" / run_id
            artifacts_dir = run_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "format": "cytobridge_evaluation_trajectory_npz",
                "point_keys": ["points_0", "points_1", "points_2"],
                "weight_keys": ["weights_0", "weights_1", "weights_2"],
                "time_points": [0.0, 1.0, 2.0],
                "observed_time_points": [0.0, 2.0],
                "observed_time_indices": [0, 2],
                "source": "test_missing_path_fallback",
                "step": 1.0,
                "metadata": {},
            }
            np.savez_compressed(
                artifacts_dir / "evaluation_trajectory.npz",
                manifest_json=np.asarray(json.dumps(manifest)),
                time_points=np.asarray([0.0, 1.0, 2.0]),
                observed_time_points=np.asarray([0.0, 2.0]),
                observed_time_indices=np.asarray([0, 2]),
                points_0=np.asarray([[0.0], [0.0]], dtype=float),
                points_1=np.asarray([[1.0], [1.0]], dtype=float),
                points_2=np.asarray([[2.0], [2.0]], dtype=float),
                weights_0=np.asarray([0.5, 0.5], dtype=float),
                weights_1=np.asarray([0.5, 0.5], dtype=float),
                weights_2=np.asarray([0.5, 0.5], dtype=float),
            )
            (artifacts_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "metrics_path": str(artifacts_dir / "metrics.json"),
                        "evaluation_trajectory_source": "builtin_simulate_trajectory",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.state["latest_training_run_id"] = run_id
            return "ok"

        self.tools._run_training_tool_impl = fake_run_training_tool_impl  # type: ignore[method-assign]
        report = self.tools._run_holdout_time_evaluation(
            {
                "enabled": True,
                "mode": "single",
                "time_points": [1.0],
                "time_key": "time_point_processed",
                "latent_key": "X_latent",
                "attach_to_custom_metrics": True,
            },
            parent_run_id="parent",
            candidate_name="balanced_ot_cfm",
            training_algorithm_id=None,
            stage="pilot",
            config_overrides={},
            run_label_prefix="fallback",
            adata_path=str(adata_path),
            device="",
            seed=None,
            decision_reason="test",
        )

        self.assertEqual(report["status"], "ok")
        self.assertAlmostEqual(float(report["mean_w1"]), 0.0, places=8)
        saved_metrics = json.loads(
            (self.root / "training_runs" / run_id / "artifacts" / "metrics.json").read_text(encoding="utf-8")
        )
        self.assertTrue(Path(saved_metrics["evaluation_trajectory_path"]).exists())
        self.assertTrue(Path(saved_metrics["artifacts"]["evaluation_trajectory_path"]).exists())


if __name__ == "__main__":
    unittest.main()
