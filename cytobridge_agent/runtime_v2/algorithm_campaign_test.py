from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from cytobridge_agent.runtime_v2.proposal_versioning_test import DummyLLM, _valid_proposal_payload
from cytobridge_agent.runtime_v2.state import ensure_runtime_v2_state
from cytobridge_agent.runtime_v2.tool_registry import SingleAgentTools
from cytobridge_agent.tools import planner_file_tools as planner_file_tools_module
from cytobridge_agent.tools import planner_tools as planner_tools_module
from cytobridge_agent.tools import workspace_policy as workspace_policy_module
from cytobridge_agent.tools.claim_metric_evaluator import load_campaign_claim_metric_evaluator
from cytobridge_agent.tools.training_tools import (
    apply_overrides,
    prune_redundant_config_overrides,
    validate_epoch_override_policy,
)


class AlgorithmCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self._original_cellcompass_root = planner_tools_module.get_cellcompass_root
        self._original_file_tools_root = planner_file_tools_module.get_cellcompass_root
        self._original_policy_root = workspace_policy_module.get_cellcompass_root
        planner_tools_module.get_cellcompass_root = lambda: Path(self.tmpdir.name)
        planner_file_tools_module.get_cellcompass_root = lambda: Path(self.tmpdir.name)
        workspace_policy_module.get_cellcompass_root = lambda: Path(self.tmpdir.name)
        self.state = ensure_runtime_v2_state(
            {
                "session_id": "algorithm-campaign",
                "output_dir": self.tmpdir.name,
                "input_path": "/tmp/input.h5ad",
                "algorithm_proposal_review_mode": "auto_approve",
            }
        )
        self.tools = SingleAgentTools(DummyLLM(), self.state, agent_role="planner", agent_id="planner")
        self._orig_start_algorithm_campaign = self.tools.start_algorithm_campaign
        self._orig_decide_campaign_trial = self.tools.planner_file_tools.decide_campaign_trial
        self._orig_save_campaign = self.tools.planner_file_tools._save_campaign
        self._disable_test_w1_backend_save_patch = False
        self.tools.start_algorithm_campaign = self._start_algorithm_campaign_with_test_w1_backend  # type: ignore[method-assign]
        self.tools.planner_file_tools.decide_campaign_trial = self._decide_campaign_trial_with_test_w1_backend  # type: ignore[method-assign]
        self.tools.planner_file_tools._save_campaign = self._save_campaign_with_test_w1_backend  # type: ignore[method-assign]
        payload = _valid_proposal_payload()
        self.assertIn("auto-approved", self.tools.create_algorithm_proposal(algorithm_id="algo_campaign", **payload))
        self.assertIn(
            "Initialized training algorithm workspace",
            self.tools.init_training_algorithm_workspace("algo_campaign", description="campaign test"),
        )

    def tearDown(self) -> None:
        planner_tools_module.get_cellcompass_root = self._original_cellcompass_root
        planner_file_tools_module.get_cellcompass_root = self._original_file_tools_root
        workspace_policy_module.get_cellcompass_root = self._original_policy_root
        self.tmpdir.cleanup()

    def _readme_path(self) -> Path:
        return Path(self.tmpdir.name) / "training_algorithms" / "algo_campaign" / "README.md"

    def _implementation_map_path(self) -> Path:
        return Path(self.tmpdir.name) / "training_algorithms" / "algo_campaign" / "IMPLEMENTATION_MAP.md"

    def _proposal_path(self) -> Path:
        return Path(self.tmpdir.name) / "training_algorithms" / "algo_campaign" / "PROPOSAL.md"

    def _patch_proposal_text(self, old: str, new: str, reason: str) -> str:
        patch = f"""*** Begin Patch
*** Update File: {self._proposal_path()}
@@
-{old}
+{new}
*** End Patch
"""
        return self.tools.apply_workspace_patch(patch, reason=reason)

    def _mark_algorithm_complete(self) -> None:
        registry = self.tools.planner_file_tools._bootstrap_algorithm_registry("algo_campaign")
        registry["algorithm_lifecycle_status"] = "complete"
        registry["algorithm_lifecycle_status_reason"] = "test final_regression locked release"
        registry["completed_campaign_id"] = "campaign_locked_test"
        registry["completed_release"] = {
            "trial_id": "trial_locked_test",
            "snapshot_id": str(registry.get("active_workspace_snapshot_id") or ""),
            "commit": "deadbeef",
            "locked_at": "2026-05-23T00:00:00Z",
        }
        self.tools.planner_file_tools._save_algorithm_registry(
            "algo_campaign",
            registry,
            sync_active_context=True,
        )

    def test_completed_algorithm_workspace_is_read_only(self) -> None:
        self._mark_algorithm_complete()

        patch = f"""*** Begin Patch
*** Update File: {self._readme_path()}
@@
-# algo_campaign
+# locked mutation
*** End Patch
"""
        patch_result = self.tools.apply_workspace_patch(patch, reason="should not mutate completed algorithm")
        self.assertIn("final-regression locked/read-only", patch_result)

        config_result = self.tools.planner_file_tools.patch_algorithm_config(
            "algo_campaign",
            {"training.plan[0].lr": 0.001},
            reason="should not mutate completed algorithm",
        )
        self.assertFalse(config_result["ok"])
        self.assertEqual(config_result["status"], "blocked")
        self.assertIn("final-regression locked/read-only", config_result["error"])

        with self.assertRaisesRegex(ValueError, "final-regression locked/read-only"):
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )

        proposal_result = self.tools.create_algorithm_proposal(
            algorithm_id="algo_campaign",
            **_valid_proposal_payload(),
        )
        self.assertIn("final-regression locked/read-only", proposal_result)

    def _write_benchmark_dataset_card(self, dataset_id: str, adata_path: str) -> None:
        dataset_dir = Path(self.tmpdir.name) / "algorithm_benchmarks" / "datasets" / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        readme_path = dataset_dir / "README.md"
        readme_path.write_text(f"# {dataset_id}\n", encoding="utf-8")
        card = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "contract_status": "ready",
            "paths": {
                "data_path": adata_path,
                "readme_path": str(readme_path),
                "builtin_configs_dir": str(dataset_dir / "builtin_configs"),
                "builtin_baselines_dir": str(dataset_dir / "builtin_baselines"),
                "leaderboard_path": str(dataset_dir / "leaderboard.json"),
            },
            "usage": {
                "campaign_dataset_entry": {
                    "dataset_id": dataset_id,
                    "adata_path": adata_path,
                    "config_overrides": {},
                },
            },
        }
        (dataset_dir / "builtin_configs").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "builtin_baselines").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset.json").write_text(json.dumps(card), encoding="utf-8")
        (dataset_dir / "leaderboard.json").write_text(
            json.dumps({"schema_version": 1, "dataset_id": dataset_id, "entries": []}),
            encoding="utf-8",
        )

    def _set_campaign_baseline_selection_policy(self, campaign_id: str, policy: str) -> None:
        campaign = self.tools.planner_file_tools.get_algorithm_campaign_status(campaign_id)
        campaign["baseline_selection_policy"] = policy
        if isinstance(campaign.get("claim_metric_spec"), dict):
            campaign["claim_metric_spec"]["baseline_selection_policy"] = policy
        self.tools.planner_file_tools._save_campaign(campaign)

    def _write_claim_metric_evaluator(self, value: float = 0.72, body: str | None = None) -> tuple[Path, dict]:
        evaluator_path = Path(self.tmpdir.name) / "claim_metric.py"
        evaluator_path.write_text(
            body
            or (
                "def evaluate_claim_metric(context):\n"
                f"    return {{'claim_metric': {value!r}}}\n"
            ),
            encoding="utf-8",
        )
        _, info = load_campaign_claim_metric_evaluator(
            {"name": "claim_metric", "direction": "greater", "evaluator_path": str(evaluator_path)}
        )
        return evaluator_path, dict(info)

    def _claim_metric_spec(self, value: float = 0.72) -> dict:
        evaluator_path, _ = self._write_claim_metric_evaluator(value)
        return {
            "name": "claim_metric",
            "direction": "greater",
            "evaluator_path": str(evaluator_path),
        }

    def _with_exact_w1_backend(self, payload):
        if isinstance(payload, dict):
            out = {key: self._with_exact_w1_backend(value) for key, value in payload.items()}
            if ("w1_mean" in out or "w1_scores" in out) and "w1_backend" not in out:
                out["w1_backend"] = "exact"
                out["w1_backend_exact"] = True
                out["w1_backend_params"] = {}
            return out
        if isinstance(payload, list):
            return [self._with_exact_w1_backend(item) for item in payload]
        return payload

    def _start_algorithm_campaign_with_test_w1_backend(self, *args, **kwargs):
        if isinstance(kwargs.get("claim_metric_spec"), dict):
            kwargs = dict(kwargs)
            kwargs["claim_metric_spec"] = self._with_exact_w1_backend(kwargs["claim_metric_spec"])
        return self._orig_start_algorithm_campaign(*args, **kwargs)

    def _decide_campaign_trial_with_test_w1_backend(self, *args, **kwargs):
        if isinstance(kwargs.get("metrics"), dict):
            kwargs = dict(kwargs)
            kwargs["metrics"] = self._with_exact_w1_backend(kwargs["metrics"])
        return self._orig_decide_campaign_trial(*args, **kwargs)

    def _save_campaign_with_test_w1_backend(self, campaign, *args, **kwargs):
        if not self._disable_test_w1_backend_save_patch:
            campaign = self._with_exact_w1_backend(campaign)
        return self._orig_save_campaign(campaign, *args, **kwargs)

    def test_start_campaign_normalizes_lower_is_better_direction(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={"primary_metric": "W1", "direction": "lower_is_better"},
            )
        )
        self.assertEqual(campaign["claim_metric_spec"]["name"], "w1_mean")
        self.assertEqual(campaign["claim_metric_spec"]["direction"], "lower")
        self.assertEqual(
            campaign["stage_policies"]["stage2_claim_validation"]["primary_direction"],
            "lower",
        )

    def _replace_readme(self, text: str) -> None:
        result = self.tools.planner_file_tools.replace_workspace_file(
            str(self._readme_path()),
            text,
            reason="campaign test edit",
        )
        self.assertIn("Replaced file", result)

    def _fill_implementation_map(self, deviation_status: str = "exact") -> None:
        content = f"""# Implementation Map: algo_campaign

Active proposal: `{self.state.get("active_proposal_id") or ""}`.

## Baseline Anchor

- Required anchor baseline: vgfm
- Why this is the nearest builtin: campaign test anchor
- Components reused or intentionally changed: package default

## Mapping Table

| Step ID | Pseudocode Step | Implementation File | Line Number(s) | Status | Semantic Check | Deviation Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | Implement the approved custom training algorithm semantics. | `algorithm.py` | `algorithm.py:1-120` | implemented | The custom spec preserves the approved proposal's training algorithm contract. | {deviation_status} | Test map row. |
"""
        result = self.tools.planner_file_tools.replace_workspace_file(
            str(self._implementation_map_path()),
            content,
            reason="fill implementation map for campaign test",
        )
        self.assertIn("Replaced file", result)

    def _fill_implementation_map_without_line_numbers(self) -> None:
        content = f"""# Implementation Map: algo_campaign

Active proposal: `{self.state.get("active_proposal_id") or ""}`.

## Baseline Anchor

- Required anchor baseline: vgfm
- Why this is the nearest builtin: campaign test anchor
- Components reused or intentionally changed: package default

## Mapping Table

| Step ID | Pseudocode Step | Implementation File | Line Number(s) | Status | Semantic Check | Deviation Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | Implement the approved custom training algorithm semantics. | `algorithm.py` | package default | completed | The custom spec preserves the approved proposal's training algorithm contract. | exact | Test map row without exact line numbers. |
"""
        result = self.tools.planner_file_tools.replace_workspace_file(
            str(self._implementation_map_path()),
            content,
            reason="fill implementation map without line numbers for campaign test",
        )
        self.assertIn("Replaced file", result)

    def _record_current_implementation_review(self, campaign_id: str) -> dict:
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        review_state = self.tools._implementation_review_state(
            "algo_campaign",
            proposal_id=str(campaign.get("proposal_id") or ""),
            campaign=campaign,
        )
        fingerprint = dict(review_state.get("fingerprint") or {})
        return self.tools.planner_file_tools.record_implementation_review(
            "algo_campaign",
            proposal_id=str(campaign.get("proposal_id") or ""),
            review_hash=str(fingerprint.get("review_hash") or ""),
            decision="approve",
            reviewer_feedback="Test implementation aligns with proposal.",
            campaign_id=campaign_id,
            campaign_trial_count=int(review_state.get("current_trial_count") or 0),
        )

    def test_campaign_promote_reject_resume_and_gate(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    **self._claim_metric_spec(),
                    "stage_baselines": {
                        "stage1_feasibility": {"w1_mean": 1.0, "tmv_scores": [0.1]},
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        archive_path = Path(campaign["archive"]["archive_path"])
        self.assertTrue((archive_path / "HEAD").exists())
        self.assertEqual(campaign["stage_policies"]["stage1_feasibility"]["max_trials"], 10)
        self.assertEqual(campaign["stage_policies"]["stage2_claim_validation"]["max_trials"], 50)
        self.assertEqual(campaign["stage_policies"]["stage3_tuning"]["max_trials"], 30)
        self.assertEqual(campaign["stage_policies"]["final_regression"]["max_trials"], 1)
        self.assertEqual(campaign["baseline_selection_policy"], "strict_all_builtin")

        trial1 = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# best\n")
        prepared1 = self.tools.planner_file_tools.prepare_campaign_trial(campaign_id)
        decision1 = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial1["trial_id"],
            metrics={"w1_scores": [0.9], "tmv_scores": [0.1], "custom_metrics": {"claim_metric": 0.5}},
        )
        self.assertEqual(decision1["decision"], "promote")
        self.assertEqual(decision1["active_best_trial_id"], trial1["trial_id"])

        trial2 = json.loads(self.tools.start_campaign_trial(campaign_id))
        self.assertEqual(self._readme_path().read_text(encoding="utf-8"), "# best\n")
        self._replace_readme("# bad\n")
        prepared2 = self.tools.planner_file_tools.prepare_campaign_trial(campaign_id)
        decision2 = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial2["trial_id"],
            metrics={"w1_scores": [1.0], "tmv_scores": [0.1], "custom_metrics": {"claim_metric": 0.6}},
        )
        self.assertEqual(decision2["decision"], "reject")
        self.assertEqual(decision2["active_best_trial_id"], trial1["trial_id"])
        self.assertEqual(decision2["restored_snapshot_id"], prepared1["snapshot_id"])
        self.assertEqual(self._readme_path().read_text(encoding="utf-8"), "# best\n")

        rejected = json.loads(self.tools.list_campaign_trials(campaign_id, decision="reject"))
        self.assertEqual(rejected["count"], 1)
        self.assertEqual(rejected["view"], "trial_list_summary")
        self.assertEqual(rejected["limit"], 10)
        self.assertFalse(rejected["has_more"])
        self.assertEqual(rejected["trials"][0]["trial_id"], trial2["trial_id"])
        self.assertIn("metrics_summary", rejected["trials"][0])
        self.assertNotIn("dataset_config_overrides", rejected["trials"][0])
        self.assertNotIn("events", rejected["trials"][0])

        resumed = json.loads(self.tools.resume_rejected_trial(campaign_id, trial2["trial_id"]))
        self.assertEqual(resumed["base_trial_id"], trial2["trial_id"])
        self.assertEqual(self._readme_path().read_text(encoding="utf-8"), "# bad\n")
        self._replace_readme("# recovered\n")
        self.tools.planner_file_tools.prepare_campaign_trial(campaign_id)
        decision3 = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=resumed["trial_id"],
            metrics={"w1_scores": [0.8], "tmv_scores": [0.1], "custom_metrics": {"claim_metric": 0.4}},
        )
        self.assertEqual(decision3["decision"], "promote")
        self.assertEqual(decision3["active_best_trial_id"], resumed["trial_id"])

        gate_check = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(gate_check["ok"], gate_check)
        self.assertFalse(gate_check["advanced"], gate_check)
        self.assertEqual(gate_check["next_stage"], "stage2_claim_validation")
        self.assertEqual(
            json.loads(self.tools.get_algorithm_campaign_status(campaign_id))["current_stage"],
            "stage1_feasibility",
        )

        gate = json.loads(self.tools.check_campaign_stage_gate(campaign_id))
        self.assertTrue(gate["ok"], gate)
        self.assertTrue(gate["advanced"], gate)
        self.assertEqual(gate["next_stage"], "stage2_claim_validation")

        rev = subprocess.run(
            ["git", f"--git-dir={archive_path}", "rev-list", "--count", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertGreaterEqual(int(rev.stdout.strip()), 4)

    def test_campaign_trial_rejects_missing_target_dataset_metrics(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# candidate\n")
        self.tools.planner_file_tools.prepare_campaign_trial(
            campaign_id,
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "simulation", "adata_path": "/tmp/simulation.h5ad"},
                    {"dataset_id": "weinreb", "adata_path": "/tmp/weinreb.h5ad"},
                ],
            },
        )

        decision = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial["trial_id"],
            metrics={
                "w1_scores": [0.1],
                "tmv_scores": [0.1],
                "custom_metrics": {"claim_metric": 0.5},
                "target_dataset_ids": ["simulation", "weinreb"],
                "campaign_dataset_metrics": [
                    {
                        "campaign_dataset_id": "simulation",
                        "w1_scores": [0.1],
                        "tmv_scores": [0.1],
                        "custom_metrics": {"claim_metric": 0.5},
                    },
                    {"campaign_dataset_id": "weinreb", "run_id": "weinreb-run"},
                ],
            },
        )

        self.assertEqual(decision["decision"], "reject", decision)
        self.assertIn("target dataset 'weinreb' has no usable metrics", "; ".join(decision["decision_reasons"]))

    def test_run_campaign_trial_without_open_trial_keeps_current_workspace_edits(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    **self._claim_metric_spec(),
                    "stage_baselines": {
                        "stage1_feasibility": {"w1_mean": 1.0, "tmv_scores": [0.1]},
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]

        first = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# best\n")
        prepared1 = self.tools.planner_file_tools.prepare_campaign_trial(campaign_id)
        decision1 = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=first["trial_id"],
            metrics={"w1_scores": [0.9], "tmv_scores": [0.1], "custom_metrics": {"claim_metric": 0.5}},
        )
        self.assertEqual(decision1["decision"], "promote")
        self.assertEqual(self._readme_path().read_text(encoding="utf-8"), "# best\n")

        self._replace_readme("# candidate edit without explicit start\n")
        prepared2 = self.tools.planner_file_tools.prepare_campaign_trial(campaign_id)
        trial2 = dict(prepared2["trial"])

        self.assertTrue(trial2["started_from_current_workspace"])
        self.assertEqual(trial2["working_base_snapshot_id"], prepared1["snapshot_id"])
        self.assertEqual(trial2["restored_snapshot_id"], "")
        self.assertEqual(
            self._readme_path().read_text(encoding="utf-8"),
            "# candidate edit without explicit start\n",
        )

        decision2 = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial2["trial_id"],
            metrics={"w1_scores": [1.1], "tmv_scores": [0.1], "custom_metrics": {"claim_metric": 0.6}},
        )
        self.assertEqual(decision2["decision"], "reject")
        self.assertEqual(decision2["restored_snapshot_id"], prepared1["snapshot_id"])
        self.assertEqual(self._readme_path().read_text(encoding="utf-8"), "# best\n")

    def test_stage_panel_does_not_persist_trial_config_overrides(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    **self._claim_metric_spec(),
                    "stage_baselines": {
                        "stage1_feasibility": {"w1_mean": 1.0, "tmv_scores": [0.1]},
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        payload = {
            "datasets": [
                {
                    "dataset_id": "dataset_a",
                    "adata_path": "/tmp/dataset_a.h5ad",
                    "config_overrides": {"training.plan[0].lr": 0.123},
                }
            ],
            "config_overrides": {"model.hidden_dim": 64},
        }

        first_payload = self.tools.planner_file_tools.resolve_campaign_trial_dataset_payload(
            campaign_id,
            payload,
        )
        self.assertEqual(first_payload["config_overrides"]["model.hidden_dim"], 64)
        self.assertEqual(first_payload["datasets"][0]["config_overrides"]["training.plan[0].lr"], 0.123)

        frozen = self.tools.planner_file_tools.get_algorithm_campaign_status(campaign_id)
        panel_payload = frozen["stages"]["stage1_feasibility"]["stage_panel"]["dataset_config_overrides"]
        self.assertNotIn("config_overrides", panel_payload)
        self.assertNotIn("config_overrides", panel_payload["datasets"][0])

        reused_payload = self.tools.planner_file_tools.resolve_campaign_trial_dataset_payload(
            campaign_id,
            {},
        )
        self.assertEqual(reused_payload.get("config_overrides") or {}, {})
        self.assertEqual(reused_payload["datasets"][0].get("config_overrides") or {}, {})
        self.assertEqual(reused_payload["datasets"][0]["adata_path"], "/tmp/dataset_a.h5ad")

        next_payload = self.tools.planner_file_tools.resolve_campaign_trial_dataset_payload(
            campaign_id,
            {"config_overrides": {"model.hidden_dim": 96}},
        )
        self.assertEqual(next_payload["config_overrides"]["model.hidden_dim"], 96)
        self.assertNotIn("config_overrides", frozen["stages"]["stage1_feasibility"]["stage_panel"]["dataset_config_overrides"])

    def test_promoted_trial_override_syncs_to_workspace_but_rejected_override_does_not(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    **self._claim_metric_spec(),
                    "stage_baselines": {
                        "stage1_feasibility": {"w1_mean": 1.0, "tmv_scores": [0.1]},
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        config_path = Path(self.tmpdir.name) / "training_algorithms" / "algo_campaign" / "config.yaml"

        trial1 = json.loads(self.tools.start_campaign_trial(campaign_id))
        prepared1 = self.tools.planner_file_tools.prepare_campaign_trial(
            campaign_id,
            dataset_config_overrides={"config_overrides": {"model.hidden_dim": 64}},
        )
        resolved1 = Path(self.tmpdir.name) / "resolved_promote.yaml"
        resolved1.write_text("model:\n  hidden_dim: 64\ntraining:\n  batch_size: 32\n", encoding="utf-8")
        decision1 = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial1["trial_id"],
            metrics={
                "w1_scores": [0.9],
                "tmv_scores": [0.1],
                "custom_metrics": {"claim_metric": 0.5},
                "resolved_config_path": str(resolved1),
            },
        )
        self.assertEqual(decision1["decision"], "promote")
        synced_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(synced_config["model"]["hidden_dim"], 64)

        trial2 = json.loads(self.tools.start_campaign_trial(campaign_id))
        self.tools.planner_file_tools.prepare_campaign_trial(
            campaign_id,
            dataset_config_overrides={"config_overrides": {"model.hidden_dim": 128}},
        )
        resolved2 = Path(self.tmpdir.name) / "resolved_reject.yaml"
        resolved2.write_text("model:\n  hidden_dim: 128\ntraining:\n  batch_size: 32\n", encoding="utf-8")
        decision2 = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial2["trial_id"],
            metrics={
                "w1_scores": [1.1],
                "tmv_scores": [0.1],
                "custom_metrics": {"claim_metric": 0.6},
                "resolved_config_path": str(resolved2),
            },
        )
        self.assertEqual(decision2["decision"], "reject")
        restored_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(restored_config["model"]["hidden_dim"], 64)

    def test_run_campaign_trial_rejects_stale_sibling_campaign_when_active_bound(self) -> None:
        first = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    **self._claim_metric_spec(),
                    "stage_baselines": {
                        "stage1_feasibility": {"w1_mean": 1.0, "tmv_scores": [0.1]},
                    },
                },
            )
        )
        second = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    **self._claim_metric_spec(),
                    "stage_baselines": {
                        "stage1_feasibility": {"w1_mean": 1.0, "tmv_scores": [0.1]},
                    },
                },
            )
        )
        self.assertNotEqual(first["campaign_id"], second["campaign_id"])
        self.assertEqual(self.state["active_algorithm_campaign_id"], second["campaign_id"])

        with self.assertRaisesRegex(ValueError, "already bound to active campaign"):
            self.tools.planner_file_tools.start_campaign_trial(first["campaign_id"])
        with self.assertRaisesRegex(ValueError, "already bound to active campaign"):
            self.tools.run_campaign_trial(first["campaign_id"])

    def test_campaign_tools_are_registered(self) -> None:
        tool_names = {str(tool.name) for tool in self.tools.get_tools()}
        for name in {
            "start_algorithm_campaign",
            "get_algorithm_campaign_status",
            "start_campaign_trial",
            "run_campaign_trial",
            "resume_rejected_trial",
            "list_campaign_trials",
            "check_campaign_stage_gate",
            "register_stage2_simulation_dataset",
            "set_campaign_stage_panel",
            "refresh_campaign_stage_baselines",
            "run_campaign_control_baseline",
            "run_campaign_locked_algorithm_baseline",
            "update_campaign_control_baseline",
            "compute_campaign_claim_metric_for_baselines",
        }:
            self.assertIn(name, tool_names)
        self.assertNotIn("register_campaign_control_baseline", tool_names)

    def test_registered_control_baseline_raises_stage2_gate_bar(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator()
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign = self.tools.planner_file_tools.get_algorithm_campaign_status(campaign_id)
        campaign["current_stage"] = "stage2_claim_validation"
        stage2 = campaign["stages"]["stage2_claim_validation"]
        stage2["status"] = "active"
        stage2["trial_count"] = 1
        stage2["stage_panel"] = {
            "target_dataset_ids": ["dataset_a"],
            "dataset_config_overrides": {"target_dataset_ids": ["dataset_a"]},
        }
        stage2["external_baseline_metrics"] = {
            "algorithm_name": "builtin_reference",
            "target_dataset_ids": ["dataset_a"],
            "w1_mean": 0.70,
            "w1_backend": "exact",
            "w1_backend_exact": True,
            "w1_backend_params": {},
            "custom_metrics": {"claim_metric": 0.50},
            "claim_metric": 0.50,
            "claim_metric_evaluator": evaluator_info,
        }
        active_metrics = {
            "primary_metric": "claim_metric",
            "primary_direction": "greater",
            "primary_value": 0.80,
            "w1_mean": 0.60,
            "w1_backend": "exact",
            "w1_backend_exact": True,
            "w1_backend_params": {},
            "tmv_gate_required": False,
            "target_dataset_ids": ["dataset_a"],
            "custom_metrics": {"claim_metric": 0.80},
            "claim_metric": 0.80,
            "claim_metric_evaluator": evaluator_info,
        }
        campaign["trials"]["trial_stage2_active"] = {
            "trial_id": "trial_stage2_active",
            "stage": "stage2_claim_validation",
            "status": "completed",
            "decision": "promote",
            "metrics_summary": active_metrics,
        }
        stage2["active_best_trial_id"] = "trial_stage2_active"
        campaign["stages"]["stage2_claim_validation"] = stage2
        self.tools.planner_file_tools._save_campaign(campaign)

        gate_before = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(gate_before["ok"], gate_before)

        registered = json.loads(
            self.tools.register_campaign_control_baseline(
                campaign_id=campaign_id,
                baseline_id="shuffle_control_v1",
                baseline_name="shuffle_control",
                baseline_type="shuffle",
                stage="stage2_claim_validation",
                metrics={
                    "w1_mean": 0.65,
                    "w1_backend": "exact",
                    "w1_backend_exact": True,
                    "w1_backend_params": {},
                    "target_dataset_ids": ["dataset_a"],
                    "custom_metrics": {
                        "claim_metric": 0.75,
                        "nested_diagnostics": [{"sample": index} for index in range(500)],
                    },
                    "claim_metric": 0.75,
                    "claim_metric_evaluator": evaluator_info,
                    "tmv_gate_required": False,
                },
                target_dataset_ids=["dataset_a"],
                metric_roles=["both"],
                source_run_id="shuffle-run-1",
                source_metrics_path="/tmp/shuffle_metrics.json",
                control_code_source={
                    "kind": "config_overrides",
                    "config_overrides": {"data": {"shuffle_time_labels": True}},
                    "description": "shuffle temporal labels before the same campaign evaluator",
                },
                reason="prove the mechanism beats shuffled temporal labels",
            )
        )
        self.assertEqual(registered["status"], "registered")

        gate_after = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertFalse(gate_after["ok"], gate_after)
        primary = gate_after["checks"]["primary_vs_external_baseline"]
        self.assertEqual(primary["baseline_algorithm"], "shuffle_control_v1")
        self.assertEqual(primary["baseline"], 0.75)
        self.assertFalse(primary["ok"])
        self.assertIn(
            "active best does not satisfy the stage primary metric gate versus external baseline",
            gate_after["blockers"],
        )
        query = json.loads(
            self.tools.query_campaign_baseline_metrics(
                campaign_id=campaign_id,
                stage="stage2_claim_validation",
                include_training_run_scan=False,
                include_benchmark_registry=False,
            )
        )
        stage_payload = query["stages"]["stage2_claim_validation"]
        self.assertEqual(stage_payload["baseline_record_count"], 1)
        baseline_record = stage_payload["baseline_records"][0]
        self.assertEqual(baseline_record["source"], "agent_registered_control_baseline")
        self.assertEqual(baseline_record["w1_backend"], "exact")
        self.assertTrue(baseline_record["w1_backend_exact"])
        self.assertEqual(baseline_record["w1_backend_params"], {})
        self.assertEqual(baseline_record["custom_metrics"], {"claim_metric": 0.75})
        self.assertEqual(baseline_record["custom_metrics_omitted_detail_count"], 1)
        self.assertIn("nested_diagnostics", baseline_record["custom_metrics_omitted_detail_keys"])
        self.assertLess(len(json.dumps(query)), 20000)
        self.assertEqual(stage_payload["best_by_metric"]["claim_metric"]["algorithm_name"], "shuffle_control_v1")
        self.assertEqual(
            stage_payload["baseline_records"][0]["control_code_source"]["kind"],
            "config_overrides",
        )

        with self.assertRaisesRegex(ValueError, "already registered"):
            self.tools.planner_file_tools.register_campaign_control_baseline(
                campaign_id,
                baseline_id="shuffle_control_v1",
                baseline_name="shuffle_control",
                stage="stage2_claim_validation",
                metrics={"w1_mean": 0.5, "target_dataset_ids": ["dataset_a"]},
                target_dataset_ids=["dataset_a"],
                control_code_source={"kind": "config_overrides", "config_overrides": {"bad": True}},
            )

        deactivated = json.loads(
            self.tools.update_campaign_control_baseline(
                campaign_id=campaign_id,
                baseline_id="shuffle_control_v1",
                stage="stage2_claim_validation",
                action="deactivate",
                reason="first shuffle control patch was wrong",
            )
        )
        self.assertEqual(deactivated["baseline_status"], "inactive")
        gate_deactivated = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(gate_deactivated["ok"], gate_deactivated)

        replaced = json.loads(
            self.tools.update_campaign_control_baseline(
                campaign_id=campaign_id,
                baseline_id="shuffle_control_v1",
                stage="stage2_claim_validation",
                action="replace",
                metrics={
                    "w1_mean": 0.64,
                    "w1_backend": "exact",
                    "w1_backend_exact": True,
                    "w1_backend_params": {},
                    "target_dataset_ids": ["dataset_a"],
                    "custom_metrics": {"claim_metric": 0.77},
                    "claim_metric": 0.77,
                    "claim_metric_evaluator": evaluator_info,
                    "tmv_gate_required": False,
                },
                target_dataset_ids=["dataset_a"],
                source_run_id="shuffle-run-2",
                source_metrics_path="/tmp/shuffle_metrics_v2.json",
                control_code_source={
                    "kind": "config_overrides",
                    "config_overrides": {"data": {"shuffle_time_labels": "within_timepoint_pairs"}},
                    "description": "corrected shuffle control",
                },
                reason="rerun corrected shuffle ablation",
            )
        )
        self.assertEqual(replaced["baseline_status"], "active")
        self.assertEqual(replaced["revision"], 3)
        gate_replaced = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertFalse(gate_replaced["ok"], gate_replaced)
        replaced_primary = gate_replaced["checks"]["primary_vs_external_baseline"]
        self.assertEqual(replaced_primary["baseline_algorithm"], "shuffle_control_v1")
        self.assertEqual(replaced_primary["baseline"], 0.77)

        with self.assertRaisesRegex(ValueError, "lacks w1_backend"):
            self.tools.register_campaign_control_baseline(
                campaign_id=campaign_id,
                baseline_id="unproven_w1_control",
                baseline_name="unproven_w1_control",
                baseline_type="ablation",
                stage="stage2_claim_validation",
                metrics={
                    "w1_mean": 0.62,
                    "target_dataset_ids": ["dataset_a"],
                    "custom_metrics": {"claim_metric": 0.76},
                    "claim_metric": 0.76,
                    "claim_metric_evaluator": evaluator_info,
                    "tmv_gate_required": False,
                },
                target_dataset_ids=["dataset_a"],
                metric_roles=["both"],
                source_run_id="unproven-run",
                control_code_source={"kind": "config_overrides", "config_overrides": {"bad": True}},
                reason="missing W1 backend provenance should be rejected",
            )

    def test_run_campaign_control_baseline_registers_without_trial_decision(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator()
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign = self.tools.planner_file_tools.get_algorithm_campaign_status(campaign_id)
        campaign["current_stage"] = "stage2_claim_validation"
        stage2 = campaign["stages"]["stage2_claim_validation"]
        stage2["status"] = "active"
        stage2["stage_panel"] = {
            "target_dataset_ids": ["dataset_a"],
            "dataset_config_overrides": {
                "target_dataset_ids": ["dataset_a"],
                "datasets": [
                    {
                        "dataset_id": "dataset_a",
                        "adata_path": "/tmp/input.h5ad",
                        "config_overrides": {"training": {"batch_size": 8}},
                    }
                ],
            },
        }
        campaign["stages"]["stage2_claim_validation"] = stage2
        self.tools.planner_file_tools._save_campaign(campaign)

        calls = []

        def fake_run_training(**kwargs):
            calls.append(kwargs)
            self.state["latest_training_run_id"] = "control-run-1"
            return "control training ok"

        def fake_latest_metrics(run_id):
            self.assertEqual(run_id, "control-run-1")
            return {
                "run_id": run_id,
                "w1_scores": [0.61],
                "w1_mean": 0.61,
                "w1_backend": "exact",
                "w1_backend_exact": True,
                "w1_backend_params": {},
                "custom_metrics": {"claim_metric": 0.72},
                "claim_metric": 0.72,
                "claim_metric_evaluator": evaluator_info,
                "metrics_path": "/tmp/control_metrics.json",
                "resolved_config_path": "/tmp/control_resolved.yaml",
            }

        self.tools._run_training_tool_impl = fake_run_training
        self.tools._campaign_latest_training_metrics = fake_latest_metrics

        result = json.loads(
            self.tools.run_campaign_control_baseline(
                campaign_id=campaign_id,
                baseline_id="no_lag_control_v1",
                baseline_name="no_lag_control",
                baseline_type="ablation",
                stage="stage2_claim_validation",
                control_code_source={
                    "kind": "config_overrides",
                    "config_overrides": {"model": {"disable_lag": True}},
                },
                metric_roles=["claim"],
                reason="test no-lag control",
            )
        )
        self.assertEqual(result["status"], "registered")
        self.assertEqual(result["run_ids"], ["control-run-1"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["decision"], "provisional")
        self.assertEqual(calls[0]["review_purpose"], "campaign_control_baseline")
        self.assertTrue(calls[0]["_skip_review_gates"])
        self.assertEqual(calls[0]["config_overrides"]["model"]["disable_lag"], True)

        after = self.tools.planner_file_tools.get_algorithm_campaign_status(campaign_id)
        self.assertFalse(after.get("current_trial_id"))
        self.assertEqual(after["trials"], {})
        registered = after["stages"]["stage2_claim_validation"]["agent_registered_baselines"]
        self.assertEqual(len(registered), 1)
        self.assertEqual(registered[0]["baseline_id"], "no_lag_control_v1")
        self.assertEqual(registered[0]["control_code_source"]["kind"], "run_campaign_control_baseline")
        self.assertEqual(registered[0]["w1_backend"], "exact")
        self.assertTrue(registered[0]["w1_backend_exact"])

    def test_run_campaign_locked_algorithm_baseline_registers_completed_reference(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator()
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign = self.tools.planner_file_tools.get_algorithm_campaign_status(campaign_id)
        campaign["current_stage"] = "stage2_claim_validation"
        stage2 = campaign["stages"]["stage2_claim_validation"]
        stage2["status"] = "active"
        stage2["stage_panel"] = {
            "target_dataset_ids": ["dataset_a"],
            "dataset_config_overrides": {
                "target_dataset_ids": ["dataset_a"],
                "datasets": [
                    {
                        "dataset_id": "dataset_a",
                        "adata_path": "/tmp/input.h5ad",
                        "config_overrides": {"training": {"batch_size": 8}},
                    }
                ],
            },
        }
        campaign["stages"]["stage2_claim_validation"] = stage2
        self.tools.planner_file_tools._save_campaign(campaign)

        payload = _valid_proposal_payload()
        self.assertIn(
            "auto-approved",
            self.tools.create_algorithm_proposal(algorithm_id="locked_reference_algo", **payload),
        )
        self.assertIn(
            "Initialized training algorithm workspace",
            self.tools.init_training_algorithm_workspace("locked_reference_algo", description="reference test"),
        )
        ref_registry = self.tools.planner_file_tools._bootstrap_algorithm_registry("locked_reference_algo")
        ref_registry["algorithm_lifecycle_status"] = "complete"
        ref_registry["algorithm_lifecycle_status_reason"] = "test final_regression locked release"
        ref_registry["completed_campaign_id"] = "campaign_locked_reference"
        ref_registry["completed_release"] = {
            "trial_id": "trial_locked_reference",
            "snapshot_id": str(ref_registry.get("active_workspace_snapshot_id") or ""),
            "commit": "cafebabe",
            "locked_at": "2026-06-21T00:00:00Z",
        }
        self.tools.planner_file_tools._save_algorithm_registry("locked_reference_algo", ref_registry)

        adapter_script = Path(self.tmpdir.name) / "locked_reference_adapter.py"
        adapter_script.write_text(
            "import argparse\n"
            "from pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--input-adata')\n"
            "parser.add_argument('--output-adata')\n"
            "parser.add_argument('--dataset-id')\n"
            "parser.add_argument('--reference-algorithm-id')\n"
            "parser.add_argument('--current-algorithm-id')\n"
            "parser.add_argument('--context-json')\n"
            "args = parser.parse_args()\n"
            "Path(args.output_adata).write_text('adapted', encoding='utf-8')\n",
            encoding="utf-8",
        )
        claim_adapter_script = Path(self.tmpdir.name) / "locked_reference_claim_adapter.py"
        claim_adapter_script.write_text(
            "def adapt_baseline_metric_context(context):\n"
            "    return {'metadata': {'claim_adapter_seen': True}}\n",
            encoding="utf-8",
        )
        calls = []

        def fake_run_training(**kwargs):
            calls.append(kwargs)
            self.state["latest_training_run_id"] = "locked-reference-run-1"
            self.state["latest_training_algorithm_id"] = kwargs["training_algorithm_id"]
            return "locked reference training ok"

        def fake_latest_metrics(run_id):
            self.assertEqual(run_id, "locked-reference-run-1")
            _, run_evaluator_info = load_campaign_claim_metric_evaluator(
                calls[0]["config_overrides"]["__campaign_claim_metric_spec"],
                baseline_algorithm="locked_reference_algo",
            )
            return {
                "run_id": run_id,
                "w1_scores": [0.58],
                "w1_mean": 0.58,
                "w1_backend": "exact",
                "w1_backend_exact": True,
                "w1_backend_params": {},
                "custom_metrics": {"claim_metric": 0.74},
                "claim_metric": 0.74,
                "claim_metric_evaluator": run_evaluator_info,
                "metrics_path": "/tmp/locked_reference_metrics.json",
                "resolved_config_path": "/tmp/locked_reference_resolved.yaml",
            }

        self.tools._run_training_tool_impl = fake_run_training
        self.tools._campaign_latest_training_metrics = fake_latest_metrics

        result = json.loads(
            self.tools.run_campaign_locked_algorithm_baseline(
                campaign_id=campaign_id,
                reference_algorithm_id="locked_reference_algo",
                baseline_id="locked_reference_algo_v1",
                baseline_name="locked_reference_algo",
                stage="stage2_claim_validation",
                metric_roles=["both"],
                reference_dataset_adapter={
                    "kind": "python_script",
                    "script_path": str(adapter_script),
                    "reason": "explicit h5ad contract adapter for reference algorithm",
                },
                reference_claim_metric_adapter={
                    "kind": "claim_metric_adapter",
                    "adapter_path": str(claim_adapter_script),
                    "version": "test-v1",
                    "reason": "explicit claim context wrapper for reference algorithm",
                },
                reason="compare against a previous completed agent algorithm",
            )
        )
        self.assertEqual(result["status"], "registered")
        self.assertEqual(result["run_ids"], ["locked-reference-run-1"])
        self.assertEqual(result["reference_algorithm_id"], "locked_reference_algo")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["training_algorithm_id"], "locked_reference_algo")
        self.assertEqual(calls[0]["decision"], "provisional")
        self.assertEqual(calls[0]["review_purpose"], "campaign_locked_algorithm_baseline")
        self.assertTrue(calls[0]["_skip_review_gates"])
        self.assertFalse(calls[0]["_record_experiment_registry"])
        self.assertIn("locked_reference_adapters", calls[0]["adata_path"])
        claim_spec = calls[0]["config_overrides"]["__campaign_claim_metric_spec"]
        self.assertEqual(claim_spec["_baseline_algorithm_id"], "locked_reference_algo")
        self.assertIn("locked_reference_algo", claim_spec["baseline_metric_adapters"])
        self.assertEqual(
            claim_spec["baseline_metric_adapters"]["locked_reference_algo"]["adapter_path"],
            str(claim_adapter_script.resolve()),
        )

        after = self.tools.planner_file_tools.get_algorithm_campaign_status(campaign_id)
        self.assertFalse(after.get("current_trial_id"))
        self.assertEqual(after["trials"], {})
        registered = after["stages"]["stage2_claim_validation"]["agent_registered_baselines"]
        self.assertEqual(len(registered), 1)
        self.assertEqual(registered[0]["baseline_id"], "locked_reference_algo_v1")
        self.assertEqual(registered[0]["baseline_kind"], "agent_registered_control")
        self.assertTrue(registered[0]["metrics"]["locked_algorithm_reference"])
        self.assertEqual(len(registered[0]["metrics"]["reference_dataset_adapter_reports"]), 1)
        self.assertTrue(registered[0]["metrics"]["reference_claim_metric_adapter_report"]["adapter_applied"])
        self.assertEqual(registered[0]["control_code_source"]["kind"], "locked_algorithm_reference")
        self.assertEqual(registered[0]["control_code_source"]["reference_algorithm_id"], "locked_reference_algo")
        self.assertEqual(len(registered[0]["control_code_source"]["reference_dataset_adapter_reports"]), 1)
        self.assertTrue(registered[0]["control_code_source"]["reference_claim_metric_adapter_report"]["adapter_applied"])
        self.assertEqual(self.state.get("active_algorithm_campaign_id"), campaign_id)

    def test_locked_algorithm_baseline_rejects_missing_current_claim_metric_by_default(self) -> None:
        evaluator_path, _ = self._write_claim_metric_evaluator()
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign = self.tools.planner_file_tools.get_algorithm_campaign_status(campaign_id)
        campaign["current_stage"] = "stage2_claim_validation"
        stage2 = campaign["stages"]["stage2_claim_validation"]
        stage2["status"] = "active"
        stage2["stage_panel"] = {
            "target_dataset_ids": ["dataset_a"],
            "dataset_config_overrides": {
                "target_dataset_ids": ["dataset_a"],
                "datasets": [{"dataset_id": "dataset_a", "adata_path": "/tmp/input.h5ad"}],
            },
        }
        campaign["stages"]["stage2_claim_validation"] = stage2
        self.tools.planner_file_tools._save_campaign(campaign)

        payload = _valid_proposal_payload()
        self.assertIn(
            "auto-approved",
            self.tools.create_algorithm_proposal(algorithm_id="locked_reference_algo", **payload),
        )
        self.assertIn(
            "Initialized training algorithm workspace",
            self.tools.init_training_algorithm_workspace("locked_reference_algo", description="reference test"),
        )
        ref_registry = self.tools.planner_file_tools._bootstrap_algorithm_registry("locked_reference_algo")
        ref_registry["algorithm_lifecycle_status"] = "complete"
        ref_registry["completed_release"] = {
            "trial_id": "trial_locked_reference",
            "snapshot_id": str(ref_registry.get("active_workspace_snapshot_id") or ""),
            "locked_at": "2026-06-21T00:00:00Z",
        }
        self.tools.planner_file_tools._save_algorithm_registry("locked_reference_algo", ref_registry)

        def fake_run_training(**kwargs):
            self.state["latest_training_run_id"] = "locked-reference-run-claim-missing"
            return "locked reference training ok"

        def fake_latest_metrics(run_id):
            return {
                "run_id": run_id,
                "w1_scores": [0.58],
                "w1_mean": 0.58,
                "w1_backend": "exact",
                "w1_backend_exact": True,
                "w1_backend_params": {},
                "metrics_path": "/tmp/locked_reference_metrics.json",
            }

        self.tools._run_training_tool_impl = fake_run_training
        self.tools._campaign_latest_training_metrics = fake_latest_metrics

        result = json.loads(
            self.tools.run_campaign_locked_algorithm_baseline(
                campaign_id=campaign_id,
                reference_algorithm_id="locked_reference_algo",
                baseline_id="locked_reference_algo_v1",
                stage="stage2_claim_validation",
            )
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "locked_algorithm_claim_metric_incompatible")
        after = self.tools.planner_file_tools.get_algorithm_campaign_status(campaign_id)
        self.assertEqual(after["stages"]["stage2_claim_validation"].get("agent_registered_baselines") or [], [])

    def test_locked_algorithm_baseline_allows_w1_only_without_claim_metric(self) -> None:
        evaluator_path, _ = self._write_claim_metric_evaluator()
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign = self.tools.planner_file_tools.get_algorithm_campaign_status(campaign_id)
        campaign["current_stage"] = "stage2_claim_validation"
        stage2 = campaign["stages"]["stage2_claim_validation"]
        stage2["status"] = "active"
        stage2["stage_panel"] = {
            "target_dataset_ids": ["dataset_a"],
            "dataset_config_overrides": {
                "target_dataset_ids": ["dataset_a"],
                "datasets": [{"dataset_id": "dataset_a", "adata_path": "/tmp/input.h5ad"}],
            },
        }
        campaign["stages"]["stage2_claim_validation"] = stage2
        self.tools.planner_file_tools._save_campaign(campaign)

        payload = _valid_proposal_payload()
        self.assertIn(
            "auto-approved",
            self.tools.create_algorithm_proposal(algorithm_id="locked_reference_algo", **payload),
        )
        self.assertIn(
            "Initialized training algorithm workspace",
            self.tools.init_training_algorithm_workspace("locked_reference_algo", description="reference test"),
        )
        ref_registry = self.tools.planner_file_tools._bootstrap_algorithm_registry("locked_reference_algo")
        ref_registry["algorithm_lifecycle_status"] = "complete"
        ref_registry["completed_release"] = {
            "trial_id": "trial_locked_reference",
            "snapshot_id": str(ref_registry.get("active_workspace_snapshot_id") or ""),
            "locked_at": "2026-06-21T00:00:00Z",
        }
        self.tools.planner_file_tools._save_algorithm_registry("locked_reference_algo", ref_registry)

        def fake_run_training(**kwargs):
            self.state["latest_training_run_id"] = "locked-reference-run-w1-only"
            return "locked reference training ok"

        def fake_latest_metrics(run_id):
            return {
                "run_id": run_id,
                "w1_scores": [0.58],
                "w1_mean": 0.58,
                "w1_backend": "exact",
                "w1_backend_exact": True,
                "w1_backend_params": {},
                "metrics_path": "/tmp/locked_reference_metrics.json",
            }

        self.tools._run_training_tool_impl = fake_run_training
        self.tools._campaign_latest_training_metrics = fake_latest_metrics

        result = json.loads(
            self.tools.run_campaign_locked_algorithm_baseline(
                campaign_id=campaign_id,
                reference_algorithm_id="locked_reference_algo",
                baseline_id="locked_reference_algo_w1",
                stage="stage2_claim_validation",
                metric_roles=["w1"],
            )
        )
        self.assertEqual(result["status"], "registered", result)
        after = self.tools.planner_file_tools.get_algorithm_campaign_status(campaign_id)
        registered = after["stages"]["stage2_claim_validation"]["agent_registered_baselines"]
        self.assertEqual(len(registered), 1)
        self.assertEqual(registered[0]["metric_roles"], ["w1"])
        self.assertEqual(registered[0]["w1_backend"], "exact")
        self.assertTrue(registered[0]["w1_backend_exact"])

    def test_locked_algorithm_baseline_rejects_retuning_overrides(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign = self.tools.planner_file_tools.get_algorithm_campaign_status(campaign_id)
        campaign["current_stage"] = "stage2_claim_validation"
        stage2 = campaign["stages"]["stage2_claim_validation"]
        stage2["status"] = "active"
        stage2["stage_panel"] = {
            "target_dataset_ids": ["dataset_a"],
            "dataset_config_overrides": {
                "target_dataset_ids": ["dataset_a"],
                "datasets": [{"dataset_id": "dataset_a", "adata_path": "/tmp/input.h5ad"}],
            },
        }
        campaign["stages"]["stage2_claim_validation"] = stage2
        self.tools.planner_file_tools._save_campaign(campaign)

        payload = _valid_proposal_payload()
        self.assertIn(
            "auto-approved",
            self.tools.create_algorithm_proposal(algorithm_id="locked_reference_algo", **payload),
        )
        self.assertIn(
            "Initialized training algorithm workspace",
            self.tools.init_training_algorithm_workspace("locked_reference_algo", description="reference test"),
        )
        ref_registry = self.tools.planner_file_tools._bootstrap_algorithm_registry("locked_reference_algo")
        ref_registry["algorithm_lifecycle_status"] = "complete"
        ref_registry["completed_release"] = {
            "trial_id": "trial_locked_reference",
            "snapshot_id": str(ref_registry.get("active_workspace_snapshot_id") or ""),
            "locked_at": "2026-06-21T00:00:00Z",
        }
        self.tools.planner_file_tools._save_algorithm_registry("locked_reference_algo", ref_registry)

        with self.assertRaisesRegex(ValueError, "reference_config_overrides are disabled"):
            self.tools.run_campaign_locked_algorithm_baseline(
                campaign_id=campaign_id,
                reference_algorithm_id="locked_reference_algo",
                stage="stage2_claim_validation",
                reference_config_overrides={"training": {"plan[0].lr": 0.001}},
            )

    def test_campaign_status_summary_omits_full_trial_registry(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    **self._claim_metric_spec(),
                    "stage_baselines": {
                        "stage1_feasibility": {"w1_mean": 1.0, "tmv_scores": [0.1]},
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# status summary candidate\n")
        self.tools.planner_file_tools.prepare_campaign_trial(campaign_id)
        self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial["trial_id"],
            metrics={"w1_scores": [0.9], "tmv_scores": [0.1], "custom_metrics": {"claim_metric": 0.5}},
        )

        summary = json.loads(
            self.tools.get_algorithm_campaign_status(
                campaign_id,
                detail="summary",
                recent_trials_limit=1,
            )
        )
        self.assertEqual(summary["view"], "summary")
        self.assertEqual(summary["campaign_id"], campaign_id)
        self.assertNotIn("trials", summary)
        self.assertIn("stage_summaries", summary)
        self.assertEqual(len(summary["recent_trials"]), 1)
        self.assertEqual(summary["recent_trials"][0]["trial_id"], trial["trial_id"])
        self.assertIn("metrics_summary", summary["recent_trials"][0])
        self.assertIn("active_best_trial", summary)

        full = json.loads(self.tools.get_algorithm_campaign_status(campaign_id, detail="full"))
        self.assertIn("trials", full)
        self.assertIn(trial["trial_id"], full["trials"])

    def test_claim_metric_changes_trigger_inference_review(self) -> None:
        def metric_a(ctx):
            del ctx
            return {"claim_metric": 1.0}

        def metric_b(ctx):
            del ctx
            return {"claim_metric": 2.0}

        def make_target(metric):
            return SimpleNamespace(
                training_mode="custom",
                spec=SimpleNamespace(
                    algorithm_id="algo_campaign",
                    base_config={},
                    config_overrides={},
                    inference_context_builder=None,
                    simulation_hook=None,
                    evaluation_metrics_hook=metric,
                    evaluation_metrics_params={"expected_ranges": {"claim_metric": {"min": 0, "max": 1}}},
                ),
            )

        fp_a = self.tools._build_inference_review_fingerprint(make_target(metric_a), purpose="campaign")
        fp_b = self.tools._build_inference_review_fingerprint(make_target(metric_b), purpose="campaign")

        self.assertTrue(fp_a["requires_review"])
        self.assertTrue(fp_b["requires_review"])
        self.assertNotEqual(fp_a["review_hash"], fp_b["review_hash"])
        self.assertIn("evaluation_metrics_hook", fp_a["payload"]["callables"])
        self.assertTrue(fp_a["payload"]["has_evaluation_metrics_hook"])
        self.assertIn("evaluation_metrics_params", fp_a["payload"])

    def test_custom_inference_surface_still_triggers_inference_review(self) -> None:
        def build_inference_context(ctx):
            del ctx
            return None

        target = SimpleNamespace(
            training_mode="custom",
            spec=SimpleNamespace(
                algorithm_id="algo_campaign",
                base_config={},
                config_overrides={},
                inference_context_builder=build_inference_context,
                simulation_hook=None,
                evaluation_metrics_hook=None,
                evaluation_metrics_params={},
            ),
        )

        fp = self.tools._build_inference_review_fingerprint(target, purpose="campaign")

        self.assertTrue(fp["requires_review"])
        self.assertTrue(fp["payload"]["has_inference_context_builder"])
        self.assertIn("inference_context_builder", fp["payload"]["callables"])
        self.assertIn("evaluation_metrics_hook", fp["payload"]["callables"])
        self.assertFalse(fp["payload"]["has_evaluation_metrics_hook"])

    def test_campaign_load_refreshes_stale_state_summary_from_disk(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    **self._claim_metric_spec(),
                    "stage_baselines": {
                        "stage1_feasibility": {"w1_mean": 1.0, "tmv_scores": [0.1]},
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# campaign summary refresh\n")
        self.tools.planner_file_tools.prepare_campaign_trial(campaign_id)
        decision = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial["trial_id"],
            metrics={"w1_scores": [0.9], "tmv_scores": [0.1], "custom_metrics": {"claim_metric": 0.5}},
        )
        self.assertEqual(decision["decision"], "promote")
        gate = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(gate["ok"], gate)

        self.state["active_algorithm_campaign"] = {
            "campaign_id": campaign_id,
            "algorithm_id": "algo_campaign",
            "current_stage": "stage1_feasibility",
            "active_best_trial_id": "",
            "trial_count": 0,
            "gate_ready": False,
        }
        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        summary = dict(self.state.get("active_algorithm_campaign") or {})
        self.assertEqual(status["stages"]["stage1_feasibility"]["trial_count"], 1)
        self.assertEqual(summary.get("active_best_trial_id"), trial["trial_id"])
        self.assertEqual(summary.get("trial_count"), 1)
        self.assertTrue(summary.get("gate_ready"))
        self.assertTrue((summary.get("last_gate_check") or {}).get("ok"))

    def test_claim_metric_evaluator_loads_python_contract(self) -> None:
        evaluator_path = Path(self.tmpdir.name) / "claim_metric.py"
        evaluator_path.write_text(
            "def evaluate_claim_metric(context):\n"
            "    return {'claim_metric': float(len(context.timepoint_results))}\n",
            encoding="utf-8",
        )
        hook, info = load_campaign_claim_metric_evaluator(
            {
                "name": "claim_metric",
                "direction": "greater",
                "evaluator_path": f"{evaluator_path}:evaluate_claim_metric",
            }
        )
        self.assertIsNotNone(hook)
        self.assertEqual(info["name"], "claim_metric")
        self.assertEqual(info["function_name"], "evaluate_claim_metric")
        metrics = hook(SimpleNamespace(timepoint_results=[{"t": 1}, {"t": 2}]))
        self.assertEqual(metrics["claim_metric"], 2.0)
        self.assertEqual(metrics["claim_metric_evaluator_source"], str(evaluator_path.resolve()))

    def test_claim_metric_evaluator_supports_baseline_adapters(self) -> None:
        spec = {
            "name": "claim_metric",
            "direction": "greater",
            "evaluator_callable": lambda context: {"claim_metric": context.metric_params["scale"]},
            "baseline_metric_adapters": {
                "vgfm": {"metric_params": {"scale": 0.7}, "version": "adapter_v1"},
                "sf2m": {"supported": False, "reason": "requires a modality not emitted by this baseline"},
            },
        }
        hook, info = load_campaign_claim_metric_evaluator(spec, baseline_algorithm="vgfm")
        self.assertEqual(info["baseline_metric_adapter"]["baseline_algorithm"], "vgfm")
        metrics = hook(SimpleNamespace(metric_params={"scale": 0.1}, metadata={}))
        self.assertEqual(metrics["claim_metric"], 0.7)
        self.assertEqual(metrics["claim_metric_baseline_algorithm"], "vgfm")
        self.assertEqual(metrics["claim_metric_adapter_version"], "adapter_v1")

        unsupported_hook, _ = load_campaign_claim_metric_evaluator(spec, baseline_algorithm="sf2m")
        unsupported = unsupported_hook(SimpleNamespace(metric_params={}, metadata={}))
        self.assertEqual(unsupported["claim_metric_support_status"], "unsupported")
        self.assertIn("modality", unsupported["claim_metric_unsupported_reason"])

    def test_campaign_claim_metric_attach_uses_explicit_reference_baseline_id(self) -> None:
        training_target = SimpleNamespace(
            training_mode="custom",
            spec=SimpleNamespace(
                algorithm_id="locked_reference_algo",
                evaluation_metrics_hook=None,
                evaluation_metrics_params={},
            ),
        )
        info = self.tools._attach_campaign_claim_metric_evaluator(
            training_target,
            {
                "name": "claim_metric",
                "direction": "greater",
                "evaluator_callable": lambda context: {"claim_metric": context.metric_params["scale"]},
                "_baseline_algorithm_id": "locked_reference_algo",
                "baseline_metric_adapters": {
                    "locked_reference_algo": {
                        "metric_params": {"scale": 0.91},
                        "version": "locked-adapter-v1",
                    },
                },
            },
        )
        self.assertEqual(info["baseline_metric_adapter"]["baseline_algorithm"], "locked_reference_algo")
        metrics = training_target.spec.evaluation_metrics_hook(SimpleNamespace(metric_params={"scale": 0.1}, metadata={}))
        self.assertEqual(metrics["claim_metric"], 0.91)
        self.assertEqual(metrics["claim_metric_baseline_algorithm"], "locked_reference_algo")
        self.assertEqual(metrics["claim_metric_adapter_version"], "locked-adapter-v1")
        self.assertEqual(
            training_target.spec.evaluation_metrics_params["campaign_claim_metric_evaluator"]["baseline_metric_adapter"]["version"],
            "locked-adapter-v1",
        )

    def test_run_campaign_trial_aggregates_dataset_panel(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.start_campaign_trial(campaign_id)
        self._replace_readme("# panel candidate\n")
        self._fill_implementation_map()
        self._record_current_implementation_review(campaign_id)
        calls = []

        def fake_run_training_tool(**kwargs):
            calls.append(kwargs)
            idx = len(calls)
            run_id = f"run_panel_{idx}"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "w1_scores": [0.8 + idx * 0.01],
                        "tmv_scores": [0.1],
                        "custom_metrics": {"claim_metric": 0.4 + idx * 0.1},
                        "runtime_sec": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {
                    "run_id": run_id,
                    "metrics_path": str(metrics_path),
                    "run_dir": str(run_dir),
                    "status": "completed",
                }
            )
            self.state["latest_training_run_id"] = run_id
            return f"✅ TRAINING COMPLETED SUCCESSFULLY.\nRun ID: {run_id}"

        self.tools._run_training_tool_impl = fake_run_training_tool  # type: ignore[method-assign]
        result = json.loads(
            self.tools.run_campaign_trial(
                campaign_id,
                dataset_config_overrides={
                    "config_overrides": {"training": {"batch_size": 128}},
                    "datasets": [
                        {"dataset_id": "small", "adata_path": "/tmp/small.h5ad", "config_overrides": {"model": {"width": 32}}},
                        {"dataset_id": "guardrail", "adata_path": "/tmp/guardrail.h5ad"},
                    ],
                },
            )
        )

        self.assertEqual(result["decision"], "promote")
        self.assertEqual(result["target_dataset_ids"], ["small", "guardrail"])
        self.assertEqual(len(result["run_ids"]), 2)
        self.assertEqual(calls[0]["adata_path"], "/tmp/small.h5ad")
        self.assertEqual(calls[1]["adata_path"], "/tmp/guardrail.h5ad")
        self.assertEqual(calls[0]["review_purpose"], "campaign")
        self.assertEqual(calls[0]["review_campaign"]["campaign_id"], campaign_id)
        self.assertTrue(calls[0]["_skip_review_gates"])
        self.assertEqual(calls[0]["decision"], "provisional")
        self.assertAlmostEqual(result["metrics_summary"]["custom_metrics"]["claim_metric"], 0.55)
        self.assertEqual(result["metrics_summary"]["target_dataset_ids"], ["small", "guardrail"])
        self.assertEqual(result["metrics_summary"]["primary_metric"], "w1_mean")
        self.assertEqual(set(result["metrics_summary"]["per_dataset"].keys()), {"small", "guardrail"})
        self.assertAlmostEqual(result["metrics_summary"]["per_dataset"]["small"]["w1_mean"], 0.81)
        self.assertAlmostEqual(result["metrics_summary"]["per_dataset"]["guardrail"]["custom_metrics"]["claim_metric"], 0.6)
        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(status["stages"]["stage1_feasibility"]["stage_panel"]["target_dataset_ids"], ["small", "guardrail"])
        self.assertEqual(status["stage_internal_status"], "active_best_promoted")
        self.assertEqual(status["stage_gate_status"], "blocked_missing_external_baseline")

        self.tools.start_campaign_trial(campaign_id)
        self._replace_readme("# wrong panel\n")
        with self.assertRaisesRegex(ValueError, "panel is already frozen"):
            self.tools.run_campaign_trial(
                campaign_id,
                dataset_config_overrides={
                    "datasets": [{"dataset_id": "other", "adata_path": "/tmp/other.h5ad"}],
                },
            )

    def test_run_campaign_trial_blocks_stage2_claim_metric_without_evaluator(self) -> None:
        with self.assertRaisesRegex(ValueError, "claim_metric_spec.evaluator_path"):
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={"name": "claim_metric", "direction": "greater"},
            )

    def test_run_campaign_trial_injects_stage2_claim_metric_evaluator_spec(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.7)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "claim_sim", "adata_path": "/tmp/claim_sim.h5ad"},
                ],
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        self.tools.planner_file_tools._save_campaign(campaign)
        self._fill_implementation_map()
        self._record_current_implementation_review(campaign_id)
        calls = []

        def fake_run_training_tool(**kwargs):
            calls.append(kwargs)
            run_id = "run_stage2_claim"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "w1_scores": [0.9],
                        "tmv_scores": [0.1],
                        "custom_metrics": {"claim_metric": 0.7},
                        "claim_metric_evaluator": dict(evaluator_info),
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {"run_id": run_id, "metrics_path": str(metrics_path), "run_dir": str(run_dir), "status": "completed"}
            )
            self.state["latest_training_run_id"] = run_id
            return f"✅ TRAINING COMPLETED SUCCESSFULLY.\nRun ID: {run_id}"

        self.tools._run_training_tool_impl = fake_run_training_tool  # type: ignore[method-assign]
        result = json.loads(self.tools.run_campaign_trial(campaign_id))

        self.assertEqual(result["decision"], "promote")
        self.assertEqual(len(calls), 1)
        injected = calls[0]["config_overrides"]["__campaign_claim_metric_spec"]
        self.assertEqual(injected["evaluator_path"], str(evaluator_path))
        self.assertEqual(result["metrics_summary"]["claim_metric_evaluator"], evaluator_info)

    def test_run_campaign_trial_merges_config_only_payload_onto_frozen_panel(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.start_campaign_trial(campaign_id)
        self._replace_readme("# frozen panel config tuning\n")
        self._fill_implementation_map()
        self._record_current_implementation_review(campaign_id)
        self.tools.set_campaign_stage_panel(
            campaign_id,
            dataset_config_overrides={
                "config_overrides": {"training": {"batch_size": 128}},
                "datasets": [
                    {"dataset_id": "small", "adata_path": "/tmp/small.h5ad"},
                    {"dataset_id": "guardrail", "adata_path": "/tmp/guardrail.h5ad"},
                ],
            },
        )
        calls = []

        def fake_run_training_tool(**kwargs):
            calls.append(kwargs)
            idx = len(calls)
            run_id = f"run_config_only_{idx}"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "w1_scores": [0.8],
                        "tmv_scores": [0.1],
                        "custom_metrics": {"claim_metric": 0.5},
                        "runtime_sec": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {"run_id": run_id, "metrics_path": str(metrics_path), "run_dir": str(run_dir), "status": "completed"}
            )
            self.state["latest_training_run_id"] = run_id
            return f"✅ TRAINING COMPLETED SUCCESSFULLY.\nRun ID: {run_id}"

        self.tools._run_training_tool_impl = fake_run_training_tool  # type: ignore[method-assign]
        result = json.loads(
            self.tools.run_campaign_trial(
                campaign_id,
                dataset_config_overrides={
                    "config_overrides": {"training": {"batch_size": 256, "lr": 0.001}},
                },
            )
        )

        self.assertEqual(result["decision"], "promote")
        self.assertEqual([call["adata_path"] for call in calls], ["/tmp/small.h5ad", "/tmp/guardrail.h5ad"])
        for call in calls:
            self.assertEqual(call["config_overrides"]["training"]["batch_size"], 256)
            self.assertEqual(call["config_overrides"]["training"]["lr"], 0.001)
            self.assertNotIn("config_overrides", call["config_overrides"])

    def test_run_campaign_trial_accepts_common_and_per_dataset_override_aliases(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.start_campaign_trial(campaign_id)
        self._replace_readme("# alias panel config tuning\n")
        self._fill_implementation_map()
        self._record_current_implementation_review(campaign_id)
        self.tools.set_campaign_stage_panel(
            campaign_id,
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "small", "adata_path": "/tmp/small.h5ad"},
                    {"dataset_id": "guardrail", "adata_path": "/tmp/guardrail.h5ad"},
                ],
            },
        )
        calls = []

        def fake_run_training_tool(**kwargs):
            calls.append(kwargs)
            idx = len(calls)
            run_id = f"run_alias_{idx}"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "w1_scores": [0.8],
                        "tmv_scores": [0.1],
                        "custom_metrics": {"claim_metric": 0.5},
                        "runtime_sec": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {"run_id": run_id, "metrics_path": str(metrics_path), "run_dir": str(run_dir), "status": "completed"}
            )
            self.state["latest_training_run_id"] = run_id
            return f"✅ TRAINING COMPLETED SUCCESSFULLY.\nRun ID: {run_id}"

        self.tools._run_training_tool_impl = fake_run_training_tool  # type: ignore[method-assign]
        self.tools.run_campaign_trial(
            campaign_id,
            dataset_config_overrides={
                "common_config_overrides": {"training": {"batch_size": 512}},
                "per_dataset_config_overrides": {
                    "small": {"model": {"width": 64}},
                    "guardrail": {"training.plan[0].lr": 0.002},
                },
            },
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["config_overrides"]["training"]["batch_size"], 512)
        self.assertEqual(calls[0]["config_overrides"]["model"]["width"], 64)
        self.assertEqual(calls[1]["config_overrides"]["training"]["batch_size"], 512)
        self.assertEqual(calls[1]["config_overrides"]["training.plan[0].lr"], 0.002)
        for call in calls:
            self.assertNotIn("common_config_overrides", call["config_overrides"])
            self.assertNotIn("per_dataset_config_overrides", call["config_overrides"])

    def test_training_config_overrides_support_bracket_path_notation(self) -> None:
        config = {"training": {"plan": [{"lr": 0.01, "epochs": 3000}]}}
        updated = apply_overrides(config, {"training.plan[0].lr": 0.002})

        self.assertEqual(updated["training"]["plan"][0]["lr"], 0.002)
        self.assertEqual(updated["training"]["plan"][0]["epochs"], 3000)

    def test_training_config_overrides_preserve_plan_entry_when_nested(self) -> None:
        config = {
            "training": {
                "plan": [
                    {
                        "name": "Train_CommWFRFM",
                        "mode": "flow_matching",
                        "epochs": 3000,
                        "lr": 0.01,
                        "delta": 0.7,
                    }
                ]
            }
        }
        updated = apply_overrides(
            config,
            {
                "training": {
                    "plan": [
                        {
                            "lr": 0.002,
                            "delta": 15,
                            "flow_matching": {"coupling": {"delta": 15}},
                        }
                    ]
                }
            },
        )

        stage = updated["training"]["plan"][0]
        self.assertEqual(stage["name"], "Train_CommWFRFM")
        self.assertEqual(stage["mode"], "flow_matching")
        self.assertEqual(stage["epochs"], 3000)
        self.assertEqual(stage["lr"], 0.002)
        self.assertEqual(stage["delta"], 15)
        self.assertEqual(stage["flow_matching"]["coupling"]["delta"], 15)

    def test_epoch_policy_allows_redundant_default_epoch_declaration(self) -> None:
        config = {"training": {"plan": [{"lr": 0.01, "epochs": 3000}]}}

        self.assertEqual(
            validate_epoch_override_policy(
                base_config=config,
                overrides={"training": {"plan": [{"epochs": 3000}], "epochs": 3000}},
            ),
            [],
        )

    def test_epoch_policy_blocks_effective_epoch_change_without_reason(self) -> None:
        config = {"training": {"plan": [{"lr": 0.01, "epochs": 3000}]}}

        blocked = validate_epoch_override_policy(
            base_config=config,
            overrides={"training.plan[0].epochs": 100},
        )

        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0]["path"], "training.plan.0.epochs")
        self.assertEqual(blocked[0]["old_value"], 3000)
        self.assertEqual(blocked[0]["new_value"], 100)

    def test_config_override_pruning_keeps_only_changed_leaf_values(self) -> None:
        config = {
            "training": {"plan": [{"lr": 0.01, "epochs": 3000}]},
            "model": {"hidden_dims": [256, 256]},
        }

        sparse, removed = prune_redundant_config_overrides(
            base_config=config,
            overrides={
                "training": {
                    "plan": [{"lr": 0.002, "epochs": 3000}],
                    "epochs": 3000,
                },
                "model": {"hidden_dims": [256, 256]},
            },
        )

        self.assertEqual(sparse, {"training.plan.0.lr": 0.002})
        self.assertTrue(any(item["path"] == "training.plan.0.epochs" for item in removed))
        self.assertTrue(any(item["path"] == "training.epochs" for item in removed))

    def test_make_benchmark_dataset_config_loads_dataset_trial_config(self) -> None:
        self._write_benchmark_dataset_card("weinreb", "/tmp/weinreb.h5ad")
        dataset_json_path = Path(self.tmpdir.name) / "algorithm_benchmarks" / "datasets" / "weinreb" / "dataset.json"
        card = json.loads(dataset_json_path.read_text(encoding="utf-8"))
        card["usage"]["campaign_dataset_entry"]["config_overrides"] = {
            "data": {"normalize": True},
            "training": {"batch_size": 32},
        }
        dataset_json_path.write_text(json.dumps(card), encoding="utf-8")
        config_dir = Path(self.tmpdir.name) / "algorithm_benchmarks" / "datasets" / "weinreb" / "trial_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "stage1_feasibility.yaml"
        config_path.write_text(
            "schema_version: 1\n"
            "dataset_id: weinreb\n"
            "stage: stage1_feasibility\n"
            "config_overrides:\n"
            "  model:\n"
            "    delta: 15\n"
            "  training:\n"
            "    batch_size: 64\n",
            encoding="utf-8",
        )

        payload = self.tools.planner_file_tools.make_benchmark_dataset_config(
            ["weinreb"],
            stage="stage1_feasibility",
            per_dataset_config_overrides={"weinreb": {"training": {"batch_size": 128}}},
        )

        entry = payload["dataset_config_overrides"]["datasets"][0]
        self.assertTrue(entry["config_overrides"]["data"]["normalize"])
        self.assertEqual(entry["config_overrides"]["model"]["delta"], 15)
        self.assertEqual(entry["config_overrides"]["training"]["batch_size"], 128)
        self.assertEqual(entry["dataset_trial_config_path"], str(config_path))

    def test_stage_panel_direct_payload_loads_dataset_trial_config(self) -> None:
        self._write_benchmark_dataset_card("weinreb", "/tmp/weinreb.h5ad")
        config_dir = Path(self.tmpdir.name) / "algorithm_benchmarks" / "datasets" / "weinreb" / "trial_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "stage1_feasibility.yaml"
        config_path.write_text(
            "config_overrides:\n"
            "  model:\n"
            "    delta: 15\n"
            "  training:\n"
            "    batch_size: 64\n",
            encoding="utf-8",
        )
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )

        panel = self.tools.planner_file_tools.set_campaign_stage_panel(
            campaign["campaign_id"],
            dataset_config_overrides={
                "datasets": [
                    {
                        "dataset_id": "weinreb",
                        "adata_path": "/tmp/weinreb.h5ad",
                        "config_overrides": {"training": {"batch_size": 128}},
                    }
                ],
            },
        )

        entry = panel["stage_panel"]["dataset_config_overrides"]["datasets"][0]
        self.assertEqual(entry["config_overrides"]["model"]["delta"], 15)
        self.assertEqual(entry["config_overrides"]["training"]["batch_size"], 64)
        self.assertEqual(entry["dataset_trial_config_path"], str(config_path))

    def test_promoted_campaign_trial_syncs_resolved_config_to_workspace(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# candidate\n")
        prepared = self.tools.planner_file_tools.prepare_campaign_trial(campaign_id)
        resolved_config_path = Path(self.tmpdir.name) / "resolved_config.yaml"
        resolved_config_path.write_text(
            "training:\n"
            "  batch_size: 96\n"
            "model:\n"
            "  width: 48\n",
            encoding="utf-8",
        )

        decision = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial["trial_id"],
            metrics={
                "w1_scores": [0.8],
                "tmv_scores": [0.1],
                "custom_metrics": {"claim_metric": 0.5},
                "resolved_config_path": str(resolved_config_path),
            },
        )

        self.assertEqual(decision["decision"], "promote")
        self.assertEqual(decision["promoted_config_sync"]["status"], "synced")
        self.assertNotEqual(decision["promoted_config_sync"]["snapshot_id"], "")
        self.assertNotEqual(decision["promoted_config_sync"]["snapshot_id"], prepared["snapshot_id"])
        workspace_config = Path(self.tmpdir.name) / "training_algorithms" / "algo_campaign" / "config.yaml"
        self.assertEqual(yaml.safe_load(workspace_config.read_text(encoding="utf-8"))["model"]["width"], 48)
        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id, detail="full"))
        self.assertEqual(
            status["stages"]["stage1_feasibility"]["active_best_snapshot_id"],
            decision["promoted_config_sync"]["snapshot_id"],
        )

    def test_multi_dataset_promote_does_not_sync_partial_resolved_config(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# partial resolved config candidate\n")
        prepared = self.tools.planner_file_tools.prepare_campaign_trial(
            campaign_id,
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "simulation", "adata_path": "/tmp/simulation.h5ad"},
                    {"dataset_id": "weinreb", "adata_path": "/tmp/weinreb.h5ad"},
                ]
            },
        )
        resolved_config_path = Path(self.tmpdir.name) / "partial_resolved_config.yaml"
        resolved_config_path.write_text("training:\n  batch_size: 96\n", encoding="utf-8")
        before_config = (
            Path(self.tmpdir.name) / "training_algorithms" / "algo_campaign" / "config.yaml"
        ).read_text(encoding="utf-8")

        decision = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial["trial_id"],
            metrics={
                "w1_scores": [0.8, 0.82],
                "tmv_scores": [0.1, 0.1],
                "custom_metrics": {"claim_metric": 0.5},
                "target_dataset_ids": ["simulation", "weinreb"],
                "campaign_dataset_metrics": [
                    {
                        "campaign_dataset_id": "simulation",
                        "w1_scores": [0.8],
                        "tmv_scores": [0.1],
                        "resolved_config_path": str(resolved_config_path),
                        "custom_metrics": {"claim_metric": 0.5},
                    },
                    {
                        "campaign_dataset_id": "weinreb",
                        "w1_scores": [0.82],
                        "tmv_scores": [0.1],
                        "custom_metrics": {"claim_metric": 0.5},
                    },
                ],
            },
        )

        self.assertEqual(decision["decision"], "promote")
        self.assertEqual(decision["promoted_config_sync"]["status"], "skipped_partial_resolved_config_paths")
        self.assertEqual(decision["promoted_config_sync"]["snapshot_id"], "")
        self.assertEqual(
            (Path(self.tmpdir.name) / "training_algorithms" / "algo_campaign" / "config.yaml").read_text(encoding="utf-8"),
            before_config,
        )
        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id, detail="full"))
        self.assertEqual(
            status["stages"]["stage1_feasibility"]["active_best_snapshot_id"],
            prepared["snapshot_id"],
        )

    def test_run_campaign_trial_blocks_without_counting_when_training_does_not_launch(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "small", "adata_path": "/tmp/small.h5ad"},
                ],
            },
        )
        trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._fill_implementation_map()
        self._record_current_implementation_review(campaign_id)

        def fake_run_training_tool(**kwargs):  # noqa: ANN003
            self.assertEqual(kwargs["review_purpose"], "campaign")
            self.assertTrue(kwargs["_skip_review_gates"])
            return (
                "Inference review required before this custom training run can be trusted. "
                "The runtime will start a read-only inference evaluator subagent now."
            )

        self.tools._run_training_tool_impl = fake_run_training_tool  # type: ignore[method-assign]
        result = json.loads(self.tools.run_campaign_trial(campaign_id))

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "training_not_launched")
        self.assertEqual(result["trial_id"], trial["trial_id"])
        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(status["current_trial_id"], "")
        self.assertEqual(status["trials"][trial["trial_id"]]["status"], "blocked_before_training")
        repeated_start = json.loads(self.tools.start_campaign_trial(campaign_id))
        self.assertNotEqual(repeated_start["trial_id"], trial["trial_id"])
        self.assertEqual(repeated_start["status"], "editing")
        stage = status["stages"]["stage1_feasibility"]
        self.assertEqual(stage.get("trial_count", 0), 0)
        self.assertEqual(stage.get("reject_count", 0), 0)

    def test_run_campaign_trial_requires_stage_panel_or_dataset_payload(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        self._fill_implementation_map()
        self._record_current_implementation_review(campaign_id)

        with self.assertRaisesRegex(ValueError, "no frozen dataset panel"):
            self.tools.run_campaign_trial(campaign_id)

    def test_run_campaign_trial_rejects_json_string_dataset_payload(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]

        with self.assertRaisesRegex(ValueError, "must be a JSON object/dict"):
            self.tools.run_campaign_trial(
                campaign_id,
                dataset_config_overrides='{"datasets":[{"dataset_id":"small"}]}',  # type: ignore[arg-type]
            )

    def test_stage_gate_requires_matching_stage_panel_baseline(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    **self._claim_metric_spec(),
                    "stage_baselines": {
                        "stage1_feasibility": {"w1_mean": 1.0, "tmv_scores": [0.1]},
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# fixed panel gate\n")
        self.tools.planner_file_tools.prepare_campaign_trial(
            campaign_id,
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "small", "adata_path": "/tmp/small.h5ad"},
                    {"dataset_id": "guardrail", "adata_path": "/tmp/guardrail.h5ad"},
                ],
            },
        )
        self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial["trial_id"],
            metrics={"w1_scores": [0.9], "tmv_scores": [0.1], "custom_metrics": {"claim_metric": 0.5}},
        )

        missing_panel = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertFalse(missing_panel["ok"], missing_panel)
        self.assertTrue(any("target_dataset_ids" in item for item in missing_panel["blockers"]))
        status_after_block = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(status_after_block["stage_internal_status"], "active_best_promoted")
        self.assertEqual(status_after_block["stage_gate_status"], "blocked_panel_mismatch")
        self.assertIn("trial promoted internally", status_after_block["user_facing_status"])

        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["stages"]["stage1_feasibility"]["external_baseline_metrics"]["target_dataset_ids"] = [
            "small",
            "guardrail",
        ]
        self.tools.planner_file_tools._save_campaign(campaign)

        matching_panel = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(matching_panel["ok"], matching_panel)
        self.assertTrue(matching_panel["checks"]["stage_panel"]["ok"])

    def test_refresh_campaign_stage_baselines_runs_panel_and_updates_gate(self) -> None:
        self._write_benchmark_dataset_card("small", "/tmp/small.h5ad")
        self._write_benchmark_dataset_card("guardrail", "/tmp/guardrail.h5ad")
        evaluator_path, _ = self._write_claim_metric_evaluator()
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# fixed panel baseline refresh\n")
        self.tools.planner_file_tools.prepare_campaign_trial(
            campaign_id,
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "small", "adata_path": "/tmp/small.h5ad"},
                    {"dataset_id": "guardrail", "adata_path": "/tmp/guardrail.h5ad"},
                ],
            },
        )
        self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial["trial_id"],
            metrics={"w1_scores": [0.9], "tmv_scores": [0.1], "custom_metrics": {"claim_metric": 0.5}},
        )
        calls = []

        def fake_run_training_tool(**kwargs):
            calls.append(kwargs)
            algorithm = str(kwargs.get("candidate_name") or "")
            w1_by_algorithm = {
                "vgfm": 0.5,
                "dynamical_ot": 1.0,
                "sf2m": 5.0,
            }
            w1 = w1_by_algorithm.get(algorithm, 2.0)
            run_id = f"run_{algorithm}_{len(calls)}"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "w1_scores": [w1],
                        "tmv_scores": [0.1],
                        "runtime_sec": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {
                    "run_id": run_id,
                    "metrics_path": str(metrics_path),
                    "run_dir": str(run_dir),
                    "status": "completed",
                }
            )
            self.state["latest_training_run_id"] = run_id
            return f"✅ TRAINING COMPLETED SUCCESSFULLY.\nRun ID: {run_id}"

        self.tools._run_training_tool_impl = fake_run_training_tool  # type: ignore[method-assign]
        self._set_campaign_baseline_selection_policy(campaign_id, "permissive")
        refresh = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                baseline_algorithms=["vgfm", "dynamical_ot", "sf2m"],
            )
        )

        self.assertEqual(refresh["status"], "updated")
        self.assertEqual(len(calls), 6)
        self.assertEqual(refresh["selected_baseline"]["algorithm_name"], "dynamical_ot")
        self.assertEqual(refresh["selected_baseline"]["selection_rule"], "second_weakest_w1_baseline_for_feasibility")
        self.assertEqual(refresh["selected_baseline"]["selection_candidate_count"], 3)
        self.assertEqual(refresh["selected_baseline"]["target_dataset_ids"], ["small", "guardrail"])

        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        baseline = status["stages"]["stage1_feasibility"]["external_baseline_metrics"]
        self.assertEqual(baseline["algorithm_name"], "dynamical_ot")
        self.assertEqual(baseline["target_dataset_ids"], ["small", "guardrail"])
        self.assertAlmostEqual(baseline["w1_mean"], 1.0)

        gate = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(gate["ok"], gate)
        self.assertTrue(gate["checks"]["stage_panel"]["ok"])

    def test_refresh_campaign_stage_baselines_applies_dataset_builtin_config(self) -> None:
        self._write_benchmark_dataset_card("weinreb_like", "/tmp/weinreb.h5ad")
        dataset_dir = Path(self.tmpdir.name) / "algorithm_benchmarks" / "datasets" / "weinreb_like"
        config_path = dataset_dir / "builtin_configs" / "builtin_wfrfm.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "schema_version: 1",
                    "algorithm_name: wfrfm",
                    "baseline_type: builtin",
                    "config_overrides:",
                    "  training.defaults.delta: 15",
                    "  training.defaults.chunk_size: 2000",
                    "  training.plan[0].delta: 15",
                    "  training.plan[0].flow_matching.coupling.delta: 15",
                    "  training.plan[0].flow_matching.coupling.chunk_size: 2000",
                    "  training.plan[0].flow_matching.path.delta: 15",
                ]
            ),
            encoding="utf-8",
        )
        evaluator_path, _ = self._write_claim_metric_evaluator()
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            dataset_config_overrides={
                "datasets": [
                    {
                        "dataset_id": "weinreb_like",
                        "adata_path": "/tmp/weinreb.h5ad",
                        "config_overrides": {
                            "training.defaults.delta": 1.5,
                            "training.plan[0].delta": 1.5,
                        },
                    },
                ],
            },
        )
        calls = []

        def fake_run_training_tool(**kwargs):
            calls.append(kwargs)
            run_id = f"run_wfrfm_{len(calls)}"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "w1_scores": [6.0],
                        "tmv_scores": [0.1],
                        "runtime_sec": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {
                    "run_id": run_id,
                    "metrics_path": str(metrics_path),
                    "run_dir": str(run_dir),
                    "status": "completed",
                }
            )
            self.state["latest_training_run_id"] = run_id
            return f"Run ID: {run_id}"

        self.tools._run_training_tool_impl = fake_run_training_tool  # type: ignore[method-assign]
        self._set_campaign_baseline_selection_policy(campaign_id, "permissive")
        refresh = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                baseline_algorithms=["wfrfm"],
                overwrite_existing=True,
            )
        )

        self.assertEqual(refresh["status"], "updated")
        self.assertEqual(len(calls), 1)
        overrides = calls[0]["config_overrides"]
        self.assertEqual(overrides["training.defaults.delta"], 15)
        self.assertEqual(overrides["training.plan[0].delta"], 15)
        self.assertEqual(overrides["training.plan[0].flow_matching.coupling.delta"], 15)
        self.assertEqual(overrides["training.plan[0].flow_matching.path.delta"], 15)
        self.assertEqual(overrides["training.defaults.chunk_size"], 2000)
        self.assertEqual(
            refresh["run_records"][0]["benchmark_baseline_config_path"],
            str(config_path),
        )

    def test_refresh_campaign_stage_baselines_defaults_to_all_builtin_candidates(self) -> None:
        self._write_benchmark_dataset_card("small", "/tmp/small.h5ad")
        evaluator_path, _ = self._write_claim_metric_evaluator()
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "small", "adata_path": "/tmp/small.h5ad"},
                ],
            },
        )
        calls = []

        def fake_run_training_tool(**kwargs):
            calls.append(kwargs)
            algorithm = str(kwargs.get("candidate_name") or "")
            run_id = f"run_{algorithm}_{len(calls)}"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "w1_scores": [1.0],
                        "tmv_scores": [0.1],
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {
                    "run_id": run_id,
                    "metrics_path": str(metrics_path),
                    "run_dir": str(run_dir),
                    "status": "completed",
                }
            )
            self.state["latest_training_run_id"] = run_id
            return f"✅ TRAINING COMPLETED SUCCESSFULLY.\nRun ID: {run_id}"

        self.tools.run_training_tool = fake_run_training_tool
        refresh = json.loads(self.tools.refresh_campaign_stage_baselines(campaign_id))

        self.assertEqual(refresh["status"], "updated")
        self.assertEqual(refresh["baseline_algorithms"], list(planner_tools_module.DEFAULT_CAMPAIGN_BUILTIN_BASELINES))
        self.assertEqual(len(calls), len(planner_tools_module.DEFAULT_CAMPAIGN_BUILTIN_BASELINES))

    def test_refresh_stage2_strict_blocks_interrupted_required_builtin(self) -> None:
        self._write_benchmark_dataset_card("small", "/tmp/small.h5ad")
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator()
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "small", "adata_path": "/tmp/small.h5ad"},
                ],
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        self.tools.planner_file_tools._save_campaign(campaign)
        calls = []

        def fake_run_training_tool(**kwargs):
            calls.append(kwargs)
            algorithm = str(kwargs.get("candidate_name") or "")
            run_id = f"run_{algorithm}_{len(calls)}"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            if algorithm == "dynamical_ot":
                (run_dir / "run_manifest.json").write_text(
                    json.dumps({"run_id": run_id, "status": "running"}),
                    encoding="utf-8",
                )
                self.state["training_runs"].append(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "status": "running",
                    }
                )
                self.state["latest_training_run_id"] = run_id
                return f"Training started but did not complete.\nRun ID: {run_id}"
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "w1_scores": [1.0],
                        "tmv_scores": [0.1],
                        "custom_metrics": {"claim_metric": 0.5},
                        "claim_metric": 0.5,
                        "claim_metric_evaluator": evaluator_info,
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {
                    "run_id": run_id,
                    "metrics_path": str(metrics_path),
                    "run_dir": str(run_dir),
                    "status": "completed",
                }
            )
            self.state["latest_training_run_id"] = run_id
            return f"✅ TRAINING COMPLETED SUCCESSFULLY.\nRun ID: {run_id}"

        self.tools.run_training_tool = fake_run_training_tool
        refresh = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                stage="stage2_claim_validation",
                overwrite_existing=True,
            )
        )

        self.assertEqual(refresh["status"], "blocked")
        self.assertEqual(refresh["reason"], "strict_required_baseline_missing_records")
        self.assertEqual(refresh["baseline_selection_policy"], "strict_all_builtin")
        self.assertEqual(refresh["strict_required_baselines"], list(planner_tools_module.DEFAULT_CAMPAIGN_BUILTIN_BASELINES))
        incomplete = {item["algorithm_name"]: item for item in refresh["strict_incomplete_baselines"]}
        self.assertIn("dynamical_ot", incomplete)
        self.assertTrue(incomplete["dynamical_ot"]["missing_records"])
        self.assertEqual(refresh["strict_missing_baselines"][0]["algorithm_name"], "dynamical_ot")
        self.assertIn("baseline training has no terminal metrics record", incomplete["dynamical_ot"]["reasons"])
        ledger = refresh["selected_baseline"]["strict_baseline_audit_ledger"]
        self.assertEqual(len(ledger), len(planner_tools_module.DEFAULT_CAMPAIGN_BUILTIN_BASELINES))
        dynamical_audit = next(item for item in ledger if item["algorithm_name"] == "dynamical_ot")
        self.assertEqual(dynamical_audit["audit_status"], "fail_incomplete_or_unusable")
        self.assertTrue(dynamical_audit["missing_records"])
        self.assertTrue(dynamical_audit["missing_required_record"])
        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        persisted = status["stages"]["stage2_claim_validation"].get("external_baseline_metrics") or {}
        self.assertEqual(
            len(persisted["strict_baseline_audit_ledger"]),
            len(planner_tools_module.DEFAULT_CAMPAIGN_BUILTIN_BASELINES),
        )
        gate = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        audit = gate["checks"]["strict_baseline_audit"]
        self.assertFalse(audit["ok"], audit)
        self.assertEqual(audit["ledger_count"], len(planner_tools_module.DEFAULT_CAMPAIGN_BUILTIN_BASELINES))
        self.assertFalse(any("audit ledger is missing" in blocker for blocker in gate["blockers"]))
        self.assertTrue(any("strict required baseline `dynamical_ot`" in blocker for blocker in gate["blockers"]))
        self.assertTrue(any("no terminal audit record" in blocker for blocker in gate["blockers"]))

    def test_refresh_stage2_strict_persists_baseline_audit_ledger(self) -> None:
        self._write_benchmark_dataset_card("small", "/tmp/small.h5ad")
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator()
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "small", "adata_path": "/tmp/small.h5ad"},
                ],
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        stage2 = campaign["stages"]["stage2_claim_validation"]
        stage2["status"] = "active"
        stage2["active_best_trial_id"] = "trial_stage2"
        campaign["trials"]["trial_stage2"] = {
            "trial_id": "trial_stage2",
            "stage": "stage2_claim_validation",
            "status": "completed",
            "decision": "promote",
            "metrics_summary": {
                "primary_metric": "claim_metric",
                "primary_direction": "greater",
                "primary_value": 0.8,
                "w1_mean": 0.8,
                "w1_backend": "exact",
                "w1_backend_exact": True,
                "w1_backend_params": {},
                "target_dataset_ids": ["small"],
                "custom_metrics": {"claim_metric": 0.8},
                "claim_metric": 0.8,
                "claim_metric_evaluator": dict(evaluator_info),
                "tmv_gate_required": False,
            },
        }
        self.tools.planner_file_tools._save_campaign(campaign)
        calls = []

        def fake_run_training_tool(**kwargs):
            calls.append(kwargs)
            algorithm = str(kwargs.get("candidate_name") or "")
            run_id = f"run_{algorithm}_{len(calls)}"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            tmv_scores = [0.5] if algorithm == "vgfm" else [0.1]
            run_verdict = (
                {
                    "status": "rejected",
                    "reasons": ["TMV quality gate failed: at least one TMV is >= 0.2"],
                }
                if algorithm == "vgfm"
                else {"status": "provisional", "reasons": []}
            )
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "w1_scores": [1.0],
                        "w1_mean": 1.0,
                        "w1_backend": "exact",
                        "w1_backend_exact": True,
                        "w1_backend_params": {},
                        "tmv_scores": tmv_scores,
                        "custom_metrics": {"claim_metric": 0.5},
                        "claim_metric": 0.5,
                        "claim_metric_evaluator": evaluator_info,
                        "run_verdict": run_verdict,
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {
                    "run_id": run_id,
                    "metrics_path": str(metrics_path),
                    "run_dir": str(run_dir),
                    "status": "completed",
                }
            )
            self.state["latest_training_run_id"] = run_id
            return f"✅ TRAINING COMPLETED SUCCESSFULLY.\nRun ID: {run_id}"

        self.tools.run_training_tool = fake_run_training_tool
        refresh = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                stage="stage2_claim_validation",
                overwrite_existing=True,
            )
        )

        self.assertEqual(refresh["status"], "updated")
        ledger = refresh["selected_baseline"]["strict_baseline_audit_ledger"]
        self.assertEqual(len(ledger), len(planner_tools_module.DEFAULT_CAMPAIGN_BUILTIN_BASELINES))
        self.assertTrue(all(item["audit_status"] == "pass_comparable" for item in ledger))
        vgfm_audit = next(item for item in ledger if item["algorithm_name"] == "vgfm")
        self.assertEqual(vgfm_audit["baseline_quality_diagnostic_status"], "failed")
        self.assertIn("TMV quality gate failed", vgfm_audit["baseline_quality_diagnostic_reason"])
        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        persisted = status["stages"]["stage2_claim_validation"]["external_baseline_metrics"]
        self.assertEqual(
            len(persisted["strict_baseline_audit_ledger"]),
            len(planner_tools_module.DEFAULT_CAMPAIGN_BUILTIN_BASELINES),
        )
        gate = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(gate["ok"], gate)
        audit = gate["checks"]["strict_baseline_audit"]
        self.assertTrue(audit["ok"], audit)
        self.assertEqual(audit["ledger_count"], len(planner_tools_module.DEFAULT_CAMPAIGN_BUILTIN_BASELINES))
        self.assertEqual(len(audit["records"]), len(planner_tools_module.DEFAULT_CAMPAIGN_BUILTIN_BASELINES))
        self.assertTrue(all(item["primary_pass"] for item in audit["records"]))
        self.assertTrue(all(item["w1_pass"] for item in audit["records"]))
        vgfm_gate_record = next(item for item in audit["records"] if item["algorithm_name"] == "vgfm")
        self.assertTrue(vgfm_gate_record["primary_pass"])
        self.assertTrue(vgfm_gate_record["w1_pass"])

        before_noop = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        before_source = dict(
            before_noop["stages"]["stage2_claim_validation"].get("external_baseline_source") or {}
        )
        calls_before_noop = len(calls)
        noop_refresh = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                stage="stage2_claim_validation",
                run_missing=False,
                overwrite_existing=False,
            )
        )
        self.assertEqual(noop_refresh["status"], "already_complete")
        self.assertFalse(noop_refresh["changed"])
        self.assertEqual(noop_refresh["strict_baseline_audit"]["ledger_count"], len(planner_tools_module.DEFAULT_CAMPAIGN_BUILTIN_BASELINES))
        self.assertEqual(noop_refresh["strict_baseline_audit"]["missing_required_records"], [])
        self.assertEqual(len(calls), calls_before_noop)
        after_noop = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(
            after_noop["stages"]["stage2_claim_validation"].get("external_baseline_source") or {},
            before_source,
        )

        overwrite_noop = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                stage="stage2_claim_validation",
                baseline_algorithms=list(planner_tools_module.DEFAULT_CAMPAIGN_BUILTIN_BASELINES),
                run_missing=False,
                overwrite_existing=True,
            )
        )
        self.assertEqual(overwrite_noop["status"], "already_complete")
        self.assertFalse(overwrite_noop["changed"])
        self.assertEqual(len(calls), calls_before_noop)
        after_overwrite_noop = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(
            after_overwrite_noop["stages"]["stage2_claim_validation"].get("external_baseline_source") or {},
            before_source,
        )

        baseline_query = json.loads(
            self.tools.query_campaign_baseline_metrics(
                campaign_id=campaign_id,
                stage="stage2_claim_validation",
                include_training_run_scan=False,
                include_benchmark_registry=False,
            )
        )
        audit_summary = baseline_query["stages"]["stage2_claim_validation"]["strict_baseline_audit"]
        self.assertTrue(audit_summary["ok"])
        self.assertEqual(audit_summary["ledger_count"], len(planner_tools_module.DEFAULT_CAMPAIGN_BUILTIN_BASELINES))
        self.assertEqual(audit_summary["missing_required_records"], [])
        self.assertEqual(audit_summary["refresh_recommendation"], "no_refresh_needed")

        self.tools.update_campaign_claim_metric_spec(
            campaign_id,
            claim_metric_spec={"claim_metric_evaluator_id": "claim-metric-v2"},
            reason="test evaluator migration invalidates old baseline claim evidence",
        )
        migrated = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                stage="stage2_claim_validation",
                run_missing=False,
            )
        )
        self.assertEqual(migrated["status"], "updated")
        self.assertFalse(migrated["strict_missing_baselines"])
        self.assertEqual(
            len(migrated["strict_failed_baselines"]),
            len(planner_tools_module.DEFAULT_CAMPAIGN_BUILTIN_BASELINES),
        )
        self.assertTrue(
            all(item["nonblocking_failed_record"] for item in migrated["strict_failed_baselines"])
        )
        migrated_gate = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(
            any("current campaign claim_metric_spec" in item for item in migrated_gate["blockers"]),
            migrated_gate,
        )

    def test_refresh_stage2_strict_failed_baseline_record_is_nonblocking(self) -> None:
        self._write_benchmark_dataset_card("small", "/tmp/small.h5ad")
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator()
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "small", "adata_path": "/tmp/small.h5ad"},
                ],
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        stage2 = campaign["stages"]["stage2_claim_validation"]
        stage2["status"] = "active"
        stage2["active_best_trial_id"] = "trial_stage2"
        campaign["trials"]["trial_stage2"] = {
            "trial_id": "trial_stage2",
            "stage": "stage2_claim_validation",
            "status": "completed",
            "decision": "promote",
            "metrics_summary": {
                "primary_metric": "claim_metric",
                "primary_direction": "greater",
                "primary_value": 0.8,
                "w1_mean": 0.8,
                "w1_backend": "exact",
                "w1_backend_exact": True,
                "w1_backend_params": {},
                "target_dataset_ids": ["small"],
                "custom_metrics": {"claim_metric": 0.8},
                "claim_metric": 0.8,
                "claim_metric_evaluator": dict(evaluator_info),
                "tmv_gate_required": False,
            },
        }
        self.tools.planner_file_tools._save_campaign(campaign)
        calls = []

        def fake_run_training_tool(**kwargs):
            calls.append(kwargs)
            algorithm = str(kwargs.get("candidate_name") or "")
            run_id = f"run_{algorithm}_{len(calls)}"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            if algorithm == "dynamical_ot":
                metrics_path.write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "run_dir": str(run_dir),
                            "status": "failed",
                            "error": "synthetic baseline crash before evaluable trajectory",
                        }
                    ),
                    encoding="utf-8",
                )
                self.state["training_runs"].append(
                    {
                        "run_id": run_id,
                        "metrics_path": str(metrics_path),
                        "run_dir": str(run_dir),
                        "status": "failed",
                    }
                )
                self.state["latest_training_run_id"] = run_id
                return f"❌ TRAINING FAILED.\nRun ID: {run_id}"
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "w1_scores": [1.0],
                        "w1_mean": 1.0,
                        "w1_backend": "exact",
                        "w1_backend_exact": True,
                        "w1_backend_params": {},
                        "tmv_scores": [0.1],
                        "custom_metrics": {"claim_metric": 0.5},
                        "claim_metric": 0.5,
                        "claim_metric_evaluator": evaluator_info,
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {
                    "run_id": run_id,
                    "metrics_path": str(metrics_path),
                    "run_dir": str(run_dir),
                    "status": "completed",
                }
            )
            self.state["latest_training_run_id"] = run_id
            return f"✅ TRAINING COMPLETED SUCCESSFULLY.\nRun ID: {run_id}"

        self.tools.run_training_tool = fake_run_training_tool
        refresh = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                stage="stage2_claim_validation",
                overwrite_existing=True,
            )
        )

        self.assertEqual(refresh["status"], "updated")
        self.assertFalse(refresh["strict_missing_baselines"])
        failed = {item["algorithm_name"]: item for item in refresh["strict_failed_baselines"]}
        self.assertIn("dynamical_ot", failed)
        self.assertTrue(failed["dynamical_ot"]["has_terminal_baseline_record"])
        self.assertTrue(failed["dynamical_ot"]["nonblocking_failed_record"])
        self.assertIn("synthetic baseline crash", "; ".join(failed["dynamical_ot"]["reasons"]))
        ledger = refresh["selected_baseline"]["strict_baseline_audit_ledger"]
        dynamical_audit = next(item for item in ledger if item["algorithm_name"] == "dynamical_ot")
        self.assertEqual(dynamical_audit["audit_status"], "fail_incomplete_or_unusable")
        self.assertTrue(dynamical_audit["nonblocking_failed_record"])
        gate = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(gate["ok"], gate)
        audit = gate["checks"]["strict_baseline_audit"]
        self.assertTrue(audit["ok"], audit)
        self.assertFalse(audit["missing_required_records"])
        self.assertTrue(
            any(item["algorithm_name"] == "dynamical_ot" for item in audit["nonblocking_failed_records"])
        )
        self.assertFalse(any("dynamical_ot" in blocker for blocker in gate["blockers"]))

    def test_refresh_campaign_stage_baselines_always_includes_implementation_map_anchor(self) -> None:
        self._write_benchmark_dataset_card("small", "/tmp/small.h5ad")
        evaluator_path, _ = self._write_claim_metric_evaluator()
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "small", "adata_path": "/tmp/small.h5ad"},
                ],
            },
        )
        self._fill_implementation_map()
        calls = []

        def fake_run_training_tool(**kwargs):
            calls.append(kwargs)
            algorithm = str(kwargs.get("candidate_name") or "")
            run_id = f"run_{algorithm}_{len(calls)}"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "w1_scores": [1.0],
                        "tmv_scores": [0.1],
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {
                    "run_id": run_id,
                    "metrics_path": str(metrics_path),
                    "run_dir": str(run_dir),
                    "status": "completed",
                }
            )
            self.state["latest_training_run_id"] = run_id
            return f"✅ TRAINING COMPLETED SUCCESSFULLY.\nRun ID: {run_id}"

        self.tools.run_training_tool = fake_run_training_tool
        refresh = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                baseline_algorithms=["sf2m"],
            )
        )

        self.assertEqual(refresh["status"], "blocked")
        self.assertEqual(refresh["reason"], "strict_required_baseline_missing_records")
        self.assertEqual(refresh["required_anchor_baseline"], "vgfm")
        self.assertFalse(refresh["anchor_baseline_added"])
        self.assertEqual(refresh["baseline_selection_policy"], "strict_all_builtin")
        self.assertEqual(refresh["requested_baseline_algorithms"], ["sf2m"])
        self.assertFalse(refresh["ignored_requested_baseline_algorithms"])
        self.assertEqual(refresh["baseline_algorithms"], list(planner_tools_module.DEFAULT_CAMPAIGN_BUILTIN_BASELINES))
        self.assertEqual(refresh["refresh_baseline_algorithms"], ["sf2m"])
        self.assertEqual(
            [call["candidate_name"] for call in calls],
            ["sf2m"],
        )

    def test_refresh_stage2_baseline_posthoc_reuses_saved_model_artifact(self) -> None:
        self._write_benchmark_dataset_card("claim_sim", "/tmp/claim_sim.h5ad")
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {
                        "dataset_id": "claim_sim",
                        "adata_path": "/tmp/claim_sim.h5ad",
                        "simulation_version": "claim_sim_v1",
                    }
                ]
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        self.tools.planner_file_tools._save_campaign(campaign)

        model_path = Path(self.tmpdir.name) / "saved_baselines" / "trained_model.h5ad"
        metrics_path = model_path.parent / "metrics.json"
        config_path = model_path.parent / "resolved_config.yaml"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.tools.planner_file_tools.record_algorithm_benchmark_baseline(
            "claim_sim",
            "vgfm",
            {
                "w1_scores": [0.9],
                "tmv_scores": [0.1],
                "trained_model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "resolved_config_path": str(config_path),
            },
            run_id="run_vgfm_saved",
            config_path=str(config_path),
        )

        posthoc_calls = []

        def fake_posthoc(metrics, *, claim_metric_spec, stage, baseline_algorithm="", dataset_entry=None):
            posthoc_calls.append({"metrics": metrics, "claim_metric_spec": claim_metric_spec, "stage": stage, "baseline_algorithm": baseline_algorithm})
            updated = dict(metrics)
            updated["custom_metrics"] = {"claim_metric": 0.72}
            updated["claim_metric_evaluator"] = dict(evaluator_info)
            updated["claim_metric_posthoc_evaluated"] = True
            return updated

        def fake_run_training_tool(**kwargs):
            raise AssertionError("refresh should reuse saved baseline model instead of retraining")

        original_posthoc = self.tools._posthoc_evaluate_claim_metric_for_baseline
        original_run_training = self.tools.run_training_tool
        self.tools._posthoc_evaluate_claim_metric_for_baseline = fake_posthoc
        self.tools.run_training_tool = fake_run_training_tool
        try:
            self._set_campaign_baseline_selection_policy(campaign_id, "permissive")
            refresh = json.loads(
                self.tools.refresh_campaign_stage_baselines(
                    campaign_id,
                    baseline_algorithms=["vgfm"],
                    run_missing=False,
                )
            )
        finally:
            self.tools._posthoc_evaluate_claim_metric_for_baseline = original_posthoc
            self.tools.run_training_tool = original_run_training

        self.assertEqual(refresh["status"], "updated")
        self.assertEqual(len(posthoc_calls), 1)
        self.assertEqual(posthoc_calls[0]["baseline_algorithm"], "vgfm")
        self.assertEqual(refresh["selected_baseline"]["algorithm_name"], "vgfm")
        self.assertEqual(
            refresh["selected_baseline"]["baseline_dataset_summaries"][0]["source"],
            "posthoc_model",
        )
        self.assertAlmostEqual(refresh["selected_baseline"]["primary_value"], 0.72)

        records = self.tools.planner_file_tools.get_algorithm_benchmark_baselines("claim_sim")["baseline_records"]
        record = next(item for item in records if item["algorithm_name"] == "vgfm")
        self.assertEqual(record["artifacts"]["trained_model_path"], str(model_path))
        self.assertEqual(record["metrics"]["custom_metrics"]["claim_metric"], 0.72)

    def test_refresh_stage2_existing_baseline_without_claim_evaluator_does_not_retrain(self) -> None:
        self._write_benchmark_dataset_card("claim_sim", "/tmp/claim_sim.h5ad")
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {
                        "dataset_id": "claim_sim",
                        "adata_path": "/tmp/claim_sim.h5ad",
                        "simulation_version": "claim_sim_v1",
                    }
                ]
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        self.tools.planner_file_tools._save_campaign(campaign)
        self.tools.planner_file_tools.record_algorithm_benchmark_baseline(
            "claim_sim",
            "vgfm",
            {
                "w1_scores": [0.9],
                "tmv_scores": [0.1],
                "algorithm_id": "vgfm",
                "base_config_name": "vgfm",
            },
            run_id="run_vgfm_saved",
        )

        def fake_run_training_tool(**kwargs):
            raise AssertionError("existing baseline without claim evaluator must not be retrained")

        original_run_training = self.tools.run_training_tool
        self.tools.run_training_tool = fake_run_training_tool
        try:
            self._set_campaign_baseline_selection_policy(campaign_id, "permissive")
            refresh = json.loads(
                self.tools.refresh_campaign_stage_baselines(
                    campaign_id,
                    baseline_algorithms=["vgfm"],
                    run_missing=True,
                )
            )
        finally:
            self.tools.run_training_tool = original_run_training

        self.assertEqual(refresh["status"], "blocked")
        self.assertEqual(refresh["reason"], "no_comparable_baseline")
        self.assertEqual(refresh["primary_metric"], "claim_metric")

    def test_refresh_stage2_baseline_aggregates_direct_claim_metric_fields(self) -> None:
        self._write_benchmark_dataset_card("claim_sim", "/tmp/claim_sim.h5ad")
        evaluator_path, _ = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "claim_sim", "adata_path": "/tmp/claim_sim.h5ad"},
                ]
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        self.tools.planner_file_tools._save_campaign(campaign)

        self.tools.planner_file_tools.record_algorithm_benchmark_baseline(
            "claim_sim",
            "balanced_ot_cfm",
            {"w1_mean": 0.9, "tmv_mean": 0.1, "claim_metric": 0.7},
            run_id="run_balanced",
        )
        self.tools.planner_file_tools.record_algorithm_benchmark_baseline(
            "claim_sim",
            "wfrfm",
            {"w1_mean": 0.8, "tmv_mean": 0.1, "custom_metrics": {"claim_metric": 0.4}},
            run_id="run_wfrfm",
        )

        self._set_campaign_baseline_selection_policy(campaign_id, "permissive")
        refresh = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                baseline_algorithms=["balanced_ot_cfm", "wfrfm"],
                run_missing=False,
            )
        )

        self.assertEqual(refresh["status"], "blocked")
        self.assertEqual(refresh["reason"], "no_comparable_baseline")
        self.assertFalse(refresh["missing_records"])
        balanced_failed = next(
            item for item in refresh["baseline_summaries"] if item["algorithm_name"] == "balanced_ot_cfm"
        )
        self.assertTrue(balanced_failed["baseline_unusable"])
        self.assertIn(
            "existing_terminal_baseline_missing_current_campaign_claim_metric",
            balanced_failed["baseline_unusable_reasons"][0],
        )

    def test_refresh_stage2_materializes_declarative_adapter_metric(self) -> None:
        self._write_benchmark_dataset_card("claim_sim", "/tmp/claim_sim.h5ad")
        evaluator_path, _ = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                    "baseline_metric_adapters": {
                        "balanced_ot_cfm": {
                            "claim_metric": 0.7,
                            "supported": True,
                            "verified": True,
                            "source_run_id": "published_anchor_claim_metric_v1",
                        }
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "claim_sim", "adata_path": "/tmp/claim_sim.h5ad"},
                ]
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        self.tools.planner_file_tools._save_campaign(campaign)
        self.tools.planner_file_tools.record_algorithm_benchmark_baseline(
            "claim_sim",
            "balanced_ot_cfm",
            {"w1_mean": 0.9, "tmv_mean": 0.1},
            run_id="run_balanced",
        )

        self._set_campaign_baseline_selection_policy(campaign_id, "permissive")
        refresh = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                baseline_algorithms=["balanced_ot_cfm"],
                run_missing=False,
            )
        )

        self.assertEqual(refresh["status"], "blocked")
        self.assertEqual(refresh["reason"], "no_comparable_baseline")
        self.assertFalse(refresh["missing_records"])
        summary = refresh["baseline_summaries"][0]
        self.assertTrue(summary["baseline_unusable"])
        self.assertIn("trained_model_path", summary["baseline_unusable_reasons"][0])

    def test_refresh_stage2_hydrates_conventional_artifact_paths_for_posthoc_claim(self) -> None:
        self._write_benchmark_dataset_card("claim_sim", "/tmp/claim_sim.h5ad")
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [{"dataset_id": "claim_sim", "adata_path": "/tmp/claim_sim.h5ad"}]
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        self.tools.planner_file_tools._save_campaign(campaign)
        recorded = self.tools.planner_file_tools.record_algorithm_benchmark_baseline(
            "claim_sim",
            "balanced_ot_cfm",
            {"w1_mean": 0.9, "tmv_mean": 0.1},
            run_id="run_balanced_artifact_dir_only",
        )
        artifact_dir = Path(self.tmpdir.name) / "baseline_artifacts" / "builtin_balanced_ot_cfm"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            "metrics.json",
            "model_artifact.json",
            "model_state.pt",
            "resolved_config.yaml",
            "evaluation_trajectory.npz",
        ):
            (artifact_dir / filename).write_text("{}", encoding="utf-8")
        record_path = Path(recorded["record_path"])
        record_payload = json.loads(record_path.read_text(encoding="utf-8"))
        record_payload["artifacts"] = {"artifact_dir": str(artifact_dir)}
        record_path.write_text(json.dumps(record_payload), encoding="utf-8")

        captured = {}
        original_posthoc = self.tools._posthoc_evaluate_claim_metric_for_baseline

        def fake_posthoc(metrics, **kwargs):
            captured.update(metrics)
            updated = dict(metrics)
            updated["custom_metrics"] = {"claim_metric": 0.72}
            updated["claim_metric_evaluator"] = dict(evaluator_info)
            updated["claim_metric_posthoc_evaluated"] = True
            return updated

        self.tools._posthoc_evaluate_claim_metric_for_baseline = fake_posthoc
        try:
            self._set_campaign_baseline_selection_policy(campaign_id, "permissive")
            refresh = json.loads(
                self.tools.refresh_campaign_stage_baselines(
                    campaign_id,
                    baseline_algorithms=["balanced_ot_cfm"],
                    run_missing=False,
                )
            )
        finally:
            self.tools._posthoc_evaluate_claim_metric_for_baseline = original_posthoc

        self.assertEqual(refresh["status"], "updated")
        self.assertEqual(captured["model_artifact_path"], str(artifact_dir / "model_artifact.json"))
        self.assertEqual(
            captured["evaluation_trajectory_path"],
            str(artifact_dir / "evaluation_trajectory.npz"),
        )
        self.assertEqual(captured["resolved_config_path"], str(artifact_dir / "resolved_config.yaml"))

    def test_refresh_stage2_reruns_explicit_existing_baseline_missing_stage_primary(self) -> None:
        self._write_benchmark_dataset_card("claim_sim", "/tmp/claim_sim.h5ad")
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "claim_sim", "adata_path": "/tmp/claim_sim.h5ad"},
                ]
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        self.tools.planner_file_tools._save_campaign(campaign)
        self.tools.planner_file_tools.record_algorithm_benchmark_baseline(
            "claim_sim",
            "balanced_ot_cfm",
            {"w1_mean": 0.9, "tmv_mean": 0.1},
            run_id="old_balanced_missing_claim",
        )

        calls = []

        def fake_run_training_tool(**kwargs):
            calls.append(kwargs)
            run_id = "run_balanced_repaired"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "algorithm_id": "balanced_ot_cfm",
                        "base_config_name": "balanced_ot_cfm",
                        "w1_scores": [0.4],
                        "tmv_scores": [0.1],
                        "custom_metrics": {"claim_metric": 0.72},
                        "claim_metric_evaluator": evaluator_info,
                        "model_artifact_path": str(run_dir / "artifacts" / "model_artifact.json"),
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {
                    "run_id": run_id,
                    "metrics_path": str(metrics_path),
                    "run_dir": str(run_dir),
                    "status": "completed",
                }
            )
            self.state["latest_training_run_id"] = run_id
            return f"✅ TRAINING COMPLETED SUCCESSFULLY.\nRun ID: {run_id}"

        original_run_training = self.tools.run_training_tool
        self.tools.run_training_tool = fake_run_training_tool
        try:
            self._set_campaign_baseline_selection_policy(campaign_id, "permissive")
            refresh = json.loads(
                self.tools.refresh_campaign_stage_baselines(
                    campaign_id,
                    baseline_algorithms=["balanced_ot_cfm"],
                    run_missing=True,
                )
            )
        finally:
            self.tools.run_training_tool = original_run_training

        self.assertEqual(refresh["status"], "updated")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["candidate_name"], "balanced_ot_cfm")
        self.assertEqual(refresh["selected_baseline"]["algorithm_name"], "balanced_ot_cfm")
        self.assertAlmostEqual(refresh["selected_baseline"]["primary_value"], 0.72)

        records = self.tools.planner_file_tools.get_algorithm_benchmark_baselines("claim_sim")["baseline_records"]
        record = next(item for item in records if item["algorithm_name"] == "balanced_ot_cfm")
        self.assertEqual(record["run_id"], "run_balanced_repaired")
        self.assertEqual(record["metrics"]["custom_metrics"]["claim_metric"], 0.72)

    def test_refresh_stage2_rerun_missing_claim_metric_records_terminal_unusable(self) -> None:
        self._write_benchmark_dataset_card("claim_sim", "/tmp/claim_sim.h5ad")
        evaluator_path, _ = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "claim_sim", "adata_path": "/tmp/claim_sim.h5ad"},
                ]
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        self.tools.planner_file_tools._save_campaign(campaign)
        self.tools.planner_file_tools.record_algorithm_benchmark_baseline(
            "claim_sim",
            "balanced_ot_cfm",
            {"w1_mean": 0.9, "tmv_mean": 0.1},
            run_id="old_balanced_missing_claim",
        )

        calls = []

        def fake_run_training_tool(**kwargs):
            calls.append(kwargs)
            run_id = "run_balanced_missing_claim_after_rerun"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "algorithm_id": "balanced_ot_cfm",
                        "base_config_name": "balanced_ot_cfm",
                        "w1_scores": [0.4],
                        "tmv_scores": [0.1],
                        "model_artifact_path": str(run_dir / "artifacts" / "model_artifact.json"),
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {
                    "run_id": run_id,
                    "metrics_path": str(metrics_path),
                    "run_dir": str(run_dir),
                    "status": "completed",
                }
            )
            self.state["latest_training_run_id"] = run_id
            return f"✅ TRAINING COMPLETED SUCCESSFULLY.\nRun ID: {run_id}"

        original_run_training = self.tools.run_training_tool
        self.tools.run_training_tool = fake_run_training_tool
        try:
            self._set_campaign_baseline_selection_policy(campaign_id, "permissive")
            refresh = json.loads(
                self.tools.refresh_campaign_stage_baselines(
                    campaign_id,
                    baseline_algorithms=["balanced_ot_cfm"],
                    run_missing=True,
                )
            )
        finally:
            self.tools.run_training_tool = original_run_training

        self.assertEqual(refresh["status"], "blocked")
        self.assertEqual(refresh["reason"], "no_comparable_baseline")
        self.assertEqual(len(calls), 1)
        self.assertFalse(refresh["missing_records"])
        summary = next(item for item in refresh["baseline_summaries"] if item["algorithm_name"] == "balanced_ot_cfm")
        self.assertTrue(summary["baseline_unusable"])
        self.assertIn("baseline_run_missing_campaign_claim_metric", summary["baseline_unusable_reasons"][0])

        records = self.tools.planner_file_tools.get_algorithm_benchmark_baselines("claim_sim")["baseline_records"]
        record = next(item for item in records if item["algorithm_name"] == "balanced_ot_cfm")
        self.assertEqual(record["run_id"], "run_balanced_missing_claim_after_rerun")
        self.assertTrue(record["metrics"]["baseline_unusable"])
        self.assertEqual(record["metrics"]["claim_metric_missing_reason"], "baseline_run_missing_campaign_claim_metric")

    def test_refresh_stage2_rejects_unverified_declarative_adapter_metric(self) -> None:
        self._write_benchmark_dataset_card("claim_sim", "/tmp/claim_sim.h5ad")
        evaluator_path, _ = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                    "baseline_metric_adapters": {
                        "balanced_ot_cfm": {
                            "claim_metric": 0.7,
                            "supported": True,
                        }
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "claim_sim", "adata_path": "/tmp/claim_sim.h5ad"},
                ]
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        self.tools.planner_file_tools._save_campaign(campaign)
        self.tools.planner_file_tools.record_algorithm_benchmark_baseline(
            "claim_sim",
            "balanced_ot_cfm",
            {"w1_mean": 0.9, "tmv_mean": 0.1},
            run_id="run_balanced",
        )

        self._set_campaign_baseline_selection_policy(campaign_id, "permissive")
        refresh = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                baseline_algorithms=["balanced_ot_cfm"],
                run_missing=False,
            )
        )

        self.assertEqual(refresh["status"], "blocked")
        self.assertEqual(refresh["reason"], "no_comparable_baseline")
        self.assertEqual(refresh["primary_metric"], "claim_metric")
        self.assertFalse(refresh["missing_records"])
        summary = refresh["baseline_summaries"][0]
        self.assertTrue(summary["baseline_unusable"])
        self.assertIn("trained_model_path", summary["baseline_unusable_reasons"][0])

    def test_refresh_stage2_excludes_rejected_baseline_verdicts(self) -> None:
        self._write_benchmark_dataset_card("claim_sim", "/tmp/claim_sim.h5ad")
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "claim_sim", "adata_path": "/tmp/claim_sim.h5ad"},
                ]
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        self.tools.planner_file_tools._save_campaign(campaign)
        self.tools.planner_file_tools.record_algorithm_benchmark_baseline(
            "claim_sim",
            "wfrfm",
            {
                "w1_mean": 0.6,
                "tmv_mean": 0.1,
                "custom_metrics": {"claim_metric": 0.9},
                "claim_metric_evaluator": dict(evaluator_info),
                "run_verdict": {"status": "rejected", "reasons": ["TMV gate failed"]},
            },
            run_id="run_wfrfm_rejected",
        )
        self.tools.planner_file_tools.record_algorithm_benchmark_baseline(
            "claim_sim",
            "balanced_ot_cfm",
            {
                "w1_mean": 0.9,
                "tmv_mean": 0.1,
                "custom_metrics": {"claim_metric": 0.4},
                "claim_metric_evaluator": dict(evaluator_info),
            },
            run_id="run_balanced",
        )

        self._set_campaign_baseline_selection_policy(campaign_id, "permissive")
        refresh = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                baseline_algorithms=["wfrfm", "balanced_ot_cfm"],
                run_missing=False,
            )
        )

        self.assertEqual(refresh["status"], "updated")
        self.assertEqual(refresh["selected_baseline"]["algorithm_name"], "balanced_ot_cfm")
        rejected_summary = next(item for item in refresh["baseline_summaries"] if item["algorithm_name"] == "wfrfm")
        self.assertTrue(rejected_summary["baseline_unusable"])
        self.assertIn("TMV gate failed", rejected_summary["baseline_unusable_reasons"][0])

    def test_refresh_stage2_baseline_prefers_saved_trajectory_for_claim_metric_and_gate(self) -> None:
        import anndata as ad
        import numpy as np
        import pandas as pd

        dataset_path = Path(self.tmpdir.name) / "claim_sim.h5ad"
        adata = ad.AnnData(
            X=np.zeros((4, 2), dtype=np.float32),
            obs=pd.DataFrame({"time_point_processed": [0.0, 0.0, 1.0, 1.0]}),
        )
        adata.obsm["X_latent"] = np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [1.1, 0.0]],
            dtype=np.float32,
        )
        adata.write_h5ad(dataset_path)
        self._write_benchmark_dataset_card("claim_sim", str(dataset_path))

        evaluator_path = Path(self.tmpdir.name) / "claim_metric.py"
        evaluator_path.write_text(
            "def evaluate_claim_metric(context):\n"
            "    if context.model is not None:\n"
            "        raise RuntimeError('trajectory posthoc path should not reload model')\n"
            "    if context.metadata.get('source') != 'posthoc_claim_metric_from_saved_trajectory':\n"
            "        raise RuntimeError('claim metric must use saved trajectory context')\n"
            "    return {'claim_metric': float(len(context.trajectory_time_points)) / 10.0}\n",
            encoding="utf-8",
        )
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {
                        "dataset_id": "claim_sim",
                        "adata_path": str(dataset_path),
                    }
                ]
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        self.tools.planner_file_tools._save_campaign(campaign)

        trajectory_path = Path(self.tmpdir.name) / "saved_baselines" / "evaluation_trajectory.npz"
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        times = [round(i / 10, 1) for i in range(11)]
        manifest = {
            "schema_version": 1,
            "format": "cytobridge_evaluation_trajectory_npz",
            "source": "test_saved_trajectory",
            "step": 0.1,
            "time_points": times,
            "observed_time_points": [0.0, 1.0],
            "observed_time_indices": [0, 10],
            "point_keys": [f"points_{i:06d}" for i in range(11)],
            "weight_keys": [f"weights_{i:06d}" for i in range(11)],
            "metadata": {},
        }
        payload = {
            "time_points": np.asarray(times, dtype=np.float64),
            "observed_time_points": np.asarray([0.0, 1.0], dtype=np.float64),
            "observed_time_indices": np.asarray([0, 10], dtype=np.int64),
            "manifest_json": np.asarray(json.dumps(manifest)),
        }
        for i, time_value in enumerate(times):
            payload[f"points_{i:06d}"] = np.asarray(
                [[time_value, 0.0], [time_value + 0.1, 0.0]],
                dtype=np.float32,
            )
            payload[f"weights_{i:06d}"] = np.asarray([0.5, 0.5], dtype=np.float32)
        np.savez_compressed(trajectory_path, **payload)

        self.tools.planner_file_tools.record_algorithm_benchmark_baseline(
            "claim_sim",
            "vgfm",
            {
                "w1_scores": [0.9],
                "tmv_scores": [0.1],
                "algorithm_id": "vgfm",
                "base_config_name": "vgfm",
                "evaluation_trajectory_path": str(trajectory_path),
            },
            run_id="run_vgfm_saved_trajectory",
        )

        def fake_run_training_tool(**kwargs):
            raise AssertionError("saved trajectory should be sufficient; baseline must not retrain")

        original_run_training = self.tools.run_training_tool
        self.tools.run_training_tool = fake_run_training_tool
        try:
            self._set_campaign_baseline_selection_policy(campaign_id, "permissive")
            refresh = json.loads(
                self.tools.refresh_campaign_stage_baselines(
                    campaign_id,
                    baseline_algorithms=["vgfm"],
                    run_missing=False,
                )
            )
        finally:
            self.tools.run_training_tool = original_run_training

        self.assertEqual(refresh["status"], "updated")
        self.assertEqual(
            refresh["selected_baseline"]["baseline_dataset_summaries"][0]["source"],
            "posthoc_trajectory",
        )
        self.assertAlmostEqual(refresh["selected_baseline"]["primary_value"], 1.1)
        records = self.tools.planner_file_tools.get_algorithm_benchmark_baselines("claim_sim")["baseline_records"]
        record = next(item for item in records if item["algorithm_name"] == "vgfm")
        self.assertTrue(Path(record["artifacts"]["evaluation_trajectory_path"]).exists())
        self.assertAlmostEqual(record["metrics"]["custom_metrics"]["claim_metric"], 1.1)

        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        baseline_evaluator = dict(refresh["selected_baseline"]["claim_metric_evaluator"])
        campaign["stages"]["stage2_claim_validation"]["active_best_trial_id"] = "trial_stage2"
        campaign["trials"]["trial_stage2"] = {
            "trial_id": "trial_stage2",
            "metrics_summary": {
                "primary_metric": "claim_metric",
                "primary_value": 1.3,
                "w1_mean": 1.0,
                "tmv_mean": 0.1,
                "target_dataset_ids": ["claim_sim"],
                "custom_metrics": {"claim_metric": 1.3},
                "claim_metric_evaluator": baseline_evaluator,
            },
        }
        self.tools.planner_file_tools._save_campaign(campaign)
        gate = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(gate["ok"], gate)
        self.assertTrue(gate["checks"]["primary_vs_external_baseline"]["ok"])
        self.assertTrue(gate["checks"]["w1_secondary_vs_external_baseline"]["ok"])

    def test_refresh_ignores_corrupt_existing_builtin_baseline_identity(self) -> None:
        self._write_benchmark_dataset_card("claim_sim", "/tmp/claim_sim.h5ad")
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={"name": "w1_mean", "direction": "lower"},
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {
                        "dataset_id": "claim_sim",
                        "adata_path": "/tmp/claim_sim.h5ad",
                        "simulation_version": "claim_sim_v1",
                    }
                ]
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        self.tools.planner_file_tools._save_campaign(campaign)

        dataset_dir = Path(self.tmpdir.name) / "algorithm_benchmarks" / "datasets" / "claim_sim"
        corrupt_path = dataset_dir / "builtin_baselines" / "builtin_vgfm.json"
        corrupt_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset_id": "claim_sim",
                    "algorithm_name": "vgfm",
                    "baseline_type": "builtin",
                    "metrics_summary": {"w1_mean": 0.9, "tmv_mean": 0.1},
                    "metrics": {
                        "w1_scores": [0.9],
                        "tmv_scores": [0.1],
                        "algorithm_id": "dynamical_ot",
                        "base_config_name": "dynamical_ot",
                    },
                    "run_id": "run_dynamical_ot",
                    "record_path": str(corrupt_path),
                }
            ),
            encoding="utf-8",
        )

        calls = []

        def fake_run_training_tool(**kwargs):
            calls.append(kwargs)
            algorithm = str(kwargs.get("candidate_name") or "")
            run_id = f"run_{algorithm}"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "algorithm_id": algorithm,
                        "base_config_name": algorithm,
                        "w1_scores": [0.3],
                        "tmv_scores": [0.1],
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {
                    "run_id": run_id,
                    "metrics_path": str(metrics_path),
                    "run_dir": str(run_dir),
                    "status": "completed",
                }
            )
            self.state["latest_training_run_id"] = run_id
            return f"✅ TRAINING COMPLETED SUCCESSFULLY.\nRun ID: {run_id}"

        self.tools.run_training_tool = fake_run_training_tool
        self._set_campaign_baseline_selection_policy(campaign_id, "permissive")
        refresh = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                baseline_algorithms=["vgfm"],
            )
        )

        self.assertEqual(refresh["status"], "updated")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["candidate_name"], "vgfm")
        self.assertEqual(refresh["ignored_corrupt_records"][0]["algorithm_name"], "vgfm")
        records = self.tools.planner_file_tools.get_algorithm_benchmark_baselines("claim_sim")["baseline_records"]
        record = next(item for item in records if item["algorithm_name"] == "vgfm")
        self.assertEqual(record["metrics"]["algorithm_id"], "vgfm")

    def test_refresh_does_not_record_stale_latest_run_when_baseline_run_fails(self) -> None:
        self._write_benchmark_dataset_card("claim_sim", "/tmp/claim_sim.h5ad")
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={"name": "w1_mean", "direction": "lower"},
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {
                        "dataset_id": "claim_sim",
                        "adata_path": "/tmp/claim_sim.h5ad",
                        "simulation_version": "claim_sim_v1",
                    }
                ]
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        self.tools.planner_file_tools._save_campaign(campaign)

        stale_run_dir = Path(self.tmpdir.name) / "runs" / "run_dynamical_ot"
        stale_run_dir.mkdir(parents=True, exist_ok=True)
        stale_metrics_path = stale_run_dir / "metrics.json"
        stale_metrics_path.write_text(
            json.dumps(
                {
                    "run_id": "run_dynamical_ot",
                    "run_dir": str(stale_run_dir),
                    "algorithm_id": "dynamical_ot",
                    "base_config_name": "dynamical_ot",
                    "w1_scores": [0.9],
                    "tmv_scores": [0.1],
                }
            ),
            encoding="utf-8",
        )
        self.state["training_runs"].append(
            {
                "run_id": "run_dynamical_ot",
                "metrics_path": str(stale_metrics_path),
                "run_dir": str(stale_run_dir),
                "status": "completed",
            }
        )
        self.state["latest_training_run_id"] = "run_dynamical_ot"

        def fake_run_training_tool(**kwargs):
            return "Error: simulated baseline training failure"

        self.tools.run_training_tool = fake_run_training_tool
        self._set_campaign_baseline_selection_policy(campaign_id, "permissive")
        refresh = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                baseline_algorithms=["vgfm"],
            )
        )

        self.assertEqual(refresh["status"], "blocked")
        self.assertEqual(refresh["missing_records"][0]["algorithm_name"], "vgfm")
        self.assertEqual(refresh["missing_records"][0]["previous_run_id"], "run_dynamical_ot")
        records = self.tools.planner_file_tools.get_algorithm_benchmark_baselines("claim_sim")["baseline_records"]
        self.assertFalse(records)

    def test_refresh_records_failed_builtin_baseline_when_failed_run_manifest_exists(self) -> None:
        self._write_benchmark_dataset_card("claim_sim", "/tmp/claim_sim.h5ad")
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={"name": "w1_mean", "direction": "lower"},
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {
                        "dataset_id": "claim_sim",
                        "adata_path": "/tmp/claim_sim.h5ad",
                        "simulation_version": "claim_sim_v1",
                    }
                ]
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        self.tools.planner_file_tools._save_campaign(campaign)

        def fake_run_training_tool(**kwargs):
            algorithm = str(kwargs.get("candidate_name") or "")
            run_id = f"run_{algorithm}_failed"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "metrics_path": str(metrics_path),
                        "algorithm_id": algorithm,
                        "base_config_name": algorithm,
                        "error": "Preflight failed during backend.prepare",
                        "runtime_sec": 0,
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {
                    "run_id": run_id,
                    "metrics_path": str(metrics_path),
                    "run_dir": str(run_dir),
                    "status": "failed",
                    "error": "Preflight failed during backend.prepare",
                }
            )
            self.state["latest_training_run_id"] = run_id
            return "Training failed: Preflight failed during backend.prepare"

        self.tools.run_training_tool = fake_run_training_tool
        self._set_campaign_baseline_selection_policy(campaign_id, "permissive")
        refresh = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                baseline_algorithms=["vgfm"],
            )
        )

        self.assertEqual(refresh["status"], "blocked")
        self.assertEqual(refresh["reason"], "no_comparable_baseline")
        self.assertEqual(refresh["run_records"][0]["algorithm_name"], "vgfm")
        records = self.tools.planner_file_tools.get_algorithm_benchmark_baselines("claim_sim")["baseline_records"]
        record = next(item for item in records if item["algorithm_name"] == "vgfm")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["metrics_summary"]["w1_mean"], None)
        self.assertEqual(record["metrics_summary"]["tmv_max"], None)
        self.assertIn("Preflight failed", record["error"])
        listing = self.tools.planner_file_tools.list_algorithm_benchmark_baselines("claim_sim")
        row = next(item for item in listing["baselines"] if item["algorithm_name"] == "vgfm")
        self.assertEqual(row["status"], "failed")
        self.assertIn("Preflight failed", row["error"])

        def fail_if_retrained(**kwargs):
            raise AssertionError("existing failed baseline should be reused unless overwrite_existing=true")

        self.tools.run_training_tool = fail_if_retrained
        rerun = json.loads(
            self.tools.refresh_campaign_stage_baselines(
                campaign_id,
                baseline_algorithms=["vgfm"],
            )
        )
        self.assertEqual(rerun["status"], "blocked")
        self.assertEqual(rerun["reason"], "no_comparable_baseline")
        self.assertEqual(
            rerun["baseline_summaries"][0]["baseline_dataset_summaries"][0]["source"],
            "existing_failed",
        )

    def test_refresh_stage2_baseline_skips_declared_unsupported_adaptor(self) -> None:
        self._write_benchmark_dataset_card("claim_sim", "/tmp/claim_sim.h5ad")
        evaluator_path = Path(self.tmpdir.name) / "claim_metric.py"
        evaluator_path.write_text(
            "def evaluate_claim_metric(context):\n"
            "    return {'claim_metric': 0.5}\n",
            encoding="utf-8",
        )
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                    "baseline_metric_adapters": {
                        "vgfm": {
                            "supported": False,
                            "reason": "claim metric requires observable branch labels absent from this baseline output",
                        }
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {
                        "dataset_id": "claim_sim",
                        "adata_path": "/tmp/claim_sim.h5ad",
                        "simulation_version": "claim_sim_v1",
                    }
                ]
            },
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        self.tools.planner_file_tools._save_campaign(campaign)

        def fake_run_training_tool(**kwargs):
            raise AssertionError("declared unsupported baseline must not be trained or posthoc-evaluated")

        original_run_training = self.tools.run_training_tool
        self.tools.run_training_tool = fake_run_training_tool
        try:
            self._set_campaign_baseline_selection_policy(campaign_id, "permissive")
            refresh = json.loads(
                self.tools.refresh_campaign_stage_baselines(
                    campaign_id,
                    baseline_algorithms=["vgfm"],
                    run_missing=True,
                )
            )
        finally:
            self.tools.run_training_tool = original_run_training

        self.assertEqual(refresh["status"], "blocked")
        self.assertEqual(refresh["reason"], "no_comparable_baseline")
        self.assertEqual(refresh["missing_records"][0]["claim_metric_support_status"], "unsupported")
        self.assertIn("branch labels", refresh["missing_records"][0]["claim_metric_unsupported_reason"])

    def test_set_campaign_stage_panel_supplies_default_trial_datasets(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        panel = json.loads(
            self.tools.set_campaign_stage_panel(
                campaign_id,
                dataset_config_overrides={
                    "datasets": [
                        {"dataset_id": "small", "adata_path": "/tmp/small.h5ad"},
                        {"dataset_id": "guardrail", "adata_path": "/tmp/guardrail.h5ad"},
                    ],
                },
            )
        )
        self.assertEqual(panel["target_dataset_ids"], ["small", "guardrail"])
        self.tools.start_campaign_trial(campaign_id)
        self._replace_readme("# default panel candidate\n")
        self._fill_implementation_map()
        self._record_current_implementation_review(campaign_id)
        calls = []

        def fake_run_training_tool(**kwargs):
            calls.append(kwargs)
            run_id = f"run_default_panel_{len(calls)}"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "w1_scores": [0.8],
                        "tmv_scores": [0.1],
                        "custom_metrics": {"claim_metric": 0.5},
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {
                    "run_id": run_id,
                    "metrics_path": str(metrics_path),
                    "run_dir": str(run_dir),
                    "status": "completed",
                }
            )
            self.state["latest_training_run_id"] = run_id
            return f"✅ TRAINING COMPLETED SUCCESSFULLY.\nRun ID: {run_id}"

        self.tools._run_training_tool_impl = fake_run_training_tool  # type: ignore[method-assign]
        result = json.loads(self.tools.run_campaign_trial(campaign_id))

        self.assertEqual(result["decision"], "promote")
        self.assertEqual([item["adata_path"] for item in calls], ["/tmp/small.h5ad", "/tmp/guardrail.h5ad"])
        self.assertEqual(calls[0]["review_purpose"], "campaign")
        self.assertTrue(calls[0]["_skip_review_gates"])
        self.assertEqual(result["target_dataset_ids"], ["small", "guardrail"])

    def test_start_campaign_trial_is_idempotent_while_trial_is_open(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        first = json.loads(self.tools.start_campaign_trial(campaign_id))
        second = json.loads(self.tools.start_campaign_trial(campaign_id))

        self.assertEqual(second["trial_id"], first["trial_id"])
        self.assertEqual(second["start_status"], "existing_open_trial")
        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(len(status["trials"]), 1)
        self.assertEqual(status["current_trial_id"], first["trial_id"])

    def test_open_trial_cannot_change_frozen_stage_panel(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "small", "adata_path": "/tmp/small.h5ad"},
                ],
            },
        )
        self.tools.start_campaign_trial(campaign_id)

        with self.assertRaisesRegex(ValueError, "trial .* is open"):
            self.tools.set_campaign_stage_panel(
                campaign_id,
                dataset_config_overrides={
                    "datasets": [
                        {"dataset_id": "guardrail", "adata_path": "/tmp/guardrail.h5ad"},
                    ],
                },
                overwrite=True,
            )

    def test_run_campaign_trial_blocks_until_implementation_map_is_filled(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        self.assertEqual(campaign["claim_metric_spec"].get("mass_modeling_scope"), "models_unbalanced_mass")
        self.assertTrue(campaign["claim_metric_spec"].get("tmv_gate_required"))
        campaign_id = campaign["campaign_id"]
        trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# needs implementation review\n")

        result = json.loads(self.tools.run_campaign_trial(campaign_id))

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "implementation_map_required")
        self.assertEqual(result["campaign_id"], campaign_id)
        self.assertIn("IMPLEMENTATION_MAP.md", result["message"])
        self.assertEqual(self.state.get("runtime_action") or {}, {})
        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(status["current_trial_id"], trial["trial_id"])
        self.assertEqual(status["trials"][trial["trial_id"]]["status"], "editing")

    def test_run_campaign_trial_blocks_until_implementation_review_after_map(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# needs implementation review\n")
        self._fill_implementation_map()

        result = json.loads(self.tools.run_campaign_trial(campaign_id))

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "implementation_review_required")
        self.assertEqual(self.state["runtime_action"]["kind"], "implementation_agent_review")
        self.assertEqual(self.state["runtime_action"]["proposal_id"], campaign["proposal_id"])
        self.assertEqual(self.state["runtime_action"]["resume_after_review"]["tool"], "run_campaign_trial")
        self.assertEqual(
            self.state["runtime_action"]["resume_after_review"]["args"]["campaign_id"],
            campaign_id,
        )
        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(status["current_trial_id"], trial["trial_id"])

    def test_run_campaign_trial_accepts_implementation_map_without_line_numbers(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.start_campaign_trial(campaign_id)
        self._replace_readme("# needs implementation review\n")
        self._fill_implementation_map_without_line_numbers()

        result = json.loads(self.tools.run_campaign_trial(campaign_id))

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "implementation_review_required")
        self.assertNotIn("implementation_map_required", json.dumps(result))

    def test_implementation_review_is_not_periodic_by_trial_count(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        self._fill_implementation_map()
        self._record_current_implementation_review(campaign_id)
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["stage_policies"]["stage1_feasibility"]["max_trials"] = 20
        campaign["stages"]["stage1_feasibility"]["trial_count"] = 10
        self.tools.planner_file_tools._save_campaign(campaign)

        review_state = self.tools._implementation_review_state(
            "algo_campaign",
            proposal_id=campaign["proposal_id"],
            campaign=json.loads(self.tools.get_algorithm_campaign_status(campaign_id)),
        )

        self.assertTrue(review_state["ok"], review_state)
        self.assertEqual(review_state["due_reasons"], [])
        self.assertEqual(review_state["current_trial_count"], 10)

    def test_workspace_edits_do_not_force_implementation_review_after_approval(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        self._fill_implementation_map()
        record = self._record_current_implementation_review(campaign_id)
        before_state = self.tools._implementation_review_state(
            "algo_campaign",
            proposal_id=campaign["proposal_id"],
            campaign=json.loads(self.tools.get_algorithm_campaign_status(campaign_id)),
        )
        before_hash = str((before_state.get("fingerprint") or {}).get("review_hash") or "")

        self._replace_readme("# tuned implementation notes\n\nREADME-only edits must not force immediate review.\n")
        after_state = self.tools._implementation_review_state(
            "algo_campaign",
            proposal_id=campaign["proposal_id"],
            campaign=json.loads(self.tools.get_algorithm_campaign_status(campaign_id)),
        )

        self.assertTrue(after_state["ok"], after_state)
        self.assertEqual(before_hash, str((after_state.get("fingerprint") or {}).get("review_hash") or ""))
        self.assertEqual(record["review_hash"], before_hash)

        self.tools.start_campaign_trial(campaign_id)
        calls = []

        def fake_run_training_tool(**kwargs):  # noqa: ANN003
            calls.append(kwargs)
            run_id = "run_after_workspace_edit"
            run_dir = Path(self.tmpdir.name) / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "w1_scores": [0.8],
                        "tmv_scores": [0.1],
                        "custom_metrics": {"claim_metric": 0.5},
                    }
                ),
                encoding="utf-8",
            )
            self.state["training_runs"].append(
                {
                    "run_id": run_id,
                    "metrics_path": str(metrics_path),
                    "run_dir": str(run_dir),
                    "status": "completed",
                }
            )
            self.state["latest_training_run_id"] = run_id
            return f"✅ TRAINING COMPLETED SUCCESSFULLY.\nRun ID: {run_id}"

        self.tools._run_training_tool_impl = fake_run_training_tool  # type: ignore[method-assign]
        result = json.loads(
            self.tools.run_campaign_trial(
                campaign_id,
                dataset_config_overrides={
                    "datasets": [
                        {"dataset_id": "small", "adata_path": "/tmp/small.h5ad"},
                    ],
                },
            )
        )

        self.assertEqual(result["decision"], "promote")
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.state.get("runtime_action") or {}, {})

    def test_proposal_revision_requires_next_implementation_review(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        self._fill_implementation_map()
        self._record_current_implementation_review(campaign_id)

        proposal_payload = _valid_proposal_payload()
        revise_result = self._patch_proposal_text(
            proposal_payload["theoretical_core"],
            proposal_payload["theoretical_core"] + " This revision changes semantics.",
            "Change proposal semantics for implementation-review gate test.",
        )
        self.assertIn("auto-approved", revise_result)
        updated_campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        review_state = self.tools._implementation_review_state(
            "algo_campaign",
            proposal_id=updated_campaign["proposal_id"],
            campaign=updated_campaign,
        )

        self.assertFalse(review_state["ok"], review_state)
        self.assertTrue(
            any(
                "different proposal_id" in item or "proposal semantics changed" in item
                for item in review_state["due_reasons"]
            ),
            review_state,
        )

    def test_tmv_gate_skips_balanced_only_campaign_trials(self) -> None:
        patch = f"""*** Begin Patch
*** Update File: {self._proposal_path()}
@@
-## Mass Modeling Scope
-> Required machine-readable value: `balanced_only` or `models_unbalanced_mass`. This proposal-stage attribute controls whether TMV is a hard campaign gate.
-models_unbalanced_mass
+## Mass Modeling Scope
+> Required machine-readable value: `balanced_only` or `models_unbalanced_mass`. This proposal-stage attribute controls whether TMV is a hard campaign gate.
+balanced_only
@@
-Enable unbalanced mass modeling because the method explicitly aims to match observed total-mass change.
+Do not model unbalanced mass in this balanced-only proposal; TMV should be recorded as diagnostic evidence only, not as a hard campaign gate.
@@
-If the coupling-aligned supervision is fit exactly, the induced endpoint weighted particle measure matches the target interval marginal. Because the growth target is defined on the same coupling, the exact-fit limit also recovers the claimed total-mass change.
+If the balanced coupling-aligned supervision is fit exactly, the induced endpoint weighted particle distribution matches the target interval marginal. This balanced-only proposal deliberately does not claim to recover total-mass changes.
*** End Patch
"""
        self.assertIn(
            "auto-approved",
            self.tools.apply_workspace_patch(
                patch,
                reason="Switch this test proposal to balanced-only scope.",
            ),
        )
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        self.assertEqual(campaign["claim_metric_spec"].get("mass_modeling_scope"), "balanced_only")
        self.assertFalse(campaign["claim_metric_spec"].get("tmv_gate_required"))
        campaign_id = campaign["campaign_id"]
        trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# balanced-only\n")
        self.tools.planner_file_tools.prepare_campaign_trial(campaign_id)

        decision = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial["trial_id"],
            metrics={
                "w1_scores": [0.9],
                "tmv_scores": [99.0],
                "custom_metrics": {"claim_metric": 0.5},
                "config": {"model": {"components": ["velocity"]}},
            },
        )

        self.assertEqual(decision["decision"], "promote", decision)
        self.assertFalse(decision["metrics_summary"]["tmv_gate_required"])
        self.assertIn("TMV hard gate skipped", "; ".join(decision["decision_reasons"]))

    def test_proposal_revision_updates_existing_campaign_tmv_gate(self) -> None:
        to_balanced = f"""*** Begin Patch
*** Update File: {self._proposal_path()}
@@
-## Mass Modeling Scope
-> Required machine-readable value: `balanced_only` or `models_unbalanced_mass`. This proposal-stage attribute controls whether TMV is a hard campaign gate.
-models_unbalanced_mass
+## Mass Modeling Scope
+> Required machine-readable value: `balanced_only` or `models_unbalanced_mass`. This proposal-stage attribute controls whether TMV is a hard campaign gate.
+balanced_only
@@
-Enable unbalanced mass modeling because the method explicitly aims to match observed total-mass change.
+Do not model unbalanced mass in this balanced-only proposal; TMV should be recorded as diagnostic evidence only, not as a hard campaign gate.
@@
-If the coupling-aligned supervision is fit exactly, the induced endpoint weighted particle measure matches the target interval marginal. Because the growth target is defined on the same coupling, the exact-fit limit also recovers the claimed total-mass change.
+If the balanced coupling-aligned supervision is fit exactly, the induced endpoint weighted particle distribution matches the target interval marginal. This balanced-only proposal deliberately does not claim to recover total-mass changes.
*** End Patch
"""
        self.assertIn(
            "auto-approved",
            self.tools.apply_workspace_patch(
                to_balanced,
                reason="Start this test from a balanced-only proposal.",
            ),
        )
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        self.assertFalse(campaign["claim_metric_spec"].get("tmv_gate_required"))

        to_unbalanced = f"""*** Begin Patch
*** Update File: {self._proposal_path()}
@@
-## Mass Modeling Scope
-> Required machine-readable value: `balanced_only` or `models_unbalanced_mass`. This proposal-stage attribute controls whether TMV is a hard campaign gate.
-balanced_only
+## Mass Modeling Scope
+> Required machine-readable value: `balanced_only` or `models_unbalanced_mass`. This proposal-stage attribute controls whether TMV is a hard campaign gate.
+models_unbalanced_mass
@@
-Do not model unbalanced mass in this balanced-only proposal; TMV should be recorded as diagnostic evidence only, not as a hard campaign gate.
+Enable unbalanced mass modeling because the revised method explicitly aims to match observed total-mass change.
@@
-If the balanced coupling-aligned supervision is fit exactly, the induced endpoint weighted particle distribution matches the target interval marginal. This balanced-only proposal deliberately does not claim to recover total-mass changes.
+If the coupling-aligned supervision is fit exactly, the induced endpoint weighted particle measure matches the target interval marginal. Because the growth target is defined on the same coupling, the exact-fit limit also recovers the claimed total-mass change.
*** End Patch
"""
        self.assertIn(
            "auto-approved",
            self.tools.apply_workspace_patch(
                to_unbalanced,
                reason="Switch this active campaign proposal back to unbalanced mass modeling.",
            ),
        )
        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertTrue(campaign["claim_metric_spec"].get("tmv_gate_required"))
        self.assertTrue(campaign["stage_policies"]["stage1_feasibility"].get("tmv_gate_required"))

        trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# revised mass-modeling\n")
        self.tools.planner_file_tools.prepare_campaign_trial(campaign_id)
        decision = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial["trial_id"],
            metrics={
                "w1_scores": [0.9],
                "tmv_scores": [99.0],
                "custom_metrics": {"claim_metric": 0.5},
                "config": {"model": {"components": ["velocity"]}},
            },
        )

        self.assertEqual(decision["decision"], "reject", decision)
        self.assertTrue(decision["metrics_summary"]["tmv_gate_required"])
        self.assertIn("TMV hard gate failed", "; ".join(decision["decision_reasons"]))

    def test_tmv_gate_rejects_mass_modeling_campaign_trials(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# mass-modeling\n")
        self.tools.planner_file_tools.prepare_campaign_trial(campaign_id)

        decision = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial["trial_id"],
            metrics={
                "w1_scores": [0.9],
                "tmv_scores": [99.0],
                "custom_metrics": {"claim_metric": 0.5},
                "config": {"model": {"components": ["velocity", "growth"]}},
            },
        )

        self.assertEqual(decision["decision"], "reject", decision)
        self.assertTrue(decision["metrics_summary"]["tmv_gate_required"])
        self.assertIn("TMV hard gate failed", "; ".join(decision["decision_reasons"]))

    def test_manual_training_verdict_uses_tmv_gate_from_model_components(self) -> None:
        balanced = self.tools._infer_training_verdict(
            {
                "w1_scores": [0.9],
                "tmv_scores": [99.0],
                "config": {"model": {"components": ["velocity"]}},
            }
        )
        self.assertEqual(balanced["status"], "provisional", balanced)
        self.assertFalse(balanced["tmv_gate_required"])

        mass_modeling = self.tools._infer_training_verdict(
            {
                "w1_scores": [0.9],
                "tmv_scores": [99.0],
                "config": {"model": {"components": ["velocity", "growth"]}},
            }
        )
        self.assertEqual(mass_modeling["status"], "rejected", mass_modeling)
        self.assertTrue(mass_modeling["tmv_gate_required"])

    def test_manual_training_verdict_prefers_proposal_mass_scope_over_growth_components(self) -> None:
        balanced_only = self.tools._infer_training_verdict(
            {
                "w1_scores": [0.9],
                "tmv_scores": [99.0],
                "algorithm_attributes": {
                    "mass_modeling_scope": "balanced_only",
                    "models_unbalanced_mass": False,
                    "tmv_gate_required": False,
                    "tmv_gate_reason": "proposal.algorithm_attributes.mass_modeling_scope",
                },
                "config": {"model": {"components": ["velocity", "growth"]}},
            }
        )
        self.assertEqual(balanced_only["status"], "provisional", balanced_only)
        self.assertFalse(balanced_only["tmv_gate_required"])
        self.assertEqual(
            balanced_only["tmv_gate_reason"],
            "proposal.algorithm_attributes.mass_modeling_scope",
        )

        mass_modeling = self.tools._infer_training_verdict(
            {
                "w1_scores": [0.9],
                "tmv_scores": [99.0],
                "algorithm_attributes": {
                    "mass_modeling_scope": "models_unbalanced_mass",
                    "models_unbalanced_mass": True,
                    "tmv_gate_required": True,
                    "tmv_gate_reason": "proposal.algorithm_attributes.mass_modeling_scope",
                },
                "config": {"model": {"components": ["velocity"]}},
            }
        )
        self.assertEqual(mass_modeling["status"], "rejected", mass_modeling)
        self.assertTrue(mass_modeling["tmv_gate_required"])

    def test_manual_training_verdict_matches_campaign_tmv_threshold(self) -> None:
        mass_modeling = self.tools._infer_training_verdict(
            {
                "w1_scores": [0.9],
                "tmv_scores": [0.19, 0.21],
                "config": {"model": {"components": ["velocity", "growth"]}},
            }
        )

        self.assertEqual(mass_modeling["status"], "rejected", mass_modeling)
        self.assertFalse(mass_modeling["tmv_pass"])
        self.assertIn("TMV quality gate failed", "; ".join(mass_modeling["reasons"]))

    def test_stage_trial_budget_blocks_new_trials(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec=self._claim_metric_spec(),
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign["stage_policies"]["stage1_feasibility"]["max_trials"] = 1
        self.tools.planner_file_tools._save_campaign(campaign)

        trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# budget candidate\n")
        self.tools.planner_file_tools.prepare_campaign_trial(campaign_id)
        decision = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial["trial_id"],
            metrics={"w1_scores": [0.9], "tmv_scores": [0.1], "custom_metrics": {"claim_metric": 0.5}},
        )
        self.assertEqual(decision["trial_count"], 1)
        self.assertEqual(decision["max_trials"], 1)
        self.assertTrue(decision["budget_exhausted"])

        with self.assertRaisesRegex(ValueError, "trial budget is exhausted"):
            self.tools.planner_file_tools.start_campaign_trial(campaign_id)

    def test_stage2_validation_seeds_optional_stage3_without_hard_gate(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                    "stage_baselines": {
                        "stage1_feasibility": {"w1_mean": 1.0, "tmv_scores": [0.1]},
                        "stage2_claim_validation": {
                            "claim_metric": 0.5,
                            "w1_mean": 1.0,
                            "w1_backend": "exact",
                            "w1_backend_exact": True,
                            "w1_backend_params": {},
                            "tmv_scores": [0.1],
                            "target_dataset_ids": ["claim_sim"],
                            "claim_metric_evaluator": dict(evaluator_info),
                        },
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]

        stage1_trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# stage1 best\n")
        self.tools.planner_file_tools.prepare_campaign_trial(campaign_id)
        self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=stage1_trial["trial_id"],
            metrics={"w1_scores": [0.9], "tmv_scores": [0.1], "custom_metrics": {"claim_metric": 0.4}},
        )
        self.assertTrue(json.loads(self.tools.check_campaign_stage_gate(campaign_id))["advanced"])

        stage2_trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# stage2 validated\n")
        self.tools.planner_file_tools.prepare_campaign_trial(
            campaign_id,
            dataset_config_overrides={
                "datasets": [
                    {"dataset_id": "claim_sim", "adata_path": "/tmp/claim_sim.h5ad"},
                ],
            },
        )
        self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=stage2_trial["trial_id"],
            metrics={
                "w1_scores": [0.9],
                "w1_backend": "exact",
                "w1_backend_exact": True,
                "w1_backend_params": {},
                "tmv_scores": [0.1],
                "custom_metrics": {"claim_metric": 0.7},
                "claim_metric_evaluator": dict(evaluator_info),
                "target_dataset_ids": ["claim_sim"],
            },
        )
        stage2_gate = json.loads(self.tools.check_campaign_stage_gate(campaign_id))
        self.assertTrue(stage2_gate["ok"], stage2_gate)
        self.assertTrue(stage2_gate["algorithm_validated"], stage2_gate)
        self.assertEqual(stage2_gate["next_stage"], "stage3_tuning")

        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(status["current_stage"], "stage3_tuning")
        self.assertTrue(status["algorithm_validated"])
        stage2_evidence = status["stages"]["stage2_claim_validation"]["stage_gate_evidence"]
        self.assertTrue(stage2_evidence["ok"])
        self.assertEqual(stage2_evidence["active_best_trial_id"], stage2_trial["trial_id"])
        self.assertEqual(stage2_evidence["external_baseline_metrics"]["target_dataset_ids"], ["claim_sim"])
        self.assertEqual(status["stages"]["stage3_tuning"]["active_best_trial_id"], stage2_trial["trial_id"])
        stage3_state = status["stages"]["stage3_tuning"]
        self.assertEqual(stage3_state["claim_guardrail_dataset_ids"], ["claim_sim"])
        self.assertEqual(stage3_state["stage_panel"]["target_dataset_ids"], ["claim_sim"])
        self.assertEqual(stage3_state["external_baseline_metrics"]["target_dataset_ids"], ["claim_sim"])
        self.assertEqual(stage3_state["external_baseline_source"]["source"], "inherited_from_stage2_claim_validation")
        self.assertEqual(stage3_state["active_best_metrics_summary"]["primary_metric"], "w1_mean")
        self.assertAlmostEqual(stage3_state["active_best_metrics_summary"]["primary_value"], 0.9)
        self.assertAlmostEqual(stage3_state["active_best_metrics_summary"]["secondary_value"], 0.7)
        self.assertEqual(status["stage_policies"]["stage3_tuning"]["secondary_tolerance"], 0.05)

        stage3_gate = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(stage3_gate["ok"], stage3_gate)
        self.assertFalse(stage3_gate["advanced"], stage3_gate)
        self.assertFalse(stage3_gate["checks"]["stage3_tuning_policy"]["pass_fail_gate"])
        self.assertTrue(stage3_gate["checks"]["stage3_optional_sota_gate"]["ok"])
        self.assertEqual(stage3_gate["checks"]["stage3_tuning_policy"]["min_trials_before_advance"], 10)
        self.assertEqual(stage3_gate["blockers"], [])

        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(status["stage_gate_status"], "passed_optional_sota_gate")
        self.assertIn("optional SOTA gate passed", status["user_facing_status"])
        stage3_evidence = status["stages"]["stage3_tuning"]["stage_gate_evidence"]
        self.assertTrue(stage3_evidence["ok"])
        self.assertEqual(stage3_evidence["checks"]["trial_budget"]["min_trials_before_advance"], 10)

        status["stages"]["stage3_tuning"]["external_baseline_metrics"] = {
            "w1_mean": 0.9,
            "w1_backend": "exact",
            "w1_backend_exact": True,
            "w1_backend_params": {},
            "custom_metrics": {"claim_metric": 0.7},
            "target_dataset_ids": ["claim_sim"],
        }
        self.tools.planner_file_tools._save_campaign(status)
        stage3_sota_ready = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(stage3_sota_ready["ok"], stage3_sota_ready)
        self.assertTrue(stage3_sota_ready["checks"]["stage3_optional_sota_gate"]["ok"])
        self.assertFalse(stage3_sota_ready["advanced"], stage3_sota_ready)

        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(status["stage_gate_status"], "passed_optional_sota_gate")
        self.assertIn("optional SOTA gate passed", status["user_facing_status"])

        status["stages"]["stage3_tuning"].pop("external_baseline_metrics", None)
        status["stages"]["stage3_tuning"]["trial_count"] = 10
        self.tools.planner_file_tools._save_campaign(status)
        stage3_not_ready = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertFalse(stage3_not_ready["ok"], stage3_not_ready)
        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        status["stages"]["stage3_tuning"]["trial_count"] = status["stage_policies"]["stage3_tuning"]["max_trials"]
        self.tools.planner_file_tools._save_campaign(status)
        stage3_budget_ready = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertFalse(stage3_budget_ready["ok"], stage3_budget_ready)
        self.assertIn("strict baseline audit requires external baseline metrics", stage3_budget_ready["blockers"])
        self.assertFalse(stage3_budget_ready["checks"]["stage3_optional_sota_gate"]["ok"])
        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(status["stage_gate_status"], "blocked_failed_checks")

    def test_stage2_simulation_gate_requires_matching_baseline_version(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                    "simulation_version": "claim_sim_v1",
                    "stage_baselines": {
                        "stage2_claim_validation": {
                            "claim_metric": 0.5,
                            "w1_mean": 1.0,
                            "tmv_scores": [0.1],
                            "claim_metric_evaluator": dict(evaluator_info),
                        },
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        campaign["stages"]["stage2_claim_validation"]["active_best_trial_id"] = "trial_sim"
        campaign["trials"]["trial_sim"] = {
            "trial_id": "trial_sim",
            "metrics_summary": {
                "primary_metric": "claim_metric",
                "primary_value": 0.7,
                "w1_mean": 0.9,
                "tmv_mean": 0.1,
                "simulation_version": "claim_sim_v1",
                "custom_metrics": {"claim_metric": 0.7},
                "claim_metric_evaluator": dict(evaluator_info),
            },
        }
        self.tools.planner_file_tools._save_campaign(campaign)

        missing_version = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertFalse(missing_version["ok"], missing_version)
        self.assertTrue(any("missing simulation_version" in item for item in missing_version["blockers"]))

        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["stages"]["stage2_claim_validation"]["external_baseline_metrics"]["simulation_version"] = "claim_sim_v1"
        self.tools.planner_file_tools._save_campaign(campaign)

        matching_version = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(matching_version["ok"], matching_version)

    def test_stage2_gate_prefers_stage_panel_simulation_version_after_panel_switch(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                    "simulation_version": "old_sim_v1",
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        self.tools.set_campaign_stage_panel(
            campaign_id,
            stage="stage2_claim_validation",
            dataset_config_overrides={
                "datasets": [
                    {
                        "dataset_id": "claim_sim",
                        "adata_path": "/tmp/claim_sim.h5ad",
                        "simulation_version": "new_sim_v2",
                    }
                ]
            },
        )

        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["current_stage"] = "stage2_claim_validation"
        stage = campaign["stages"]["stage2_claim_validation"]
        stage["status"] = "active"
        stage["active_best_trial_id"] = "trial_new_panel"
        stage["external_baseline_metrics"] = {
            "primary_metric": "claim_metric",
            "primary_value": 0.5,
            "w1_mean": 1.0,
            "tmv_scores": [0.1],
            "target_dataset_ids": ["claim_sim"],
            "simulation_version": "new_sim_v2",
            "custom_metrics": {"claim_metric": 0.5},
            "claim_metric_evaluator": dict(evaluator_info),
        }
        campaign["trials"]["trial_new_panel"] = {
            "trial_id": "trial_new_panel",
            "metrics_summary": {
                "primary_metric": "claim_metric",
                "primary_value": 0.72,
                "w1_mean": 0.9,
                "tmv_scores": [0.1],
                "target_dataset_ids": ["claim_sim"],
                "simulation_version": "new_sim_v2",
                "custom_metrics": {"claim_metric": 0.72},
                "claim_metric_evaluator": dict(evaluator_info),
            },
        }
        self.tools.planner_file_tools._save_campaign(campaign)

        gate = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(gate["ok"], gate)
        self.assertFalse(any("simulation_version" in item for item in gate["blockers"]), gate)

    def test_stage2_gate_requires_ten_percent_claim_gain_and_uses_scale_aware_w1_guardrail(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                    "stage_baselines": {
                        "stage2_claim_validation": {
                            "claim_metric": 0.5,
                            "w1_mean": 1.0,
                            "w1_backend": "exact",
                            "w1_backend_exact": True,
                            "w1_backend_params": {},
                            "tmv_scores": [0.1],
                            "target_dataset_ids": ["claim_sim"],
                            "claim_metric_evaluator": dict(evaluator_info),
                        },
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        campaign["stages"]["stage2_claim_validation"]["active_best_trial_id"] = "trial_stage2"
        campaign["trials"]["trial_stage2"] = {
            "trial_id": "trial_stage2",
            "metrics_summary": {
                "primary_metric": "claim_metric",
                "primary_value": 0.54,
                "w1_mean": 1.25,
                "tmv_mean": 0.1,
                "target_dataset_ids": ["claim_sim"],
                "custom_metrics": {"claim_metric": 0.54},
                "claim_metric_evaluator": dict(evaluator_info),
                "w1_backend": "exact",
                "w1_backend_exact": True,
                "w1_backend_params": {},
            },
        }
        self.tools.planner_file_tools._save_campaign(campaign)

        too_small_gain = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertFalse(too_small_gain["ok"], too_small_gain)
        self.assertTrue(any("primary metric gate" in item for item in too_small_gain["blockers"]))
        self.assertTrue(too_small_gain["checks"]["w1_secondary_vs_external_baseline"]["ok"])

        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["trials"]["trial_stage2"]["metrics_summary"]["primary_value"] = 0.56
        campaign["trials"]["trial_stage2"]["metrics_summary"]["custom_metrics"]["claim_metric"] = 0.56
        self.tools.planner_file_tools._save_campaign(campaign)
        enough_gain = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(enough_gain["ok"], enough_gain)
        self.assertEqual(
            enough_gain["checks"]["w1_secondary_vs_external_baseline"]["multiplier_reason"],
            "mid_baseline_w1_scale",
        )
        self.assertAlmostEqual(
            enough_gain["checks"]["w1_secondary_vs_external_baseline"]["effective_multiplier"],
            1.3,
        )

        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["trials"]["trial_stage2"]["metrics_summary"]["w1_mean"] = 1.4
        self.tools.planner_file_tools._save_campaign(campaign)
        w1_too_high = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertFalse(w1_too_high["ok"], w1_too_high)
        self.assertFalse(w1_too_high["checks"]["w1_secondary_vs_external_baseline"]["ok"])

        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["stages"]["stage2_claim_validation"]["external_baseline_metrics"]["w1_mean"] = 6.0
        campaign["trials"]["trial_stage2"]["metrics_summary"]["w1_mean"] = 8.5
        self.tools.planner_file_tools._save_campaign(campaign)
        high_scale_tight = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertFalse(high_scale_tight["ok"], high_scale_tight)
        self.assertEqual(
            high_scale_tight["checks"]["w1_secondary_vs_external_baseline"]["multiplier_reason"],
            "high_baseline_w1_scale",
        )
        self.assertAlmostEqual(
            high_scale_tight["checks"]["w1_secondary_vs_external_baseline"]["effective_multiplier"],
            1.2,
        )
        self.assertAlmostEqual(
            high_scale_tight["checks"]["w1_secondary_vs_external_baseline"]["allowed_w1"],
            7.2,
        )

    def test_stage2_gate_blocks_approximate_w1_without_baseline_provenance(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                    "stage_baselines": {
                        "stage2_claim_validation": {
                            "claim_metric": 0.5,
                            "w1_mean": 1.0,
                            "tmv_scores": [0.1],
                            "target_dataset_ids": ["claim_sim"],
                            "claim_metric_evaluator": dict(evaluator_info),
                        },
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign["current_stage"] = "stage2_claim_validation"
        campaign["stages"]["stage2_claim_validation"]["status"] = "active"
        campaign["stages"]["stage2_claim_validation"]["active_best_trial_id"] = "trial_stage2"
        campaign["trials"]["trial_stage2"] = {
            "trial_id": "trial_stage2",
            "metrics_summary": {
                "primary_metric": "claim_metric",
                "primary_value": 0.72,
                "w1_mean": 0.5,
                "tmv_mean": 0.1,
                "target_dataset_ids": ["claim_sim"],
                "custom_metrics": {"claim_metric": 0.72},
                "claim_metric_evaluator": dict(evaluator_info),
                "w1_backend": "geomloss_sinkhorn_online",
                "w1_backend_exact": False,
                "w1_backend_params": {"p": 1, "blur": 1.0, "backend": "online"},
            },
        }
        baseline = campaign["stages"]["stage2_claim_validation"]["external_baseline_metrics"]
        baseline.pop("w1_backend", None)
        baseline.pop("w1_backend_exact", None)
        baseline.pop("w1_backend_params", None)
        self._disable_test_w1_backend_save_patch = True
        try:
            self.tools.planner_file_tools._save_campaign(campaign)
        finally:
            self._disable_test_w1_backend_save_patch = False

        gate = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertFalse(gate["ok"], gate)
        self.assertFalse(gate["checks"]["w1_secondary_vs_external_baseline"]["ok"])
        self.assertTrue(
            any("external baseline lacks W1 backend provenance" in item for item in gate["blockers"]),
            gate,
        )

    def test_stage2_gate_uses_independent_claim_and_w1_sota_baselines(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign["current_stage"] = "stage2_claim_validation"
        stage = campaign["stages"]["stage2_claim_validation"]
        stage["status"] = "active"
        stage["active_best_trial_id"] = "trial_stage2"
        stage["external_baseline_metrics"] = {
            "algorithm_name": "claim_sota",
            "primary_metric": "claim_metric",
            "primary_value": 0.8,
            "custom_metrics": {"claim_metric": 0.8},
            "w1_mean": 5.0,
            "w1_backend": "exact",
            "w1_backend_exact": True,
            "w1_backend_params": {},
            "target_dataset_ids": ["claim_sim"],
            "claim_metric_evaluator": dict(evaluator_info),
            "claim_sota_baseline_metrics": {
                "algorithm_name": "claim_sota",
                "primary_metric": "claim_metric",
                "primary_value": 0.8,
                "custom_metrics": {"claim_metric": 0.8},
                "w1_mean": 5.0,
                "w1_backend": "exact",
                "w1_backend_exact": True,
                "w1_backend_params": {},
                "target_dataset_ids": ["claim_sim"],
                "claim_metric_evaluator": dict(evaluator_info),
            },
            "w1_sota_baseline_metrics": {
                "algorithm_name": "w1_sota",
                "primary_metric": "claim_metric",
                "primary_value": 0.4,
                "custom_metrics": {"claim_metric": 0.4},
                "w1_mean": 1.0,
                "w1_backend": "exact",
                "w1_backend_exact": True,
                "w1_backend_params": {},
                "target_dataset_ids": ["claim_sim"],
                "claim_metric_evaluator": dict(evaluator_info),
            },
        }
        campaign["trials"]["trial_stage2"] = {
            "trial_id": "trial_stage2",
            "metrics_summary": {
                "primary_metric": "claim_metric",
                "primary_value": 0.89,
                "custom_metrics": {"claim_metric": 0.89},
                "w1_mean": 1.4,
                "tmv_mean": 0.1,
                "target_dataset_ids": ["claim_sim"],
                "claim_metric_evaluator": dict(evaluator_info),
                "w1_backend": "exact",
                "w1_backend_exact": True,
                "w1_backend_params": {},
            },
        }
        self.tools.planner_file_tools._save_campaign(campaign)

        gate = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertFalse(gate["ok"], gate)
        self.assertTrue(gate["checks"]["primary_vs_external_baseline"]["ok"], gate)
        self.assertEqual(gate["checks"]["primary_vs_external_baseline"]["baseline_algorithm"], "claim_sota")
        self.assertFalse(gate["checks"]["w1_secondary_vs_external_baseline"]["ok"], gate)
        self.assertEqual(gate["checks"]["w1_secondary_vs_external_baseline"]["baseline_algorithm"], "w1_sota")

        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["trials"]["trial_stage2"]["metrics_summary"]["w1_mean"] = 1.2
        self.tools.planner_file_tools._save_campaign(campaign)
        fixed_w1 = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(fixed_w1["ok"], fixed_w1)

    def test_stage2_locked_gate_spec_from_env_is_enforced(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "M2_growth",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign["current_stage"] = "stage2_claim_validation"
        stage = campaign["stages"]["stage2_claim_validation"]
        stage["status"] = "active"
        stage["active_best_trial_id"] = "trial_stage2_locked"
        stage["external_baseline_metrics"] = {
            "algorithm_name": "baseline",
            "primary_metric": "M2_growth",
            "primary_value": 0.5,
            "custom_metrics": {"M2_growth": 0.5},
            "w1_mean": 1.0,
            "target_dataset_ids": ["public10"],
            "claim_metric_evaluator": dict(evaluator_info),
            "claim_sota_baseline_metrics": {
                "algorithm_name": "claim_sota",
                "primary_metric": "M2_growth",
                "primary_value": 0.5,
                "custom_metrics": {"M2_growth": 0.5},
                "w1_mean": 1.0,
                "target_dataset_ids": ["public10"],
                "claim_metric_evaluator": dict(evaluator_info),
            },
            "w1_sota_baseline_metrics": {
                "algorithm_name": "w1_sota",
                "w1_mean": 1.0,
                "target_dataset_ids": ["public10"],
                "claim_metric_evaluator": dict(evaluator_info),
            },
        }
        campaign["trials"]["trial_stage2_locked"] = {
            "trial_id": "trial_stage2_locked",
            "metrics_summary": {
                "primary_metric": "M2_growth",
                "primary_value": 0.7,
                "custom_metrics": {"M2_growth": 0.7},
                "total_score": 0.7,
                "w1_mean": 1.1,
                "tmv_mean": 0.1,
                "target_dataset_ids": ["public10"],
                "claim_metric_evaluator": dict(evaluator_info),
            },
        }
        self.tools.planner_file_tools._save_campaign(campaign)
        previous = os.environ.get("CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON")
        os.environ["CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON"] = json.dumps(
            {
                "gate_id": "public10_growth_gate",
                "scope": {"algorithm_id": "algo_campaign"},
                "stages": {
                    "stage2_claim_validation": {
                        "requirements": [
                            {"metric": "M2_growth", "min": 0.6},
                            {"metric": "total_score", "min": 0.75},
                        ]
                    }
                },
            }
        )
        try:
            blocked = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
            self.assertFalse(blocked["ok"], blocked)
            self.assertIn("locked_stage_gate", blocked["checks"])
            locked = blocked["checks"]["locked_stage_gate"]
            self.assertFalse(locked["ok"], locked)
            self.assertEqual(locked["gate_id"], "public10_growth_gate")

            campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
            campaign["trials"]["trial_stage2_locked"]["metrics_summary"]["total_score"] = 0.76
            self.tools.planner_file_tools._save_campaign(campaign)
            passed = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
            self.assertTrue(passed["ok"], passed)
            self.assertTrue(passed["checks"]["locked_stage_gate"]["ok"])
        finally:
            if previous is None:
                os.environ.pop("CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON", None)
            else:
                os.environ["CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON"] = previous

    def test_stage2_flat_locked_gate_spec_from_env_is_enforced(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "M4_fate",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign["current_stage"] = "stage2_claim_validation"
        stage = campaign["stages"]["stage2_claim_validation"]
        stage["status"] = "active"
        stage["active_best_trial_id"] = "trial_stage2_flat_locked"
        stage["external_baseline_metrics"] = {
            "algorithm_name": "baseline",
            "primary_metric": "M4_fate",
            "primary_value": 0.1,
            "custom_metrics": {"M4_fate": 0.1, "total_score": 0.4},
            "w1_mean": 1.0,
            "target_dataset_ids": ["case_a", "case_b"],
            "claim_metric_evaluator": dict(evaluator_info),
            "claim_sota_baseline_metrics": {
                "algorithm_name": "claim_sota",
                "primary_metric": "M4_fate",
                "primary_value": 0.1,
                "custom_metrics": {"M4_fate": 0.1, "total_score": 0.4},
                "w1_mean": 1.0,
                "target_dataset_ids": ["case_a", "case_b"],
                "claim_metric_evaluator": dict(evaluator_info),
            },
            "w1_sota_baseline_metrics": {
                "algorithm_name": "w1_sota",
                "w1_mean": 1.0,
                "target_dataset_ids": ["case_a", "case_b"],
                "claim_metric_evaluator": dict(evaluator_info),
            },
        }
        campaign["trials"]["trial_stage2_flat_locked"] = {
            "trial_id": "trial_stage2_flat_locked",
            "metrics_summary": {
                "primary_metric": "M4_fate",
                "primary_value": 0.5,
                "custom_metrics": {"M4_fate": 0.5, "total_score": 0.65},
                "w1_mean": 1.0,
                "tmv_mean": 0.1,
                "target_dataset_ids": ["case_a", "case_b"],
                "claim_metric_evaluator": dict(evaluator_info),
            },
        }
        self.tools.planner_file_tools._save_campaign(campaign)
        previous = os.environ.get("CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON")
        os.environ["CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON"] = json.dumps(
            {
                "version": "1.0",
                "stage": "stage2",
                "target_dataset_ids": ["case_a", "case_b"],
                "requirements": {
                    "aggregate": [
                        {"metric": "M4_fate", "op": ">=", "value": 0.6},
                        {"metric": "total_score", "op": ">=", "value": 0.7},
                    ],
                    "require_exact_target_dataset_ids": True,
                },
            }
        )
        try:
            blocked = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
            self.assertFalse(blocked["ok"], blocked)
            locked = blocked["checks"]["locked_stage_gate"]
            self.assertFalse(locked["ok"], locked)
            self.assertEqual(locked["requirements"][0]["metric"], "target_dataset_ids")
            self.assertTrue(locked["requirements"][0]["ok"])
            self.assertEqual(locked["requirements"][1]["min"], 0.6)

            campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
            summary = campaign["trials"]["trial_stage2_flat_locked"]["metrics_summary"]
            summary["custom_metrics"]["M4_fate"] = 0.62
            summary["primary_value"] = 0.62
            summary["custom_metrics"]["total_score"] = 0.72
            self.tools.planner_file_tools._save_campaign(campaign)
            passed = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
            self.assertTrue(passed["ok"], passed)
            self.assertTrue(passed["checks"]["locked_stage_gate"]["ok"])
        finally:
            if previous is None:
                os.environ.pop("CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON", None)
            else:
                os.environ["CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON"] = previous

    def test_stage3_promotion_preserves_inherited_stage2_locked_floor(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign["current_stage"] = "stage3_tuning"
        campaign["stages"]["stage2_claim_validation"]["status"] = "passed"
        stage = campaign["stages"]["stage3_tuning"]
        stage["status"] = "active"
        stage["active_best_trial_id"] = "trial_stage2_seed"
        stage["active_best_metrics_summary"] = {
            "primary_metric": "w1_mean",
            "primary_direction": "lower",
            "primary_value": 1.0,
            "secondary_metric": "claim_metric",
            "secondary_direction": "greater",
            "secondary_value": 0.72,
            "w1_mean": 1.0,
            "tmv_mean": 0.1,
            "custom_metrics": {"claim_metric": 0.72, "total_score": 0.72},
            "target_dataset_ids": ["claim_sim"],
            "claim_metric_evaluator": dict(evaluator_info),
        }
        campaign["trials"]["trial_stage2_seed"] = {
            "trial_id": "trial_stage2_seed",
            "stage": "stage2_claim_validation",
            "status": "completed",
            "decision": "promote",
            "metrics_summary": dict(stage["active_best_metrics_summary"]),
        }
        campaign["trials"]["trial_stage3_regressed"] = {
            "trial_id": "trial_stage3_regressed",
            "stage": "stage3_tuning",
            "status": "running",
            "snapshot_id": "",
        }
        self.tools.planner_file_tools._save_campaign(campaign)
        previous = os.environ.get("CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON")
        os.environ["CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON"] = json.dumps(
            {
                "version": "1.0",
                "stage": "stage2",
                "target_dataset_ids": ["claim_sim"],
                "requirements": {
                    "aggregate": [
                        {"metric": "total_score", "op": ">=", "value": 0.7},
                    ],
                    "require_exact_target_dataset_ids": True,
                },
            }
        )
        try:
            decision = self.tools.planner_file_tools.decide_campaign_trial(
                campaign_id,
                trial_id="trial_stage3_regressed",
                metrics={
                    "w1_scores": [0.7],
                    "tmv_scores": [0.1],
                    "custom_metrics": {"claim_metric": 0.72, "total_score": 0.65},
                    "target_dataset_ids": ["claim_sim"],
                    "claim_metric_evaluator": dict(evaluator_info),
                },
            )
            self.assertEqual(decision["decision"], "reject", decision)
            self.assertTrue(any("inherited Stage 2 locked-gate floor" in item for item in decision["decision_reasons"]))
            status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
            self.assertEqual(status["stages"]["stage3_tuning"]["active_best_trial_id"], "trial_stage2_seed")
        finally:
            if previous is None:
                os.environ.pop("CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON", None)
            else:
                os.environ["CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON"] = previous

    def test_stage3_claim_metric_regression_floor_is_fixed_to_stage2_active_best(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign["current_stage"] = "stage3_tuning"
        stage2_metrics = {
            "primary_metric": "claim_metric",
            "primary_direction": "greater",
            "primary_value": 0.70,
            "secondary_metric": "w1_mean",
            "secondary_direction": "lower",
            "secondary_value": 0.9,
            "w1_mean": 0.9,
            "tmv_mean": 0.1,
            "custom_metrics": {"claim_metric": 0.70},
            "target_dataset_ids": ["claim_sim"],
            "claim_metric_evaluator": dict(evaluator_info),
        }
        campaign["stages"]["stage2_claim_validation"].update(
            {
                "status": "passed",
                "active_best_trial_id": "trial_stage2_seed",
                "stage_panel": {"target_dataset_ids": ["claim_sim"]},
                "stage_gate_evidence": {
                    "ok": True,
                    "active_best_trial_id": "trial_stage2_seed",
                    "active_best_metrics_summary": dict(stage2_metrics),
                    "policy": dict(campaign["stage_policies"]["stage2_claim_validation"]),
                },
            }
        )
        stage3 = campaign["stages"]["stage3_tuning"]
        stage3["status"] = "active"
        stage3["active_best_trial_id"] = "trial_stage3_current"
        stage3["active_best_metrics_summary"] = {
            "primary_metric": "w1_mean",
            "primary_direction": "lower",
            "primary_value": 0.8,
            "secondary_metric": "claim_metric",
            "secondary_direction": "greater",
            "secondary_value": 0.67,
            "w1_mean": 0.8,
            "tmv_mean": 0.1,
            "custom_metrics": {"claim_metric": 0.67},
            "target_dataset_ids": ["claim_sim"],
            "claim_metric_evaluator": dict(evaluator_info),
        }
        campaign["trials"]["trial_stage2_seed"] = {
            "trial_id": "trial_stage2_seed",
            "stage": "stage2_claim_validation",
            "status": "completed",
            "decision": "promote",
            "metrics_summary": dict(stage2_metrics),
        }
        campaign["trials"]["trial_stage3_current"] = {
            "trial_id": "trial_stage3_current",
            "stage": "stage3_tuning",
            "status": "completed",
            "decision": "promote",
            "metrics_summary": dict(stage3["active_best_metrics_summary"]),
        }
        campaign["trials"]["trial_stage3_candidate"] = {
            "trial_id": "trial_stage3_candidate",
            "stage": "stage3_tuning",
            "status": "running",
            "snapshot_id": "",
        }
        self.tools.planner_file_tools._save_campaign(campaign)

        decision = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id="trial_stage3_candidate",
            metrics={
                "w1_scores": [0.7],
                "tmv_scores": [0.1],
                "custom_metrics": {"claim_metric": 0.64},
                "target_dataset_ids": ["claim_sim"],
                "claim_metric_evaluator": dict(evaluator_info),
            },
        )
        self.assertEqual(decision["decision"], "reject", decision)
        self.assertTrue(any("Stage 2 claim metric regression floor failed" in item for item in decision["decision_reasons"]))
        floor = decision["metrics_summary"]["stage2_claim_metric_regression_floor"]
        self.assertFalse(floor["ok"], floor)
        metric_check = next(item for item in floor["requirements"] if item.get("metric") == "claim_metric")
        self.assertAlmostEqual(metric_check["min"], 0.665)

    def test_stage3_budget_exhaustion_still_requires_inherited_stage2_locked_floor(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign["current_stage"] = "stage3_tuning"
        stage = campaign["stages"]["stage3_tuning"]
        stage["status"] = "active"
        stage["trial_count"] = campaign["stage_policies"]["stage3_tuning"]["max_trials"]
        stage["active_best_trial_id"] = "trial_stage3_active"
        stage["external_baseline_metrics"] = {
            "algorithm_name": "w1_sota",
            "w1_mean": 0.5,
            "target_dataset_ids": ["claim_sim"],
            "custom_metrics": {"claim_metric": 0.9, "total_score": 0.9},
            "claim_sota_baseline_metrics": {
                "algorithm_name": "claim_sota",
                "w1_mean": 0.5,
                "target_dataset_ids": ["claim_sim"],
                "custom_metrics": {"claim_metric": 0.9, "total_score": 0.9},
            },
            "w1_sota_baseline_metrics": {
                "algorithm_name": "w1_sota",
                "w1_mean": 0.5,
                "target_dataset_ids": ["claim_sim"],
                "custom_metrics": {"claim_metric": 0.9, "total_score": 0.9},
            },
        }
        campaign["trials"]["trial_stage3_active"] = {
            "trial_id": "trial_stage3_active",
            "metrics_summary": {
                "primary_metric": "w1_mean",
                "primary_value": 0.7,
                "w1_mean": 0.7,
                "tmv_mean": 0.1,
                "target_dataset_ids": ["claim_sim"],
                "custom_metrics": {"claim_metric": 0.72, "total_score": 0.65},
                "claim_metric_evaluator": dict(evaluator_info),
            },
        }
        self.tools.planner_file_tools._save_campaign(campaign)
        previous = os.environ.get("CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON")
        os.environ["CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON"] = json.dumps(
            {
                "version": "1.0",
                "stage": "stage2",
                "target_dataset_ids": ["claim_sim"],
                "requirements": {
                    "aggregate": [
                        {"metric": "total_score", "op": ">=", "value": 0.7},
                    ],
                    "require_exact_target_dataset_ids": True,
                },
            }
        )
        try:
            blocked = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
            self.assertFalse(blocked["ok"], blocked)
            self.assertIn("stage3_inherited_stage2_locked_gate_floor", blocked["checks"])
            self.assertFalse(blocked["checks"]["stage3_inherited_stage2_locked_gate_floor"]["ok"])

            campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
            campaign["trials"]["trial_stage3_active"]["metrics_summary"]["custom_metrics"]["total_score"] = 0.72
            self.tools.planner_file_tools._save_campaign(campaign)
            passed = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
            self.assertTrue(passed["ok"], passed)
            self.assertTrue(passed["checks"]["stage3_inherited_stage2_locked_gate_floor"]["ok"])
            self.assertFalse(passed["checks"]["stage3_optional_sota_gate"]["ok"])
        finally:
            if previous is None:
                os.environ.pop("CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON", None)
            else:
                os.environ["CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON"] = previous

    def test_stage3_optional_gate_uses_independent_claim_and_w1_sota_baselines(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign["current_stage"] = "stage3_tuning"
        stage = campaign["stages"]["stage3_tuning"]
        stage["status"] = "active"
        stage["active_best_trial_id"] = "trial_stage3"
        stage["external_baseline_metrics"] = {
            "algorithm_name": "legacy_claim_sota",
            "primary_metric": "w1_mean",
            "primary_value": 1.5,
            "w1_mean": 1.5,
            "custom_metrics": {"claim_metric": 0.7},
            "target_dataset_ids": ["claim_sim"],
            "claim_sota_baseline_metrics": {
                "algorithm_name": "claim_sota",
                "w1_mean": 1.5,
                "custom_metrics": {"claim_metric": 0.7},
                "target_dataset_ids": ["claim_sim"],
            },
            "w1_sota_baseline_metrics": {
                "algorithm_name": "w1_sota",
                "w1_mean": 0.8,
                "custom_metrics": {"claim_metric": 0.1},
                "target_dataset_ids": ["claim_sim"],
            },
        }
        campaign["trials"]["trial_stage3"] = {
            "trial_id": "trial_stage3",
            "metrics_summary": {
                "primary_metric": "w1_mean",
                "primary_value": 0.9,
                "w1_mean": 0.9,
                "custom_metrics": {"claim_metric": 0.72},
                "target_dataset_ids": ["claim_sim"],
                "claim_metric_evaluator": dict(evaluator_info),
            },
        }
        self.tools.planner_file_tools._save_campaign(campaign)

        gate = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertFalse(gate["ok"], gate)
        optional = gate["checks"]["stage3_optional_sota_gate"]
        self.assertFalse(optional["ok"], optional)
        self.assertFalse(optional["primary_ok"], optional)
        self.assertTrue(optional["claim_ok"], optional)
        self.assertEqual(optional["primary_baseline_algorithm"], "w1_sota")
        self.assertEqual(optional["claim_baseline_algorithm"], "claim_sota")

        campaign = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        campaign["trials"]["trial_stage3"]["metrics_summary"]["w1_mean"] = 0.7
        campaign["trials"]["trial_stage3"]["metrics_summary"]["primary_value"] = 0.7
        self.tools.planner_file_tools._save_campaign(campaign)
        fixed_w1 = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(fixed_w1["ok"], fixed_w1)
        self.assertTrue(fixed_w1["checks"]["stage3_optional_sota_gate"]["ok"])

    def test_final_regression_inherits_active_best_and_blocks_new_trials(self) -> None:
        evaluator_path, evaluator_info = self._write_claim_metric_evaluator(0.72)
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    "name": "claim_metric",
                    "direction": "greater",
                    "evaluator_path": str(evaluator_path),
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign["current_stage"] = "stage3_tuning"
        stage2_metrics = {
            "primary_metric": "claim_metric",
            "primary_direction": "greater",
            "primary_value": 0.72,
            "secondary_metric": "w1_mean",
            "secondary_direction": "lower",
            "secondary_value": 0.9,
            "w1_mean": 0.9,
            "tmv_mean": 0.1,
            "custom_metrics": {"claim_metric": 0.72},
            "target_dataset_ids": ["claim_sim"],
            "claim_metric_evaluator": dict(evaluator_info),
        }
        stage3_metrics = {
            "primary_metric": "w1_mean",
            "primary_direction": "lower",
            "primary_value": 0.7,
            "secondary_metric": "claim_metric",
            "secondary_direction": "greater",
            "secondary_value": 0.72,
            "w1_mean": 0.7,
            "tmv_mean": 0.1,
            "custom_metrics": {"claim_metric": 0.72},
            "target_dataset_ids": ["claim_sim"],
            "claim_metric_evaluator": dict(evaluator_info),
        }
        campaign["stages"]["stage2_claim_validation"].update(
            {
                "status": "passed",
                "stage_panel": {"target_dataset_ids": ["claim_sim"]},
                "stage_gate_evidence": {
                    "ok": True,
                    "active_best_trial_id": "trial_stage2",
                    "active_best_metrics_summary": dict(stage2_metrics),
                    "policy": dict(campaign["stage_policies"]["stage2_claim_validation"]),
                },
            }
        )
        stage3 = campaign["stages"]["stage3_tuning"]
        stage3["status"] = "active"
        stage3["active_best_trial_id"] = "trial_stage3_best"
        stage3["active_best_metrics_summary"] = dict(stage3_metrics)
        stage3["active_best_run_ids"] = ["run_stage3_best"]
        stage3["external_baseline_metrics"] = {
            "w1_sota_baseline_metrics": {
                "algorithm_name": "w1_sota",
                "w1_mean": 0.8,
                "target_dataset_ids": ["claim_sim"],
                "custom_metrics": {"claim_metric": 0.1},
            },
            "claim_sota_baseline_metrics": {
                "algorithm_name": "claim_sota",
                "w1_mean": 1.0,
                "target_dataset_ids": ["claim_sim"],
                "custom_metrics": {"claim_metric": 0.7},
            },
            "w1_mean": 0.8,
            "target_dataset_ids": ["claim_sim"],
            "custom_metrics": {"claim_metric": 0.7},
        }
        campaign["trials"]["trial_stage2"] = {
            "trial_id": "trial_stage2",
            "stage": "stage2_claim_validation",
            "status": "completed",
            "metrics_summary": dict(stage2_metrics),
        }
        campaign["trials"]["trial_stage3_best"] = {
            "trial_id": "trial_stage3_best",
            "stage": "stage3_tuning",
            "status": "completed",
            "decision": "promote",
            "run_ids": ["run_stage3_best"],
            "metrics_summary": dict(stage3_metrics),
        }
        self.tools.planner_file_tools._save_campaign(campaign)

        advanced = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=True))
        self.assertTrue(advanced["ok"], advanced)
        self.assertEqual(advanced["next_stage"], "final_regression")
        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(status["current_stage"], "final_regression")
        self.assertEqual(status["stages"]["final_regression"]["active_best_trial_id"], "trial_stage3_best")
        self.assertTrue(status["stages"]["final_regression"]["confirmation_only"])

        run_attempt = json.loads(self.tools.run_campaign_trial(campaign_id))
        self.assertEqual(run_attempt["status"], "blocked", run_attempt)
        self.assertEqual(run_attempt["reason"], "final_regression_confirmation_only")

        locked = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=True))
        self.assertTrue(locked["ok"], locked)
        final_status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(final_status["status"], "locked")
        self.assertEqual(final_status["locked_release"]["trial_id"], "trial_stage3_best")

    def test_final_regression_locks_without_external_baseline_gate(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    **self._claim_metric_spec(),
                    "stage_baselines": {
                        "stage1_feasibility": {"w1_mean": 1.0, "tmv_scores": [0.1]},
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        campaign["current_stage"] = "final_regression"
        campaign["stages"]["final_regression"]["status"] = "active"
        campaign["stages"]["final_regression"]["active_best_trial_id"] = "trial_final"
        campaign["trials"]["trial_final"] = {
            "trial_id": "trial_final",
            "metrics_summary": {
                "w1_mean": 6.5,
                "tmv_mean": 0.1,
                "target_dataset_ids": ["weinreb"],
                "custom_metrics": {"claim_metric": 0.8},
            },
        }
        self.tools.planner_file_tools._save_campaign(campaign)

        registry_before = self.tools.planner_file_tools._bootstrap_algorithm_registry("algo_campaign")
        self.assertEqual(registry_before["algorithm_lifecycle_status"], "developing")

        gate = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=False))
        self.assertTrue(gate["ok"], gate)
        self.assertFalse(gate["checks"]["final_regression_policy"]["pass_fail_gate"])
        self.assertNotIn("primary_vs_external_baseline", gate["checks"])

        locked = json.loads(self.tools.check_campaign_stage_gate(campaign_id, advance=True))
        self.assertTrue(locked["ok"], locked)
        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(status["status"], "locked")
        self.assertEqual(status["locked_release"]["trial_id"], "trial_final")
        registry_after = self.tools.planner_file_tools._bootstrap_algorithm_registry("algo_campaign")
        self.assertEqual(registry_after["algorithm_lifecycle_status"], "complete")
        self.assertEqual(registry_after["completed_campaign_id"], campaign_id)
        self.assertEqual(registry_after["completed_release"]["trial_id"], "trial_final")
        with self.assertRaises(ValueError):
            self.tools.planner_file_tools.mark_algorithm_failed(
                "algo_campaign",
                reason="completed releases must not be reclassified as failed",
            )

    def test_agent_can_mark_developing_algorithm_failed(self) -> None:
        self.tools.planner_file_tools.set_active_algorithm_context("algo_campaign")
        failure = json.loads(
            self.tools.mark_algorithm_failed(
                "algo_campaign",
                reason="component diagnostics show the current direction cannot satisfy the user goal",
                evidence=["diagnostics/component_gap.md"],
            )
        )

        self.assertEqual(failure["algorithm_lifecycle_status"], "failed")
        self.assertIn("revise", failure["next_required_action"])
        registry = self.tools.planner_file_tools._bootstrap_algorithm_registry("algo_campaign")
        self.assertEqual(registry["algorithm_lifecycle_status"], "failed")
        self.assertEqual(
            registry["algorithm_lifecycle_status_reason"],
            "component diagnostics show the current direction cannot satisfy the user goal",
        )
        active_context = self.state["active_algorithm_context"]
        self.assertEqual(active_context["algorithm_lifecycle_status"], "failed")
        history = json.loads(self.tools.list_experiment_history("algo_campaign"))
        self.assertEqual(history["algorithm_lifecycle_status"], "failed")
        self.assertTrue(
            any(item.get("decision") == "algorithm_failed" for item in history["recent_decisions"]),
            history["recent_decisions"],
        )

    def test_mark_algorithm_failed_closes_active_campaign_bookkeeping(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    **self._claim_metric_spec(),
                    "stage_baselines": {
                        "stage1_feasibility": {"w1_mean": 1.0, "tmv_scores": [0.1]},
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]

        failure = json.loads(
            self.tools.mark_algorithm_failed(
                "algo_campaign",
                reason="stage diagnostics show this campaign cannot satisfy the user goal",
                evidence=["campaign.json"],
            )
        )

        self.assertEqual(failure["closed_campaign_ids"], [campaign_id])
        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(status["status"], "failed")
        self.assertEqual(
            status["stages"]["stage1_feasibility"]["status"],
            "failed_needs_revision",
        )
        self.assertIn("stage diagnostics", status["failure_reason"])
        self.assertEqual(status["algorithm_lifecycle_status"], "failed")

    def test_campaign_promotion_clears_stale_failed_lifecycle_status(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    **self._claim_metric_spec(),
                    "stage_baselines": {
                        "stage1_feasibility": {"w1_mean": 1.0, "tmv_scores": [0.1]},
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]

        rejected_trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self.tools.planner_file_tools.prepare_campaign_trial(campaign_id)
        rejected = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=rejected_trial["trial_id"],
            metrics={"w1_scores": [0.9], "tmv_scores": [0.5], "custom_metrics": {"claim_metric": 0.5}},
        )
        self.assertEqual(rejected["decision"], "reject")
        registry_after_reject = self.tools.planner_file_tools._bootstrap_algorithm_registry("algo_campaign")
        self.assertEqual(registry_after_reject["algorithm_lifecycle_status"], "failed")

        promoted_trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self.tools.planner_file_tools.prepare_campaign_trial(campaign_id)
        promoted = self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=promoted_trial["trial_id"],
            metrics={"w1_scores": [0.8], "tmv_scores": [0.1], "custom_metrics": {"claim_metric": 0.6}},
        )
        self.assertEqual(promoted["decision"], "promote")

        registry_after_promote = self.tools.planner_file_tools._bootstrap_algorithm_registry("algo_campaign")
        self.assertEqual(registry_after_promote["algorithm_lifecycle_status"], "developing")
        self.assertIn("promoted trial", registry_after_promote["algorithm_lifecycle_status_reason"])
        self.assertEqual(registry_after_promote["completed_campaign_id"], "")

    def test_proposal_revision_resets_active_campaign_to_stage1(self) -> None:
        campaign = json.loads(
            self.tools.start_algorithm_campaign(
                "algo_campaign",
                claim_metric_spec={
                    **self._claim_metric_spec(),
                    "stage_baselines": {
                        "stage1_feasibility": {"w1_mean": 1.0, "tmv_scores": [0.1]},
                    },
                },
            )
        )
        campaign_id = campaign["campaign_id"]
        old_proposal_id = campaign["proposal_id"]

        trial = json.loads(self.tools.start_campaign_trial(campaign_id))
        self._replace_readme("# pre-reset best\n")
        self.tools.planner_file_tools.prepare_campaign_trial(campaign_id)
        self.tools.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial["trial_id"],
            metrics={"w1_scores": [0.9], "tmv_scores": [0.1], "custom_metrics": {"claim_metric": 0.5}},
        )
        self.assertTrue(json.loads(self.tools.check_campaign_stage_gate(campaign_id))["advanced"])

        proposal_payload = _valid_proposal_payload()
        revise_result = self._patch_proposal_text(
            proposal_payload["theoretical_core"],
            proposal_payload["theoretical_core"] + " This semantic revision restarts campaign validation.",
            "Change the algorithm semantics, so campaign evidence must restart.",
        )
        self.assertIn("auto-approved", revise_result)
        new_proposal_id = str(self.state.get("active_proposal_id") or "")
        self.assertNotEqual(new_proposal_id, old_proposal_id)

        status = json.loads(self.tools.get_algorithm_campaign_status(campaign_id))
        self.assertEqual(status["proposal_id"], new_proposal_id)
        self.assertEqual(status["current_stage"], "stage1_feasibility")
        self.assertFalse(status["algorithm_validated"])
        self.assertEqual(status["stages"]["stage1_feasibility"]["trial_count"], 0)
        self.assertEqual(status["stages"]["stage1_feasibility"]["active_best_trial_id"], "")
        self.assertEqual(status["stages"]["stage2_claim_validation"]["status"], "pending")
        self.assertEqual(status["reset_history"][-1]["previous_proposal_id"], old_proposal_id)
        self.assertEqual(status["reset_history"][-1]["proposal_id"], new_proposal_id)


class ClaimMetricCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.file_tools = planner_file_tools_module.PlannerFileTools(
            state={"output_dir": self.tmpdir.name},
            policy_provider=lambda: None,
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_claim_metric_evaluator_materializes_named_value_result(self) -> None:
        evaluator_path = Path(self.tmpdir.name) / "claim_metric_named_value.py"
        evaluator_path.write_text(
            "def evaluate_claim_metric(context):\n"
            "    return {'metric_name': 'claim_metric', 'value': 0.73, 'details': {'claim_metric': 0.73}}\n",
            encoding="utf-8",
        )
        hook, _ = load_campaign_claim_metric_evaluator(
            {
                "name": "claim_metric",
                "direction": "greater",
                "evaluator_path": f"{evaluator_path}:evaluate_claim_metric",
            }
        )

        metrics = hook(SimpleNamespace(timepoint_results=[]))

        self.assertEqual(metrics["claim_metric"], 0.73)
        self.assertEqual(metrics["metric_name"], "claim_metric")
        self.assertEqual(metrics["value"], 0.73)

    def test_metric_value_reads_named_value_custom_metric(self) -> None:
        metrics = {
            "custom_metrics": {
                "metric_name": "claim_metric",
                "value": 0.81,
                "details": {"claim_metric": 0.82},
            }
        }

        self.assertEqual(self.file_tools._metric_value(metrics, "claim_metric"), 0.81)

    def test_metric_value_reads_named_details_custom_metric(self) -> None:
        metrics = {
            "custom_metrics": {
                "metric_name": "claim_metric",
                "value": "not numeric",
                "details": {"claim_metric": 0.82},
            }
        }

        self.assertEqual(self.file_tools._metric_value(metrics, "claim_metric"), 0.82)

    def test_metric_value_ignores_mismatched_named_value_custom_metric(self) -> None:
        metrics = {
            "custom_metrics": {
                "metric_name": "other_metric",
                "value": 0.81,
                "details": {"claim_metric": 0.82},
            }
        }

        self.assertIsNone(self.file_tools._metric_value(metrics, "claim_metric"))


if __name__ == "__main__":
    unittest.main()
