from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from cytobridge_agent.tools import file_tools, planner_file_tools, workspace_policy
from cytobridge_agent.tools.workspace_policy import WorkspacePolicy


class PlannerFileToolsReadTests(unittest.TestCase):
    def _make_tools(self, root: str, events: list[tuple[str, dict]]) -> planner_file_tools.PlannerFileTools:
        workspace_root = Path(root).resolve()
        policy = WorkspacePolicy(
            workspace_root=workspace_root,
            output_root=workspace_root / "out",
            read_roots=[workspace_root],
            write_roots=[workspace_root],
            blocked_roots=[],
            unrestricted_reads=True,
        )
        return planner_file_tools.PlannerFileTools(
            state={},
            policy_provider=lambda: policy,
            event_sink=lambda event_type, payload: events.append((event_type, payload)),
        )

    def _make_owner_tools(
        self,
        root: str,
        session_id: str,
        events: list[tuple[str, dict]],
    ) -> planner_file_tools.PlannerFileTools:
        workspace_root = Path(root).resolve()
        policy = WorkspacePolicy(
            workspace_root=workspace_root,
            output_root=workspace_root / "out",
            read_roots=[workspace_root],
            write_roots=[workspace_root],
            blocked_roots=[],
            unrestricted_reads=True,
        )
        return planner_file_tools.PlannerFileTools(
            state={"session_id": session_id},
            policy_provider=lambda: policy,
            event_sink=lambda event_type, payload: events.append((event_type, payload)),
        )

    def test_algorithm_ownership_blocks_cross_session_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            events: list[tuple[str, dict]] = []
            with patch.object(planner_file_tools, "get_cellcompass_root", return_value=Path(tmpdir)):
                owner = self._make_owner_tools(tmpdir, "session-a", events)
                same_session = self._make_owner_tools(tmpdir, "session-a", events)
                other = self._make_owner_tools(tmpdir, "session-b", events)

                owner._ensure_algorithm_id_mutable("shared_algo", action="unit_test")
                same_session._ensure_algorithm_id_mutable("shared_algo", action="unit_test_again")
                owner._ensure_algorithm_id_mutable("second_algo", action="unit_test_new_algorithm")

                with self.assertRaisesRegex(ValueError, "owned by session 'session-a'"):
                    other._ensure_algorithm_id_mutable("shared_algo", action="cross_session_write")

            registry_path = Path(tmpdir) / "algorithm_ownership.json"
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["algorithms"]["shared_algo"]["owner_session_id"], "session-a")
            self.assertEqual(payload["algorithms"]["second_algo"]["owner_session_id"], "session-a")
            self.assertTrue(any(event_type == "algorithm_ownership_blocked" for event_type, _ in events))

    def test_workspace_policy_allows_paper_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir).resolve()
            output_root = workspace_root / "outputs"
            policy = WorkspacePolicy(
                workspace_root=workspace_root,
                output_root=output_root,
                read_roots=[workspace_root],
                write_roots=[],
                blocked_roots=[],
                unrestricted_reads=True,
            )

            target = output_root / "paper" / "main.tex"
            self.assertEqual(policy.validate_write_path(str(target)), target)

    def test_w1_backend_blocker_rejects_mixed_approximate_backends(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = self._make_tools(tmpdir, [])
            blockers = tools._w1_backend_provenance_blockers(
                candidate_metrics={
                    "w1_mean": 1.0,
                    "w1_backend": "geomloss_sinkhorn_online",
                    "w1_backend_exact": False,
                    "w1_backend_params": {"p": 1, "blur": 1.0, "backend": "online"},
                },
                baseline_metrics={
                    "w1_mean": 1.1,
                    "w1_backend": "exact",
                    "w1_backend_exact": True,
                    "w1_backend_params": {},
                },
                metric_name="w1_mean",
            )
        self.assertTrue(blockers)

    def test_w1_backend_blocker_rejects_missing_baseline_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = self._make_tools(tmpdir, [])
            blockers = tools._w1_backend_provenance_blockers(
                candidate_metrics={
                    "w1_mean": 1.0,
                    "w1_backend": "exact",
                    "w1_backend_exact": True,
                    "w1_backend_params": {},
                },
                baseline_metrics={"w1_mean": 1.1},
                metric_name="w1_mean",
            )
        self.assertTrue(blockers)
        self.assertTrue(any("lacks W1 backend provenance" in item for item in blockers))

    def test_w1_backend_blocker_rejects_missing_both_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = self._make_tools(tmpdir, [])
            blockers = tools._w1_backend_provenance_blockers(
                candidate_metrics={"w1_mean": 1.0},
                baseline_metrics={"w1_mean": 1.1},
                metric_name="w1_mean",
            )
        self.assertTrue(blockers)
        self.assertTrue(any("both lack W1 backend provenance" in item for item in blockers))

    def test_w1_backend_rehydrate_accepts_nested_baseline_summary_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = self._make_tools(tmpdir, [])
            metadata = tools._rehydrate_w1_backend_metadata_from_baseline_record(
                {
                    "algorithm_name": "builtin_wfrfm",
                    "baseline_type": "builtin",
                    "target_dataset_ids": ["toy"],
                    "w1_mean": 0.9,
                    "metrics_summary": {
                        "w1_mean": 0.9,
                        "w1_backend": "exact",
                        "w1_backend_exact": True,
                        "w1_backend_params": {},
                    },
                }
            )
        self.assertEqual(metadata["w1_backend"], "exact")
        self.assertTrue(metadata["w1_backend_exact"])
        self.assertEqual(metadata["w1_backend_params"], {})

    def test_workspace_policy_rejects_workspace_input_area_with_script_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir).resolve()
            output_root = workspace_root / "outputs"
            policy = WorkspacePolicy(
                workspace_root=workspace_root,
                output_root=output_root,
                read_roots=[workspace_root],
                write_roots=[],
                blocked_roots=[],
                unrestricted_reads=True,
            )

            with self.assertRaisesRegex(ValueError, r"_workspace/.*output_dir/scripts"):
                policy.validate_write_path(str(output_root / "_workspace" / "analysis.py"))

    def test_workspace_policy_allows_resolved_symlink_write_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir).resolve()
            real_cellcompass = workspace_root / "data_cellcompass"
            real_training = real_cellcompass / "training_algorithms"
            real_training.mkdir(parents=True)
            linked_cellcompass = workspace_root / ".cellcompass"
            linked_cellcompass.symlink_to(real_cellcompass, target_is_directory=True)
            output_root = workspace_root / "outputs"
            skills_root = linked_cellcompass / "skills"

            with (
                patch.object(workspace_policy, "get_cellcompass_root", return_value=linked_cellcompass),
                patch.object(workspace_policy, "get_cellcompass_skills_root", return_value=skills_root),
            ):
                policy = workspace_policy.build_planner_workspace_policy(
                    {"output_dir": str(output_root)},
                    workspace_root=workspace_root,
                )

            target = real_training / "toy_algo" / "algorithm.py"
            self.assertEqual(policy.validate_write_path(str(target)), target)

    def test_workspace_policy_allows_shared_portfolio_ledger_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir).resolve()
            output_root = workspace_root / "outputs"
            portfolio_root = workspace_root / "data" / "cytobridge" / "portfolio"
            portfolio_root.mkdir(parents=True)
            skills_root = workspace_root / ".cellcompass" / "skills"

            with (
                patch.object(workspace_policy, "get_cellcompass_root", return_value=workspace_root / ".cellcompass"),
                patch.object(workspace_policy, "get_cellcompass_skills_root", return_value=skills_root),
                patch.object(workspace_policy, "get_cytobridge_portfolio_root", return_value=portfolio_root),
            ):
                policy = workspace_policy.build_planner_workspace_policy(
                    {"output_dir": str(output_root)},
                    workspace_root=workspace_root,
                )

            target = portfolio_root / "shared_gap_ledger_en.md"
            self.assertEqual(policy.validate_write_path(str(target)), target)

    def test_read_workspace_file_routes_through_read_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir).resolve() / "sample.txt"
            target.write_text("alpha\nbeta\n", encoding="utf-8")
            events: list[tuple[str, dict]] = []
            tools = self._make_tools(tmpdir, events)

            with patch.object(planner_file_tools, "read_file_impl", return_value="hello world") as mock_read_file:
                result = tools.read_workspace_file(str(target), start_line=5, end_line=None, max_chars=321)

            self.assertEqual(result, "hello world")
            mock_read_file.assert_called_once_with(
                file_path=str(target),
                offset=5,
                limit=None,
                max_chars=321,
                default_text_limit=None,
            )
            self.assertEqual(events, [("workspace_file_read", {
                "scope": "planner",
                "path": str(target),
                "start_line": 5,
                "end_line": None,
                "max_chars": 321,
                "preview": "hello world",
            })])

    def test_read_workspace_file_preserves_unbounded_line_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir).resolve() / "many_lines.txt"
            target.write_text("\n".join(f"line {idx}" for idx in range(1, 2601)), encoding="utf-8")
            events: list[tuple[str, dict]] = []
            tools = self._make_tools(tmpdir, events)

            with patch.object(file_tools, "READ_TEXT_FAST_PATH_MAX_BYTES", 1):
                result = tools.read_workspace_file(str(target), start_line=2500, end_line=None, max_chars=20000)

            self.assertIn("Line range: 2500-2600", result)
            self.assertIn("L2500: line 2500", result)
            self.assertIn("L2600: line 2600", result)
            self.assertNotIn("L1: line 1", result)

    def test_patch_algorithm_config_dry_run_apply_and_list_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            events: list[tuple[str, dict]] = []
            tools = self._make_tools(tmpdir, events)
            cellcompass_root = root / ".cellcompass"
            algo_dir = cellcompass_root / "training_algorithms" / "toy_algo"
            algo_dir.mkdir(parents=True)
            (algo_dir / "PROPOSAL.json").write_text(
                json.dumps({"algorithm_id": "toy_algo", "proposal_id": "proposal_test", "status": "approved"}),
                encoding="utf-8",
            )
            config_path = algo_dir / "config.yaml"
            config_path.write_text(
                "training:\n"
                "  plan:\n"
                "  - lr: 0.001\n"
                "    batch_size: 128\n"
                "    epochs: 3000\n"
                "model:\n"
                "  hidden: 64\n",
                encoding="utf-8",
            )
            tools.state["active_algorithm_context"] = {"algorithm_id": "toy_algo"}

            with patch.object(planner_file_tools, "get_cellcompass_root", return_value=cellcompass_root):
                dry_run = tools.patch_algorithm_config(
                    updates={"training.plan[0].lr": 0.002},
                    dry_run=True,
                    reason="tune lr",
                )
                self.assertTrue(dry_run["ok"])
                self.assertEqual(dry_run["status"], "dry_run")
                self.assertIn("-  - lr: 0.001", dry_run["diff"])
                self.assertIn("+  - lr: 0.002", dry_run["diff"])
                self.assertIn("lr: 0.001", config_path.read_text(encoding="utf-8"))

                applied = tools.patch_algorithm_config(
                    updates={"training.plan.0.lr": 0.002},
                    reason="apply lr tune",
                )
                self.assertTrue(applied["ok"])
                self.assertEqual(applied["status"], "applied")
                self.assertIn("lr: 0.002", config_path.read_text(encoding="utf-8"))

                blocked = tools.patch_algorithm_config(
                    updates={"training.plan": [{"lr": 9.0}]},
                    dry_run=False,
                    reason="bad whole-list replacement",
                )
                self.assertFalse(blocked["ok"])
                self.assertEqual(blocked["status"], "blocked")
                self.assertIn("whole list", blocked["blocked_updates"][0]["reason"])
                self.assertNotIn("lr: 9.0", config_path.read_text(encoding="utf-8"))

                same_epoch = tools.patch_algorithm_config(
                    updates={"training.plan[0].epochs": 3000},
                    dry_run=True,
                    reason="redundant default epoch declaration",
                )
                self.assertTrue(same_epoch["ok"])
                self.assertEqual(same_epoch["status"], "dry_run")

                epoch_blocked = tools.patch_algorithm_config(
                    updates={"training.plan[0].epochs": 1},
                    dry_run=False,
                    reason="bad epoch shortcut",
                )
                self.assertFalse(epoch_blocked["ok"])
                self.assertIn("epoch override changes", epoch_blocked["blocked_updates"][0]["reason"])

    def test_abort_current_campaign_trial_clears_stale_running_trial_without_counting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            events: list[tuple[str, dict]] = []
            tools = self._make_tools(tmpdir, events)
            cellcompass_root = root / ".cellcompass"
            campaign_dir = cellcompass_root / "algorithm_campaigns" / "toy_algo" / "campaign_test"
            campaign_dir.mkdir(parents=True)
            campaign_path = campaign_dir / "campaign.json"
            campaign_path.write_text(
                json.dumps(
                    {
                        "campaign_id": "campaign_test",
                        "algorithm_id": "toy_algo",
                        "proposal_id": "proposal_test",
                        "status": "active",
                        "current_stage": "stage3_tuning",
                        "current_trial_id": "trial_running",
                        "stages": {
                            "stage3_tuning": {
                                "status": "active",
                                "trial_count": 3,
                                "promote_count": 0,
                                "reject_count": 3,
                                "active_best_trial_id": "trial_best",
                                "active_best_snapshot_id": "snapshot_best",
                            }
                        },
                        "trials": {
                            "trial_best": {
                                "trial_id": "trial_best",
                                "stage": "stage2_claim_validation",
                                "status": "completed",
                                "decision": "promote",
                            },
                            "trial_running": {
                                "trial_id": "trial_running",
                                "stage": "stage3_tuning",
                                "status": "running",
                                "decision": "",
                                "run_ids": [],
                                "snapshot_id": "snapshot_running",
                                "created_at": "2026-05-05T00:00:00+00:00",
                                "updated_at": "2026-05-05T00:00:01+00:00",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            tools.state["active_algorithm_campaign_id"] = "campaign_test"
            tools.state["active_algorithm_campaign"] = {
                "campaign_id": "campaign_test",
                "algorithm_id": "toy_algo",
            }

            with patch.object(planner_file_tools, "get_cellcompass_root", return_value=cellcompass_root):
                payload = tools.abort_current_campaign_trial(
                    "campaign_test",
                    reason="stale running trial after interrupted process",
                    restore_active_best=False,
                )

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["status"], "aborted")
            self.assertEqual(payload["trial_id"], "trial_running")
            self.assertFalse(payload["restored_active_best"])
            updated = json.loads(campaign_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["current_trial_id"], "")
            self.assertEqual(updated["trials"]["trial_running"]["status"], "aborted")
            self.assertEqual(updated["trials"]["trial_running"]["decision"], "aborted")
            self.assertEqual(updated["trials"]["trial_running"]["original_status"], "running")
            self.assertEqual(updated["stages"]["stage3_tuning"]["trial_count"], 3)
            self.assertEqual(updated["stages"]["stage3_tuning"]["reject_count"], 3)
            self.assertIn("stale running trial", updated["trials"]["trial_running"]["abort_reason"])
            self.assertTrue((campaign_dir / "trials" / "trial_running.json").exists())
            self.assertIn("algorithm_campaign_trial_aborted", [event_type for event_type, _payload in events])

    def test_switch_campaign_stage_panel_preserves_stage_counts_and_invalidates_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            events: list[tuple[str, dict]] = []
            tools = self._make_tools(tmpdir, events)
            cellcompass_root = root / ".cellcompass"
            campaign_dir = cellcompass_root / "algorithm_campaigns" / "toy_algo" / "campaign_test"
            campaign_dir.mkdir(parents=True)
            campaign_path = campaign_dir / "campaign.json"
            campaign_path.write_text(
                json.dumps(
                    {
                        "campaign_id": "campaign_test",
                        "algorithm_id": "toy_algo",
                        "proposal_id": "proposal_test",
                        "status": "active",
                        "current_stage": "stage3_tuning",
                        "current_trial_id": "",
                        "stages": {
                            "stage3_tuning": {
                                "status": "active",
                                "trial_count": 8,
                                "promote_count": 1,
                                "reject_count": 7,
                                "active_best_trial_id": "trial_best",
                                "active_best_snapshot_id": "snapshot_best",
                                "active_best_commit": "abc123",
                                "active_best_run_ids": ["run_best"],
                                "active_best_metrics_summary": {
                                    "w1_mean": 0.1,
                                    "target_dataset_ids": ["old_dataset"],
                                },
                                "external_baseline_metrics": {
                                    "w1_mean": 0.2,
                                    "target_dataset_ids": ["old_dataset"],
                                },
                                "gate_ready": True,
                                "last_gate_check": {"ok": True},
                                "stage_panel": {
                                    "target_dataset_ids": ["old_dataset"],
                                    "dataset_config_overrides": {
                                        "datasets": [{"dataset_id": "old_dataset"}],
                                        "target_dataset_ids": ["old_dataset"],
                                    },
                                    "source": "manual",
                                    "frozen_at": "2026-05-05T00:00:00+00:00",
                                },
                            }
                        },
                        "trials": {
                            "trial_best": {
                                "trial_id": "trial_best",
                                "stage": "stage3_tuning",
                                "status": "completed",
                                "decision": "promote",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            tools.state["active_algorithm_campaign_id"] = "campaign_test"
            tools.state["active_algorithm_campaign"] = {
                "campaign_id": "campaign_test",
                "algorithm_id": "toy_algo",
            }

            with patch.object(planner_file_tools, "get_cellcompass_root", return_value=cellcompass_root):
                payload = tools.switch_campaign_stage_panel(
                    "campaign_test",
                    stage="stage3_tuning",
                    dataset_config_overrides={
                        "datasets": [{"dataset_id": "new_dataset"}],
                    },
                    reason="add real benchmark panel",
                )

            self.assertEqual(payload["status"], "switched")
            self.assertEqual(payload["from_target_dataset_ids"], ["old_dataset"])
            self.assertEqual(payload["target_dataset_ids"], ["new_dataset"])
            self.assertEqual(payload["trial_count_preserved"], 8)
            updated = json.loads(campaign_path.read_text(encoding="utf-8"))
            stage_state = updated["stages"]["stage3_tuning"]
            self.assertEqual(stage_state["trial_count"], 8)
            self.assertEqual(stage_state["promote_count"], 1)
            self.assertEqual(stage_state["reject_count"], 7)
            self.assertEqual(stage_state["stage_panel"]["target_dataset_ids"], ["new_dataset"])
            self.assertNotIn("active_best_trial_id", stage_state)
            self.assertEqual(stage_state["active_best_run_ids"], [])
            self.assertEqual(stage_state["external_baseline_metrics"], {})
            self.assertFalse(stage_state["gate_ready"])
            self.assertFalse(stage_state["last_gate_check"]["ok"])
            self.assertEqual(stage_state["previous_panel_active_best"]["active_best_trial_id"], "trial_best")
            self.assertIn("algorithm_campaign_stage_panel_switched", [event_type for event_type, _payload in events])

    def test_create_algorithm_workspace_artifact_file_directory_and_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            events: list[tuple[str, dict]] = []
            tools = self._make_tools(tmpdir, events)
            cellcompass_root = root / ".cellcompass"
            algo_dir = cellcompass_root / "training_algorithms" / "toy_algo"
            algo_dir.mkdir(parents=True)
            (algo_dir / "PROPOSAL.json").write_text(
                json.dumps({"algorithm_id": "toy_algo", "proposal_id": "proposal_test", "status": "approved"}),
                encoding="utf-8",
            )
            tools.state["active_algorithm_context"] = {"algorithm_id": "toy_algo"}

            with patch.object(planner_file_tools, "get_cellcompass_root", return_value=cellcompass_root):
                directory_result = tools.create_algorithm_workspace_artifact(
                    "diagnostics",
                    kind="directory",
                    reason="prepare risk diagnostics",
                )
                self.assertIn("Created directory", directory_result)
                self.assertTrue((algo_dir / "diagnostics").is_dir())

                file_result = tools.create_algorithm_workspace_artifact(
                    "diagnostics/check_mass_drift.py",
                    kind="file",
                    content="print('ok')\n",
                    reason="diagnose mass drift",
                )
                self.assertIn("Wrote file", file_result)
                self.assertEqual((algo_dir / "diagnostics" / "check_mass_drift.py").read_text(encoding="utf-8"), "print('ok')\n")

                traversal_result = tools.create_algorithm_workspace_artifact(
                    "../escape.py",
                    kind="file",
                    content="bad\n",
                )
                self.assertIn("relative_path must be a clean path", traversal_result)
                self.assertFalse((cellcompass_root / "training_algorithms" / "escape.py").exists())

                other_dir = cellcompass_root / "training_algorithms" / "other_algo"
                other_dir.mkdir(parents=True)
                (other_dir / "PROPOSAL.json").write_text(
                    json.dumps({"algorithm_id": "other_algo", "proposal_id": "proposal_other", "status": "approved"}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "active authoring"):
                    tools.create_algorithm_workspace_artifact(
                        "diagnostics/check.py",
                        kind="file",
                        content="print('other')\n",
                        algorithm_id="other_algo",
                    )
                self.assertFalse((other_dir / "diagnostics" / "check.py").exists())

                event_types = [event_type for event_type, _payload in events]
                self.assertIn("workspace_directory_created", event_types)
                self.assertIn("workspace_file_written", event_types)
                self.assertTrue((tools.state.get("active_algorithm_context") or {}).get("dirty_since_snapshot"))

    def test_record_implementation_review_updates_risk_markdown_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            events: list[tuple[str, dict]] = []
            tools = self._make_tools(tmpdir, events)
            cellcompass_root = root / ".cellcompass"
            algo_dir = cellcompass_root / "training_algorithms" / "toy_algo"
            algo_dir.mkdir(parents=True)
            risk_path = algo_dir / "risk.md"
            risk_path.write_text(
                "# Proposal Risk Assessment: toy_algo\n\n"
                "## Structured Risk Assessment\n"
                "Proposal reviewer risk stays here.\n",
                encoding="utf-8",
            )

            with patch.object(planner_file_tools, "get_cellcompass_root", return_value=cellcompass_root):
                record = tools.record_implementation_review(
                    "toy_algo",
                    proposal_id="proposal_test",
                    review_hash="hash_test",
                    decision="approve",
                    reviewer_feedback="Implementation follows the proposal.",
                    advisory_risks=["Chunk weighting should be monitored against builtin UOT sampling."],
                    efficiency_recommendations=["Reuse _solve_uot_from_pairwise_cost for GPU POT solves."],
                    generalization_shortcut_risks=["No warm-start shortcut detected."],
                    risk_assessment="### Chunked OT alignment\n\nCheck per-chunk mass weighting in diagnostics.",
                )

            self.assertEqual(record["implementation_risk_path"], str(risk_path))
            text = risk_path.read_text(encoding="utf-8")
            self.assertIn("Proposal reviewer risk stays here.", text)
            self.assertIn("Implementation Reviewer Risk Assessment", text)
            self.assertIn("Chunk weighting should be monitored", text)
            self.assertIn("_solve_uot_from_pairwise_cost", text)
            self.assertIn("Chunked OT alignment", text)

    def test_algorithm_benchmark_registration_rejects_unprepared_data(self) -> None:
        try:
            import anndata as ad  # type: ignore
            import numpy as np  # type: ignore
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"anndata/numpy/pandas unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir).resolve() / "source.h5ad"
            obs = pd.DataFrame({"samples": [0, 0, 1, 1]}, index=[f"cell{i}" for i in range(4)])
            ad.AnnData(X=np.asarray([[0.0, 0.1], [0.2, 0.3], [1.0, 1.1], [1.2, 1.3]]), obs=obs).write_h5ad(source)
            events: list[tuple[str, dict]] = []
            tools = self._make_tools(tmpdir, events)
            cellcompass_root = Path(tmpdir).resolve() / ".cellcompass"

            with patch.object(planner_file_tools, "get_cellcompass_root", return_value=cellcompass_root):
                registered = tools.register_algorithm_benchmark_dataset(
                    "toy_2d",
                    str(source),
                    tags=["simulation", "tiny"],
                    stage_relevance=["stage1_feasibility"],
                )
                self.assertEqual(registered["status"], "invalid_contract")
                self.assertIn("adata.obs['time_point_processed']", registered["missing_fields"])
                self.assertIn("adata.obsm['X_latent']", registered["missing_fields"])
                self.assertFalse((cellcompass_root / "algorithm_benchmarks" / "datasets" / "toy_2d" / "dataset.json").exists())

            self.assertEqual(events[0][0], "algorithm_benchmark_dataset_rejected")

    def test_algorithm_benchmark_registration_and_dataset_config(self) -> None:
        try:
            import anndata as ad  # type: ignore
            import numpy as np  # type: ignore
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"anndata/numpy/pandas unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir).resolve() / "source.h5ad"
            preprocessing_script = Path(tmpdir).resolve() / "prepare_toy_2d.py"
            preprocessing_script.write_text(
                "import anndata as ad\n"
                "# Reproducible preprocessing script captured with the benchmark.\n",
                encoding="utf-8",
            )
            obs = pd.DataFrame(
                {"time_point_processed": [0, 0, 1, 1]},
                index=[f"cell{i}" for i in range(4)],
            )
            adata = ad.AnnData(
                X=np.asarray([[0.0, 0.1], [0.2, 0.3], [1.0, 1.1], [1.2, 1.3]]),
                obs=obs,
            )
            adata.obsm["X_latent"] = np.asarray(adata.X)
            adata.write_h5ad(source)
            events: list[tuple[str, dict]] = []
            tools = self._make_tools(tmpdir, events)
            cellcompass_root = Path(tmpdir).resolve() / ".cellcompass"

            with patch.object(planner_file_tools, "get_cellcompass_root", return_value=cellcompass_root):
                registered = tools.register_algorithm_benchmark_dataset(
                    "toy_2d",
                    str(source),
                    tags=["simulation", "tiny"],
                    stage_relevance=["stage1_feasibility"],
                    preprocessing_script_path=str(preprocessing_script),
                )
                self.assertEqual(registered["status"], "registered")
                card = registered["dataset"]
                self.assertEqual(card["contract_status"], "ready")
                self.assertEqual(card["transformations"], [])
                copied_script_path = Path(card["paths"]["preprocessing_script_path"])
                self.assertTrue(copied_script_path.exists())
                self.assertEqual(copied_script_path.read_text(encoding="utf-8"), preprocessing_script.read_text(encoding="utf-8"))
                self.assertEqual(
                    card["provenance"]["preprocessing_script"]["sha256"],
                    planner_file_tools.PlannerFileTools._file_sha256(preprocessing_script),
                )
                copied = ad.read_h5ad(card["paths"]["data_path"])
                self.assertIn("time_point_processed", copied.obs)
                self.assertIn("X_latent", copied.obsm)
                self.assertEqual(copied.obsm["X_latent"].shape, (4, 2))

                listing = tools.list_algorithm_benchmarks(stage="stage1_feasibility", ready_only=True)
                self.assertEqual(listing["count"], 1)
                self.assertEqual(listing["datasets"][0]["dataset_id"], "toy_2d")
                self.assertEqual(listing["datasets"][0]["preprocessing_script_path"], str(copied_script_path))

                config = tools.make_benchmark_dataset_config(dataset_ids=["toy_2d"])
                dataset_entry = config["dataset_config_overrides"]["datasets"][0]
                self.assertEqual(dataset_entry["dataset_id"], "toy_2d")
                self.assertEqual(dataset_entry["adata_path"], card["paths"]["data_path"])
                self.assertEqual(config["dataset_config_overrides"]["target_dataset_ids"], ["toy_2d"])

                baselines = tools.get_algorithm_benchmark_baselines("toy_2d")
                self.assertEqual(baselines["baseline_count"], 0)
                self.assertIn("No builtin baselines", baselines["message"])

                run_dir = Path(tmpdir).resolve() / "baseline_run"
                (run_dir / "artifacts").mkdir(parents=True)
                (run_dir / "logs").mkdir(parents=True)
                metrics_source = run_dir / "artifacts" / "metrics.json"
                model_source = run_dir / "artifacts" / "trained_model.h5ad"
                config_source = run_dir / "resolved_config.yaml"
                metrics_source.write_text('{"w1_scores":[0.2,0.4],"tmv_scores":[0.05,0.1]}', encoding="utf-8")
                config_source.write_text("training:\n  seed: 1\n", encoding="utf-8")
                shutil_source = Path(card["paths"]["data_path"])
                model_source.write_bytes(shutil_source.read_bytes())
                (run_dir / "run_manifest.json").write_text('{"run_id":"run_crufm"}', encoding="utf-8")
                (run_dir / "logs" / "training.log").write_text("ok\n", encoding="utf-8")
                (run_dir / "logs" / "planner_context.json").write_text("{}", encoding="utf-8")
                recorded = tools.record_algorithm_benchmark_baseline(
                    "toy_2d",
                    "crufm",
                    {
                        "w1_scores": [0.2, 0.4],
                        "tmv_scores": [0.05, 0.1],
                        "runtime_sec": 3.0,
                        "run_dir": str(run_dir),
                        "metrics_path": str(metrics_source),
                        "trained_model_path": str(model_source),
                        "resolved_config_path": str(config_source),
                        "w1_backend": "exact",
                        "w1_backend_exact": True,
                        "w1_backend_params": {},
                    },
                    run_id="run_crufm",
                )
                self.assertEqual(recorded["status"], "recorded")
                self.assertAlmostEqual(recorded["metrics_summary"]["w1_mean"], 0.3)
                self.assertIn("/baseline_artifacts/builtin_crufm/", recorded["artifacts"]["trained_model_path"])
                self.assertTrue(Path(recorded["artifacts"]["trained_model_path"]).exists())
                self.assertTrue(Path(recorded["artifacts"]["training_log_path"]).exists())
                baselines = tools.get_algorithm_benchmark_baselines("toy_2d")
                self.assertEqual(baselines["baseline_count"], 1)
                self.assertEqual(baselines["leaderboard"]["entries"][0]["algorithm_name"], "crufm")
                baseline_list = tools.list_algorithm_benchmark_baselines("toy_2d")
                self.assertEqual(baseline_list["baseline_count"], 1)
                self.assertEqual(baseline_list["baselines"][0]["algorithm_name"], "crufm")
                self.assertEqual(baseline_list["baselines"][0]["w1_backend"], "exact")
                self.assertTrue(baseline_list["baselines"][0]["artifacts_available"]["trained_model"])
                self.assertTrue(baseline_list["baselines"][0]["ckpt_path"].endswith("trained_model.h5ad"))

                skipped = tools.record_algorithm_benchmark_baseline(
                    "toy_2d",
                    "crufm",
                    {
                        "status": "completed",
                        "runtime_sec": 1.0,
                        "run_dir": str(run_dir),
                    },
                    run_id="run_crufm_incomplete",
                )
                self.assertEqual(skipped["status"], "skipped_incomplete_record_preserved")
                baselines_after_skip = tools.get_algorithm_benchmark_baselines("toy_2d")
                preserved = baselines_after_skip["baseline_records"][0]
                self.assertEqual(preserved["run_id"], "run_crufm")
                self.assertAlmostEqual(preserved["w1_mean"], 0.3)
                self.assertEqual(preserved["w1_backend"], "exact")

            self.assertEqual(events[0][0], "algorithm_benchmark_dataset_registered")
            self.assertEqual(events[0][1]["dataset_id"], "toy_2d")
            self.assertEqual(events[1][0], "algorithm_benchmark_baseline_recorded")

    def test_stage2_simulation_registration_versions_generator_and_dataset_config(self) -> None:
        try:
            import anndata as ad  # type: ignore
            import numpy as np  # type: ignore
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"anndata/numpy/pandas unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source = root / "claim_sim.h5ad"
            generator = root / "make_claim_sim.py"
            generator.write_text("SEED = 7\n# generate claim simulation\n", encoding="utf-8")
            obs = pd.DataFrame(
                {"time_point_processed": [0, 0, 1, 1]},
                index=[f"cell{i}" for i in range(4)],
            )
            adata = ad.AnnData(
                X=np.asarray([[0.0, 0.1], [0.2, 0.3], [1.0, 1.1], [1.2, 1.3]]),
                obs=obs,
            )
            adata.obsm["X_latent"] = np.asarray(adata.X)
            adata.write_h5ad(source)
            events: list[tuple[str, dict]] = []
            tools = self._make_tools(tmpdir, events)
            cellcompass_root = root / ".cellcompass"

            with patch.object(planner_file_tools, "get_cellcompass_root", return_value=cellcompass_root):
                registered = tools.register_stage2_simulation_dataset(
                    "claim_sim",
                    str(source),
                    str(generator),
                    algorithm_id="algo_claim",
                    proposal_id="proposal_1",
                    claim_metric_name="branch_recovery",
                )
                self.assertEqual(registered["status"], "registered")
                self.assertTrue(registered["simulation_version"].startswith("claim_sim_"))
                card = tools.get_algorithm_benchmark_dataset("claim_sim")
                self.assertEqual(card["simulation"]["simulation_version"], registered["simulation_version"])
                self.assertIn("Stage 2 Simulation", card["readme"])
                self.assertTrue(Path(card["simulation"]["generator_copy_path"]).exists())

                config = tools.make_benchmark_dataset_config(dataset_ids=["claim_sim"])
                entry = config["dataset_config_overrides"]["datasets"][0]
                self.assertEqual(entry["simulation_version"], registered["simulation_version"])
                self.assertEqual(entry["simulation_generator"]["generator_sha256"], card["simulation"]["generator_sha256"])
                self.assertEqual(config["dataset_config_overrides"]["target_dataset_ids"], ["claim_sim"])

                registry = tools.ensure_algorithm_benchmark_registry()
                self.assertIn(registered["simulation_version"], registry["simulation_generators"])

            self.assertEqual(events[0][0], "algorithm_benchmark_dataset_registered")
            self.assertEqual(events[1][0], "stage2_simulation_dataset_registered")


if __name__ == "__main__":
    unittest.main()
