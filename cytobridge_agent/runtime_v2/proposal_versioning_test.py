from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cytobridge_agent.tools import planner_file_tools as planner_file_tools_module
from cytobridge_agent.tools import planner_tools as planner_tools_module
from cytobridge_agent.tools import workspace_policy as workspace_policy_module
from cytobridge_agent.runtime_v2.state import ensure_runtime_v2_state
from cytobridge_agent.runtime_v2.tool_registry import (
    SingleAgentTools,
    _CreateAlgorithmProposalInput,
)


class DummyLLM:
    pass


def _valid_proposal_payload() -> dict[str, str]:
    return {
        "abstract": (
            "This proposal learns coupling-aligned interval dynamics so that the exact-fit limit recovers "
            "weighted temporal marginals and, when configured, total-mass changes."
        ),
        "literature_and_package_grounding": (
            "Checked package builtin semantics for dynamical_ot, ot_cfm, sf2m, vgfm, crufm, ruot, unbalanced_ot, "
            "and cyto_simulation, plus the local survey notes. Relevant citations: "
            "1. TrajectoryNet - dynamic OT neural formulation; "
            "2. MIOFlow - manifold-aware temporal flow baseline; "
            "3. Conditional Flow Matching Simulation-Free Dynamic Optimal Transport - OT-CFM objective; "
            "4. Flow Matching for Generative Modeling - conditional FM foundation; "
            "5. Simulation-free Schrödinger bridges via score and flow matching - stochastic bridge baseline; "
            "6. Reconstructing growth and dynamic trajectories from single-cell transcriptomics data - growth/UOT baseline; "
            "7. Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport - RUOT comparison; "
            "8. Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge - unbalanced MFSB context; "
            "9. Joint Velocity-Growth Flow Matching for Single-Cell Dynamics Modeling - VGFM-style velocity/growth semantics; "
            "10. Computational Optimal Transport - transport objective reference. "
            "The remaining gap is coupling-aligned interval supervision with explicit weighted recovery semantics."
        ),
        "problem_statement": (
            "The algorithm addresses temporal snapshot reconstruction when endpoint distributions and optional "
            "mass changes must be matched by a single learned dynamical model."
        ),
        "problem_mathematical_form": (
            "For adjacent snapshots, fit a path-induced predictor whose terminal weighted measure equals the "
            "target empirical measure, optionally with learned log-mass dynamics."
        ),
        "claimed_capability": (
            "The method claims to recover adjacent-time weighted distributions and optional total-mass changes "
            "from coupling-aligned supervision; W1 and TMV validate those claims."
        ),
        "objective": "Learn interval dynamics whose exact-fit limit recovers the observed weighted temporal marginals.",
        "theoretical_core": (
            "We fit a conditional velocity-growth family whose supervision is defined directly from adjacent-time transport couplings, "
            "so exact fit implies recovery of the claimed temporal target measure."
        ),
        "mathematical_abstraction": (
            "For each interval we define a coupling pi_k over source-target cells, a conditional path law, "
            "a velocity field, and when enabled a mass-growth field over weighted particles."
        ),
        "algorithm_semantics_table": (
            "| Mathematical Object | Runtime Layer | Semantic Definition |\n"
            "| --- | --- | --- |\n"
            "| Coupling pi_k | backend | Adjacent-time transport plan used for pair supervision |\n"
            "| Conditional path | path layer | Time interpolation law between sampled endpoints |\n"
            "| Growth field | mass layer | Log-mass dynamics for weighted particles when enabled |\n"
        ),
        "implementation_pseudocode": (
            "Inputs: source cells, target cells, times, coupling plan\n"
            "Outputs: fitted velocity-growth model, weighted inference particles\n"
            "Definitions: pi_k is the adjacent-time coupling and rho_t is the induced path law\n"
            "P1: Build interval supervision\n"
            "Inputs: source cells, target cells, pi_k\n"
            "Outputs: supervised pairs and path states\n"
            "Invariants: pair sampling stays aligned with pi_k and the declared target semantics\n"
            "Forbidden deviations: replacing pi_k-aligned supervision with unrelated heuristics\n"
            "Equation: pair_loss = E_(i,j)~pi_k [||v_theta - v_star||^2]\n"
            "P2: Fit weighted dynamics\n"
            "Inputs: path states, mass targets\n"
            "Outputs: velocity and optional growth parameters\n"
            "Invariants: optimization targets preserve the proposal's recoverability argument\n"
            "Forbidden deviations: changing endpoint semantics without revising the proposal\n"
            "Equation: total_loss = pair_loss + lambda * growth_loss\n"
        ),
        "evaluation_plan": (
            "Keep builtin W1/TMV fixed, compare against the active baseline, and add only proposal-specific diagnostics "
            "that test whether the claimed weighted temporal measure is recovered."
        ),
        "expected_evaluation_outcome": (
            "The method should work best on adjacent-time snapshot panels where coupling structure is informative and "
            "mass changes, when enabled, are driven by smooth growth factors. Expected evidence is lower W1 on held-out "
            "interval rollouts, TMV below the campaign threshold for unbalanced runs, and a proposal-specific coupling "
            "alignment diagnostic above the strongest relevant builtin baseline because the supervision follows pi_k."
        ),
        "inductive_generalization_argument": (
            "At inference time the learned velocity-growth rule consumes only a new valid t0 particle state, time, "
            "allowed context, config, and trained parameters. The coupling pi_k is training-only supervision; no cell id, "
            "row id, barcode lookup, future snapshot, or target-specific table is used to roll out new particles."
        ),
        "mass_modeling_scope": "models_unbalanced_mass",
        "unbalanced_decision": "Enable unbalanced mass modeling because the method explicitly aims to match observed total-mass change.",
        "stochasticity_decision": "Use deterministic dynamics first; only add stochasticity if the exact-fit argument requires it.",
        "distribution_recovery_argument": (
            "If the coupling-aligned supervision is fit exactly, the induced endpoint weighted particle measure matches the target interval marginal. "
            "Because the growth target is defined on the same coupling, the exact-fit limit also recovers the claimed total-mass change."
        ),
        "overengineering_self_check": (
            "Do not add auxiliary latent machinery unless the recoverability argument fails without it."
        ),
        "uncertainty_and_risks": "Potential numerical stiffness may require careful optimization, but the proposal remains mathematically simple.",
    }


class ProposalVersioningTests(unittest.TestCase):
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
                "session_id": "proposal-versioning",
                "output_dir": self.tmpdir.name,
                "input_path": "/tmp/input.h5ad",
                "algorithm_proposal_review_mode": "auto_approve",
            }
        )
        self.tools = SingleAgentTools(DummyLLM(), self.state, agent_role="planner", agent_id="planner")

    def tearDown(self) -> None:
        planner_tools_module.get_cellcompass_root = self._original_cellcompass_root
        planner_file_tools_module.get_cellcompass_root = self._original_file_tools_root
        workspace_policy_module.get_cellcompass_root = self._original_policy_root
        self.tmpdir.cleanup()

    def _patch_proposal_text(self, algorithm_id: str, old: str, new: str, reason: str) -> str:
        proposal_path = Path(self.tmpdir.name) / "training_algorithms" / algorithm_id / "PROPOSAL.md"
        patch = f"""*** Begin Patch
*** Update File: {proposal_path}
@@
-{old}
+{new}
*** End Patch
"""
        return self.tools.apply_workspace_patch(patch, reason=reason)

    def test_create_proposal_accepts_literature_alias_and_references(self) -> None:
        self.assertIn("literature_and_package_grounding", _CreateAlgorithmProposalInput.model_fields)
        self.assertIn("literature", _CreateAlgorithmProposalInput.model_fields)
        self.assertIn("references", _CreateAlgorithmProposalInput.model_fields)
        self.assertIn("mathematical_derivation_to_algorithm_design", _CreateAlgorithmProposalInput.model_fields)
        self.assertIn("novelty_and_contributions", _CreateAlgorithmProposalInput.model_fields)
        self.assertNotIn("mathematical_derivation_to_trainable_loss", _CreateAlgorithmProposalInput.model_fields)
        payload = _valid_proposal_payload()
        literature = payload.pop("literature_and_package_grounding")
        references = (
            "- TrajectoryNet: dynamic OT neural formulation.\n"
            "- Conditional Flow Matching Simulation-Free Dynamic Optimal Transport: OT-CFM objective.\n"
            "- Simulation-free Schrödinger bridges via score and flow matching: stochastic bridge baseline.\n"
        )
        result = self.tools.create_algorithm_proposal(
            algorithm_id="algo_lit_alias",
            literature=literature,
            references=references,
            **payload,
        )
        self.assertIn("auto-approved", result)

        proposal_record = dict((self.state.get("algorithm_proposals") or {}).get("algo_lit_alias") or {})
        proposal = dict(proposal_record.get("proposal") or {})
        self.assertEqual(proposal.get("literature_and_package_grounding"), literature)
        self.assertEqual(proposal.get("references"), references.strip())

        proposal_dir = Path(self.tmpdir.name) / "training_algorithms" / "algo_lit_alias"
        latest_markdown = (proposal_dir / "PROPOSAL.md").read_text(encoding="utf-8")
        self.assertIn("## Literature and Package Grounding", latest_markdown)
        self.assertIn("## References", latest_markdown)
        self.assertIn("TrajectoryNet: dynamic OT neural formulation", latest_markdown)

    def test_create_proposal_warns_but_accepts_informal_pseudocode(self) -> None:
        payload = _valid_proposal_payload()
        payload["implementation_pseudocode"] = (
            "Build adjacent interval supervision from the coupling, fit the proposed velocity-growth losses, "
            "roll out weighted particles from valid source cells, and evaluate W1, TMV, and the proposal-specific "
            "claim diagnostics against the same generated trajectory outputs."
        )
        result = self.tools.create_algorithm_proposal(algorithm_id="algo_informal_pseudocode", **payload)
        self.assertIn("auto-approved", result)
        self.assertIn("Warnings:", result)
        self.assertIn("implementation_pseudocode_format", result)

        proposal_record = dict((self.state.get("algorithm_proposals") or {}).get("algo_informal_pseudocode") or {})
        warnings = proposal_record.get("proposal_warnings") or []
        self.assertTrue(warnings)
        self.assertIn("implementation_pseudocode_format", warnings[0])
        proposal_dir = Path(self.tmpdir.name) / "training_algorithms" / "algo_informal_pseudocode"
        latest_markdown = (proposal_dir / "PROPOSAL.md").read_text(encoding="utf-8")
        self.assertIn("## Proposal Tool Warnings", latest_markdown)

    def test_proposal_template_tool_exposes_section_guidance(self) -> None:
        template = self.tools.get_algorithm_proposal_template()
        self.assertIn("## Problem Mathematical Form", template)
        self.assertIn("## Mathematical Derivation to Algorithm Design", template)
        self.assertIn("Do not merely rewrite the proposed algorithm", template)
        self.assertIn("## Novelty and Contributions", template)
        self.assertIn("mass_modeling_scope", template)
        self.assertIn("meaningful growth/dynamics mechanism", template)
        self.assertIn("time-only count-ratio clocks", template)

    def test_latest_proposal_only_keeps_latest_implementation_risks(self) -> None:
        payload = _valid_proposal_payload()
        result = self.tools.create_algorithm_proposal(algorithm_id="algo_x", **payload)
        self.assertIn("auto-approved", result)

        review_result = self.tools.review_algorithm_proposal(
            algorithm_id="algo_x",
            decision="approve",
            reviewer_feedback="The recoverability argument is theoretically sufficient.",
            implementation_risks=[
                "Importance weights may become numerically sharp during rollout.",
                "Growth-loss scaling may require careful tuning in practice.",
            ],
            risk_assessment=(
                "## Importance-weight sharpness\n\n"
                "- Risk: importance weights may become numerically sharp during rollout.\n"
                "- Possible CytoBridge manifestation: TMV may spike, W1 may improve while mass fit degrades, or campaign trials may reject after long rollouts.\n"
                "- Diagnostics: inspect per-time weight histograms, effective sample size, raw predicted total mass, TMV by time, and ablate growth-loss scaling.\n"
            ),
        )
        self.assertIn("approved", review_result)

        proposal_dir = Path(self.tmpdir.name) / "training_algorithms" / "algo_x"
        latest_markdown = (proposal_dir / "PROPOSAL.md").read_text(encoding="utf-8")
        self.assertIn("## Evaluator Implementation Risks", latest_markdown)
        self.assertIn("## Evaluator Risk Assessment", latest_markdown)
        self.assertIn("Importance weights may become numerically sharp during rollout.", latest_markdown)
        risk_markdown = (proposal_dir / "risk.md").read_text(encoding="utf-8")
        self.assertIn("# Proposal Risk Assessment: algo_x", risk_markdown)
        self.assertIn("Possible CytoBridge manifestation", risk_markdown)
        self.assertIn("per-time weight histograms", risk_markdown)

        previous_proposal_id = str(self.state.get("active_proposal_id") or "")
        previous_record = dict((self.state.get("algorithm_proposals") or {}).get("algo_x") or {})
        self.assertEqual(previous_record.get("implementation_risks"), [
            "Importance weights may become numerically sharp during rollout.",
            "Growth-loss scaling may require careful tuning in practice.",
        ])
        self.assertIn("Possible CytoBridge manifestation", previous_record.get("risk_assessment") or "")
        self.assertEqual(len(previous_record.get("review_history") or []), 1)

        revise_result = self._patch_proposal_text(
            "algo_x",
            payload["theoretical_core"],
            payload["theoretical_core"] + " The endpoint notation is now made explicit.",
            "Tighten the endpoint notation without changing the main idea.",
        )
        self.assertIn("auto-approved", revise_result)

        current_record = dict((self.state.get("algorithm_proposals") or {}).get("algo_x") or {})
        self.assertNotEqual(current_record.get("proposal_id"), previous_proposal_id)
        self.assertEqual(current_record.get("implementation_risks"), [])
        self.assertEqual(current_record.get("risk_assessment"), "")
        self.assertEqual(current_record.get("review_history"), [])

        latest_markdown_after_revision = (proposal_dir / "PROPOSAL.md").read_text(encoding="utf-8")
        self.assertIn("## Evaluator Implementation Risks", latest_markdown_after_revision)
        self.assertNotIn("Importance weights may become numerically sharp during rollout.", latest_markdown_after_revision)
        self.assertIn("- (none)", latest_markdown_after_revision)
        risk_markdown_after_revision = (proposal_dir / "risk.md").read_text(encoding="utf-8")
        self.assertNotIn("per-time weight histograms", risk_markdown_after_revision)
        self.assertIn("(none yet)", risk_markdown_after_revision)

        registry_record_path = proposal_dir / "registry" / "proposals" / f"{previous_proposal_id}.json"
        old_record = json.loads(registry_record_path.read_text(encoding="utf-8"))
        self.assertEqual(old_record.get("implementation_risks"), [
            "Importance weights may become numerically sharp during rollout.",
            "Growth-loss scaling may require careful tuning in practice.",
        ])
        self.assertIn("Possible CytoBridge manifestation", old_record.get("risk_assessment") or "")
        self.assertEqual(len(old_record.get("review_history") or []), 1)
        old_risk_path = proposal_dir / "registry" / "proposals" / f"{previous_proposal_id}.risk.md"
        self.assertIn("per-time weight histograms", old_risk_path.read_text(encoding="utf-8"))

    def test_proposal_patch_revision_updates_markdown_and_structured_attributes(self) -> None:
        payload = _valid_proposal_payload()
        result = self.tools.create_algorithm_proposal(algorithm_id="algo_patch", **payload)
        self.assertIn("auto-approved", result)
        proposal_dir = Path(self.tmpdir.name) / "training_algorithms" / "algo_patch"
        proposal_path = proposal_dir / "PROPOSAL.md"
        original_proposal_id = str(self.state.get("active_proposal_id") or "")
        status = json.loads(self.tools.get_algorithm_proposal_status("algo_patch"))
        self.assertEqual(status.get("editable_proposal_path"), str(proposal_path))
        self.assertIn("editable_proposal_path", status.get("proposal_patch_instruction") or "")
        patch = f"""*** Begin Patch
*** Update File: {proposal_path}
@@
-## Mass Modeling Scope
-> Required machine-readable value: `balanced_only` or `models_unbalanced_mass`. This proposal-stage attribute controls whether TMV is a hard campaign gate.
-models_unbalanced_mass
+## Mass Modeling Scope
+> Required machine-readable value: `balanced_only` or `models_unbalanced_mass`. This proposal-stage attribute controls whether TMV is a hard campaign gate.
+balanced_only
@@
-Enable unbalanced mass modeling because the method explicitly aims to match observed total-mass change.
+Do not model unbalanced mass in this revised balanced-only proposal; TMV is diagnostic only and should not be used as a hard gate.
@@
-Because the growth target is defined on the same coupling, the exact-fit limit also recovers the claimed total-mass change.
+This revised balanced-only proposal deliberately does not claim to recover total-mass change.
*** End Patch
"""
        revise_result = self.tools.apply_workspace_patch(
            patch,
            reason="Use patch editing to switch to balanced-only scope.",
        )
        self.assertIn("auto-approved", revise_result)
        current_record = dict((self.state.get("algorithm_proposals") or {}).get("algo_patch") or {})
        self.assertNotEqual(current_record.get("proposal_id"), original_proposal_id)
        self.assertEqual(current_record["proposal"].get("mass_modeling_scope"), "balanced_only")
        self.assertFalse(current_record["algorithm_attributes"].get("tmv_gate_required"))
        self.assertEqual(current_record.get("revision_source"), "apply_workspace_patch")
        self.assertEqual(current_record.get("previous_proposal_id"), original_proposal_id)
        revision_record = dict(current_record.get("proposal_revision") or {})
        self.assertEqual(revision_record.get("previous_proposal_id"), original_proposal_id)
        self.assertIn("balanced_only", revision_record.get("proposal_diff") or "")
        self.assertIn("PROPOSAL.md", revision_record.get("proposal_patch") or "")
        updated_markdown = proposal_path.read_text(encoding="utf-8")
        self.assertIn("## Abstract", updated_markdown)
        self.assertIn("## Problem Statement", updated_markdown)
        self.assertIn("## Claimed Capability", updated_markdown)
        self.assertIn("## Expected Evaluation Outcome", updated_markdown)
        self.assertIn("balanced_only", updated_markdown)

    def test_revise_algorithm_proposal_supports_full_markdown_and_patch(self) -> None:
        payload = _valid_proposal_payload()
        result = self.tools.create_algorithm_proposal(algorithm_id="algo_revise", **payload)
        self.assertIn("auto-approved", result)
        proposal_dir = Path(self.tmpdir.name) / "training_algorithms" / "algo_revise"
        proposal_path = proposal_dir / "PROPOSAL.md"
        original_proposal_id = str(self.state.get("active_proposal_id") or "")

        old_core = payload["theoretical_core"]
        new_core = old_core + " Full-markdown revision clarifies the endpoint geometry."
        full_markdown = proposal_path.read_text(encoding="utf-8").replace(old_core, new_core)
        full_result = self.tools.revise_algorithm_proposal(
            algorithm_id="algo_revise",
            revision_note="Use full markdown replacement for a large proposal rewrite.",
            proposal_markdown=full_markdown,
        )
        self.assertIn("auto-approved", full_result)
        full_record = dict((self.state.get("algorithm_proposals") or {}).get("algo_revise") or {})
        self.assertNotEqual(full_record.get("proposal_id"), original_proposal_id)
        self.assertEqual(full_record.get("revision_source"), "revise_algorithm_proposal")
        full_revision = dict(full_record.get("proposal_revision") or {})
        self.assertEqual(full_revision.get("revision_mode"), "full_markdown")
        self.assertTrue(full_revision.get("proposal_markdown_replacement"))
        self.assertIn("Full-markdown revision", full_revision.get("proposal_diff") or "")
        self.assertIn("Full-markdown revision", proposal_path.read_text(encoding="utf-8"))

        after_full_proposal_id = str(self.state.get("active_proposal_id") or "")
        old_objective = payload["objective"]
        new_objective = old_objective + " Patch revision clarifies the target benchmark evidence."
        patch = f"""*** Begin Patch
*** Update File: {proposal_path}
@@
-{old_objective}
+{new_objective}
*** End Patch
"""
        patch_result = self.tools.revise_algorithm_proposal(
            algorithm_id="algo_revise",
            revision_note="Use proposal_patch for a focused section edit.",
            proposal_patch=patch,
        )
        self.assertIn("auto-approved", patch_result)
        patch_record = dict((self.state.get("algorithm_proposals") or {}).get("algo_revise") or {})
        self.assertNotEqual(patch_record.get("proposal_id"), after_full_proposal_id)
        self.assertEqual(patch_record.get("revision_source"), "revise_algorithm_proposal")
        patch_revision = dict(patch_record.get("proposal_revision") or {})
        self.assertEqual(patch_revision.get("revision_mode"), "patch")
        self.assertIn("proposal_patch", patch_revision)
        self.assertIn("Patch revision", patch_revision.get("proposal_diff") or "")
        self.assertIn("Patch revision", proposal_path.read_text(encoding="utf-8"))

    def test_agent_review_events_keep_editable_and_review_proposal_paths_separate(self) -> None:
        self.state["algorithm_proposal_review_mode"] = "agent_decide"
        payload = _valid_proposal_payload()
        result = self.tools.create_algorithm_proposal(algorithm_id="algo_review_path", **payload)
        self.assertIn("queued for runtime evaluator review", result)
        proposal_dir = Path(self.tmpdir.name) / "training_algorithms" / "algo_review_path"
        root_proposal_path = str(proposal_dir / "PROPOSAL.md")
        runtime_action = dict(self.state.get("runtime_action") or {})
        self.assertEqual(runtime_action.get("proposal_path"), root_proposal_path)
        self.assertEqual(runtime_action.get("editable_proposal_path"), root_proposal_path)
        self.assertTrue(str(runtime_action.get("review_proposal_path") or "").endswith(".md"))
        self.assertIn("/registry/proposals/", str(runtime_action.get("review_proposal_path") or ""))
        self.assertNotEqual(runtime_action.get("review_proposal_path"), root_proposal_path)

    def test_review_rejects_stale_proposal_id(self) -> None:
        payload = _valid_proposal_payload()
        result = self.tools.create_algorithm_proposal(algorithm_id="algo_stale", **payload)
        self.assertIn("auto-approved", result)
        previous_proposal_id = str(self.state.get("active_proposal_id") or "")

        revise_result = self._patch_proposal_text(
            "algo_stale",
            payload["theoretical_core"],
            payload["theoretical_core"] + " This revision adds a clearer exact-fit statement.",
            "Create a second proposal version.",
        )
        self.assertIn("auto-approved", revise_result)
        current_record = dict((self.state.get("algorithm_proposals") or {}).get("algo_stale") or {})
        self.assertNotEqual(current_record.get("proposal_id"), previous_proposal_id)

        stale_review = self.tools.review_algorithm_proposal(
            algorithm_id="algo_stale",
            proposal_id=previous_proposal_id,
            decision="approve",
            reviewer_feedback="This stale approval must not apply to the latest proposal.",
        )
        self.assertIn("proposal_id mismatch", stale_review)

    def test_active_context_locks_writes_and_dirty_snapshot_flow(self) -> None:
        payload = _valid_proposal_payload()
        self.assertIn("auto-approved", self.tools.create_algorithm_proposal(algorithm_id="algo_a", **payload))
        self.assertIn(
            "Initialized training algorithm workspace",
            self.tools.init_training_algorithm_workspace(
                algorithm_id="algo_a",
                description="Algorithm A",
            ),
        )
        self.assertIn("auto-approved", self.tools.create_algorithm_proposal(algorithm_id="algo_b", **payload))
        self.assertIn(
            "Initialized training algorithm workspace",
            self.tools.init_training_algorithm_workspace(
                algorithm_id="algo_b",
                description="Algorithm B",
            ),
        )

        self.tools.planner_file_tools.activate_algorithm_workspace("algo_a")
        history_before = dict(self.state.get("active_algorithm_context") or {})
        self.tools.list_experiment_history("algo_b")
        self.assertEqual((self.state.get("active_algorithm_context") or {}).get("algorithm_id"), "algo_a")
        self.assertEqual(history_before.get("algorithm_id"), "algo_a")

        algo_b_readme = Path(self.tmpdir.name) / "training_algorithms" / "algo_b" / "README.md"
        with self.assertRaises(ValueError):
            self.tools.planner_file_tools.replace_workspace_file(
                str(algo_b_readme),
                "# illegal while algo_a is active\n",
                reason="cross algorithm write should fail",
            )

        self.tools.planner_file_tools.activate_algorithm_workspace("algo_b")
        write_result = self.tools.planner_file_tools.replace_workspace_file(
            str(algo_b_readme),
            "# algo_b\n\nupdated\n",
            reason="mark dirty",
        )
        self.assertIn("Replaced file", write_result)
        context = self.tools.planner_file_tools.get_active_algorithm_context()
        self.assertEqual(context.get("algorithm_id"), "algo_b")
        self.assertTrue(context.get("dirty_since_snapshot"))

        verification = self.tools.planner_file_tools.verify_algorithm_context("algo_b", stage="training")
        self.assertFalse(verification.get("ok"))
        self.assertIn("unsnapshotted edits", "; ".join(verification.get("blockers") or []))

        snapshot_payload = json.loads(self.tools.snapshot_active_algorithm_workspace("test clean snapshot"))
        self.assertTrue(snapshot_payload.get("snapshot_id"))
        self.assertEqual(snapshot_payload.get("active_algorithm_id"), "algo_b")
        self.assertFalse(snapshot_payload.get("dirty_since_snapshot"))
        verification_after = self.tools.planner_file_tools.verify_algorithm_context("algo_b", stage="training")
        self.assertTrue(verification_after.get("ok"), verification_after.get("blockers"))

    def test_list_experiment_history_returns_concise_bounded_summary(self) -> None:
        payload = _valid_proposal_payload()
        self.assertIn("auto-approved", self.tools.create_algorithm_proposal(algorithm_id="algo_history", **payload))
        registry = self.tools.planner_file_tools._bootstrap_algorithm_registry("algo_history")
        long_error = "dense output " + ("x" * 12000)
        registry.setdefault("runs", {})
        registry.setdefault("workspace_snapshots", {})
        registry.setdefault("recent_decisions", [])
        for idx in range(15):
            run_id = f"run_{idx:02d}"
            registry["runs"][run_id] = {
                "run_id": run_id,
                "algorithm_id": "algo_history",
                "created_at": f"2026-05-08T00:{idx:02d}:00Z",
                "status": "completed",
                "metrics_summary": {"w1_mean": idx, "raw_blob": long_error},
                "error": long_error,
            }
            snapshot_id = f"snapshot_{idx:02d}"
            registry["workspace_snapshots"][snapshot_id] = {
                "snapshot_id": snapshot_id,
                "algorithm_id": "algo_history",
                "created_at": f"2026-05-08T01:{idx:02d}:00Z",
                "path": f"/tmp/{snapshot_id}",
                "reason": long_error,
            }
            registry["recent_decisions"].append(
                {
                    "decision_id": f"decision_{idx:02d}",
                    "created_at": f"2026-05-08T02:{idx:02d}:00Z",
                    "decision": "reject",
                    "rationale": long_error,
                }
            )
        self.tools.planner_file_tools._save_algorithm_registry("algo_history", registry)

        history = json.loads(
            self.tools.list_experiment_history(
                "algo_history",
                limit=3,
                max_field_chars=300,
            )
        )
        self.assertEqual(history["view"], "experiment_history_summary")
        self.assertEqual(history["counts"]["runs"], 15)
        self.assertEqual(history["omitted"]["runs"], 12)
        self.assertEqual(len(history["recent_runs"]), 3)
        self.assertEqual(len(history["recent_workspace_snapshots"]), 3)
        self.assertEqual(len(history["recent_decisions"]), 3)
        rendered = json.dumps(history, ensure_ascii=False)
        self.assertIn("...[truncated", rendered)
        self.assertNotIn("x" * 4000, rendered)
        self.assertNotIn("runs", history)
        self.assertIn("read_file", history["full_history_access"])

    def test_disk_approved_proposal_overrides_stale_pending_state_after_resume(self) -> None:
        payload = _valid_proposal_payload()
        self.assertIn("auto-approved", self.tools.create_algorithm_proposal(algorithm_id="algo_resume", **payload))
        self.assertIn(
            "Initialized training algorithm workspace",
            self.tools.init_training_algorithm_workspace(
                algorithm_id="algo_resume",
                description="Resume proposal cache regression",
            ),
        )
        self.tools.planner_file_tools.activate_algorithm_workspace("algo_resume")

        proposals = self.state.get("algorithm_proposals") or {}
        stale = dict(proposals.get("algo_resume") or {})
        stale["status"] = "pending_agent_review"
        proposals["algo_resume"] = stale
        self.state["algorithm_proposals"] = proposals

        target = Path(self.tmpdir.name) / "training_algorithms" / "algo_resume" / "README.md"
        result = self.tools.planner_file_tools.replace_workspace_file(
            str(target),
            "# algo_resume\n\nedit after resumed stale checkpoint\n",
            reason="stale proposal cache should not block approved disk state",
        )
        self.assertIn("Replaced file", result)
        refreshed = dict((self.state.get("algorithm_proposals") or {}).get("algo_resume") or {})
        self.assertIn(refreshed.get("status"), {"approved", "approved_auto"})

    def test_registry_approved_proposal_heals_stale_root_proposal_file(self) -> None:
        payload = _valid_proposal_payload()
        self.assertIn("auto-approved", self.tools.create_algorithm_proposal(algorithm_id="algo_registry_resume", **payload))
        proposal_dir = Path(self.tmpdir.name) / "training_algorithms" / "algo_registry_resume"
        root_json_path = proposal_dir / "PROPOSAL.json"
        root_record = json.loads(root_json_path.read_text(encoding="utf-8"))
        registry_json_path = proposal_dir / "registry" / "proposals" / f"{root_record['proposal_id']}.json"
        registry_record = json.loads(registry_json_path.read_text(encoding="utf-8"))
        self.assertIn(registry_record.get("status"), {"approved", "approved_auto"})

        root_record["status"] = "pending_agent_review"
        root_record["review_decision"] = ""
        root_record["reviewed_at"] = ""
        root_json_path.write_text(json.dumps(root_record, ensure_ascii=False, indent=2), encoding="utf-8")
        proposals = self.state.get("algorithm_proposals") or {}
        proposals["algo_registry_resume"] = dict(root_record)
        self.state["algorithm_proposals"] = proposals

        result = self.tools.init_training_algorithm_workspace(
            algorithm_id="algo_registry_resume",
            description="Registry approved proposal should repair stale root status",
        )
        self.assertIn("Initialized training algorithm workspace", result)
        healed = json.loads(root_json_path.read_text(encoding="utf-8"))
        self.assertIn(healed.get("status"), {"approved", "approved_auto"})
        refreshed = dict((self.state.get("algorithm_proposals") or {}).get("algo_registry_resume") or {})
        self.assertIn(refreshed.get("status"), {"approved", "approved_auto"})

    def test_approved_review_of_nonactive_algorithm_activates_target_context(self) -> None:
        payload = _valid_proposal_payload()
        self.assertIn("auto-approved", self.tools.create_algorithm_proposal(algorithm_id="algo_active", **payload))
        self.assertIn("auto-approved", self.tools.create_algorithm_proposal(algorithm_id="algo_target", **payload))
        target_record = dict((self.state.get("algorithm_proposals") or {}).get("algo_target") or {})

        self.tools.planner_file_tools.activate_algorithm_workspace("algo_active")
        review_result = self.tools.review_algorithm_proposal(
            algorithm_id="algo_target",
            proposal_id=str(target_record.get("proposal_id") or ""),
            decision="approve",
            reviewer_feedback="Approved target should become the active authoring context.",
        )
        self.assertIn("approved", review_result)
        self.assertEqual((self.state.get("active_algorithm_context") or {}).get("algorithm_id"), "algo_target")

    def test_current_workflow_context_reports_active_bindings_plan_and_paths(self) -> None:
        payload = _valid_proposal_payload()
        self.assertIn("auto-approved", self.tools.create_algorithm_proposal(algorithm_id="algo_ctx", **payload))
        self.tools.update_plan(
            [
                {"step": "Inspect active algorithm context", "status": "completed"},
                {"step": "Continue implementation", "status": "in_progress"},
            ],
            explanation="Context smoke test",
        )
        self.state["preprocessed_path"] = "/tmp/preprocessed.h5ad"
        self.state["planner_need"] = {"reason": "awaiting_user_review", "algorithm_id": "algo_ctx"}

        with patch.dict("os.environ", {"ACCESS_KEY": "", "BOHRIUM_ACCESS_KEY": ""}, clear=False):
            payload_text = self.tools.get_current_workflow_context()
        context = json.loads(payload_text)

        self.assertTrue(context.get("read_only"))
        self.assertEqual(context["active_algorithm"]["algorithm_id"], "algo_ctx")
        self.assertEqual(context["active_algorithm"]["proposal_id"], self.state.get("active_proposal_id"))
        self.assertEqual(context["active_algorithm"]["proposal_status"], "approved_auto")
        self.assertEqual(context["active_algorithm"]["proposal_claimed_capability"], payload["claimed_capability"])
        self.assertEqual(context["active_algorithm"]["expected_evaluation_outcome"], payload["expected_evaluation_outcome"])
        self.assertEqual(
            context["active_algorithm"]["algorithm_attributes"].get("mass_modeling_scope"),
            "models_unbalanced_mass",
        )
        self.assertEqual(context["paths"]["input_path"], "/tmp/input.h5ad")
        self.assertEqual(context["paths"]["preprocessed_path"], "/tmp/preprocessed.h5ad")
        self.assertEqual(context["pending"]["planner_need"]["reason"], "awaiting_user_review")
        self.assertEqual(context["plan"]["items"][1]["status"], "in_progress")
        self.assertIn("Continue implementation", context["plan"]["rendered"])
        bohrium_status = context["external_literature_tools"]["bohrium_paper_search"]
        self.assertFalse(bohrium_status["available"])
        self.assertFalse(bohrium_status["safe_to_call"])
        self.assertIn("ACCESS_KEY", bohrium_status["required_env_vars"])
        self.assertIn("BOHRIUM_ACCESS_KEY", bohrium_status["required_env_vars"])

        skill_payload = json.loads(self.tools.list_skills())
        self.assertEqual(set(skill_payload), {"skills"})
        self.assertTrue(any(item.get("name") == "algorithm-orchestrator" for item in skill_payload["skills"]))
        deprecated_algorithm_skills = {
            "algorithm-development",
            "algorithm-proposal-theory",
            "algorithm-authoring",
            "algorithm-review-and-training",
            "algorithm-tuning-playbook",
        }
        self.assertFalse(any(item.get("name") in deprecated_algorithm_skills for item in skill_payload["skills"]))
        allowed_skill_keys = {"name", "description", "skill_md_path"}
        for item in skill_payload["skills"]:
            self.assertEqual(set(item), allowed_skill_keys)
            self.assertNotIn("loaded", item)
            self.assertNotIn("source", item)
            self.assertNotIn("enabled", item)
            self.assertNotIn("hidden", item)

        tool_names = {str(tool.name) for tool in self.tools.get_tools()}
        self.assertIn("list_skills", tool_names)
        self.assertIn("get_current_workflow_context", tool_names)
        self.assertIn("snapshot_active_algorithm_workspace", tool_names)

    def test_current_workflow_context_reports_bohrium_key_availability_without_secret(self) -> None:
        secret = "bohrium-secret-for-test"
        with patch.dict("os.environ", {"BOHRIUM_ACCESS_KEY": secret, "ACCESS_KEY": ""}, clear=False):
            context = json.loads(self.tools.get_current_workflow_context())

        bohrium_status = context["external_literature_tools"]["bohrium_paper_search"]
        self.assertTrue(bohrium_status["available"])
        self.assertTrue(bohrium_status["safe_to_call"])
        self.assertEqual(bohrium_status["configured_env_var"], "BOHRIUM_ACCESS_KEY")
        self.assertNotIn(secret, json.dumps(context))


if __name__ == "__main__":
    unittest.main()
