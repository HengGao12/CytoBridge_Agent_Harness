from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cytobridge_agent.runtime_v2.state import ensure_runtime_v2_state
from cytobridge_agent.runtime_v2.tool_registry import SingleAgentTools
from cytobridge_agent.tools import planner_file_tools as planner_file_tools_module
from cytobridge_agent.tools import planner_tools as planner_tools_module
from cytobridge_agent.tools import research_idea_registry as research_idea_registry_module


class DummyLLM:
    pass


def _valid_proposal_payload() -> dict[str, str]:
    return {
        "abstract": (
            "This proposal learns coupling-aligned interval dynamics so exact-fit training recovers weighted "
            "temporal marginals and, when enabled, observed total-mass changes."
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
            "The proposal targets temporal snapshot reconstruction where one model must explain both distributional "
            "transport and optional growth or total-mass changes."
        ),
        "problem_mathematical_form": (
            "For adjacent snapshots, fit a path-induced terminal weighted measure to the next observed empirical "
            "measure, optionally augmented with log-mass dynamics."
        ),
        "claimed_capability": (
            "The method claims to recover adjacent-time weighted distributions and optional mass changes; W1 and "
            "TMV are the validation evidence."
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
            "The method should work best on adjacent-time temporal panels with informative coupling geometry and optional "
            "smooth total-mass changes. Expected evidence is competitive W1, TMV below threshold only when unbalanced mass "
            "is modeled, and a proposal-specific recovery diagnostic that improves over the strongest relevant baseline."
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


class ResearchIdeaVersioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self._original_cellcompass_root = planner_tools_module.get_cellcompass_root
        self._original_file_tools_root = planner_file_tools_module.get_cellcompass_root
        self._original_idea_registry_root = research_idea_registry_module.get_cellcompass_root
        planner_tools_module.get_cellcompass_root = lambda: Path(self.tmpdir.name)
        planner_file_tools_module.get_cellcompass_root = lambda: Path(self.tmpdir.name)
        research_idea_registry_module.get_cellcompass_root = lambda: Path(self.tmpdir.name)
        self.state = ensure_runtime_v2_state(
            {
                "session_id": "idea-versioning",
                "output_dir": self.tmpdir.name,
                "input_path": "/tmp/input.h5ad",
                "idea_review_mode": "auto_approve",
                "algorithm_proposal_review_mode": "auto_approve",
            }
        )
        self.tools = SingleAgentTools(DummyLLM(), self.state, agent_role="planner", agent_id="planner")

    def tearDown(self) -> None:
        planner_tools_module.get_cellcompass_root = self._original_cellcompass_root
        planner_file_tools_module.get_cellcompass_root = self._original_file_tools_root
        research_idea_registry_module.get_cellcompass_root = self._original_idea_registry_root
        self.tmpdir.cleanup()

    def test_research_idea_revision_and_algorithm_linkage(self) -> None:
        idea_id = "test_research_idea_versioning_fixture"
        result = self.tools.create_research_idea(
            title="TEST research idea versioning fixture",
            primary_track="biology-journal",
            problem_definition="Determine whether observed branch-probability shifts arise from genuine fate rewiring or from differential growth/death selection across time.",
            scientific_object="Condition-specific fate distribution under explicit growth/death decomposition.",
            current_method_failure_mode="Current methods often explain snapshot redistribution as transport alone and confound true fate changes with abundance selection effects.",
            prior_work=(
                "RNA velocity gives local directional signal, WOT and related OT methods couple adjacent marginals, "
                "and PRESCIENT-like models learn stochastic dynamics, but prior work still does not cleanly separate "
                "branch occupancy change caused by incoming transport from occupancy change caused by growth/death selection."
            ),
            why_this_matters="Without this distinction, developmental, perturbation, and treatment conclusions can assign mechanism to the wrong biological driver.",
            why_now="Time-resolved single-cell assays and perturbation datasets now make it possible to compare trajectory rewiring against growth-mediated redistribution directly.",
            falsifiable_success_criteria="A successful method should separate transport and growth contributions well enough that trajectory rewiring claims change when growth effects dominate.",
            non_goals="Do not claim full causal identification from snapshots alone or universal perturbation generalization without new assumptions.",
            evidence_basis="- survey: papers/agent_algo_survey_20260417/synthesis.md\n- note: snapshot methods confound growth and transport",
            feasible_direction_families=(
                "- Joint transport-growth decomposition with explicit abundance accounting and uncertainty on branch assignment.\n"
                "- Multi-condition shared-backbone dynamics with a separate context-specific rewiring component and growth field.\n"
                "- Counterfactual interval diagnostics that reject rewiring claims when abundance-only explanations suffice."
            ),
            feasibility_constraints="Requires multiple timepoints with reliable latent states; cannot claim full identifiability without extra assumptions or side information.",
            idea_id=idea_id,
        )
        self.assertIn("auto-approved", result)
        idea_record = json.loads(self.tools.get_research_idea_status(idea_id))
        self.assertEqual(idea_record["review_status"], "approved")
        original_revision_id = idea_record["current_revision_id"]

        revise_result = self.tools.revise_research_idea(
            idea_id=idea_id,
            revision_note="Tighten the scientific object to emphasize condition-specific decomposition.",
            scientific_object="Condition-specific transport/growth decomposition of fate redistribution across adjacent intervals.",
        )
        self.assertIn("auto-approved", revise_result)
        revised_record = json.loads(self.tools.get_research_idea_status(idea_id))
        self.assertNotEqual(revised_record["current_revision_id"], original_revision_id)
        self.assertEqual(revised_record["scientific_object"], "Condition-specific transport/growth decomposition of fate redistribution across adjacent intervals.")
        self.assertIn("RNA velocity gives local directional signal", revised_record["prior_work"])

        payload = _valid_proposal_payload()
        proposal_result = self.tools.create_algorithm_proposal(
            algorithm_id="algo_idea_x",
            primary_idea_id=idea_id,
            **payload,
        )
        self.assertIn("auto-approved", proposal_result)
        proposal_status = json.loads(self.tools.get_algorithm_proposal_status("algo_idea_x"))
        self.assertEqual(proposal_status["primary_idea_id"], idea_id)

        linked_record = json.loads(self.tools.get_research_idea_status(idea_id))
        self.assertIn("algo_idea_x", linked_record["linked_algorithms"])
        self.assertGreaterEqual(int(linked_record["attempt_count"]), 1)


if __name__ == "__main__":
    unittest.main()
