from __future__ import annotations

import json
import tempfile
import unittest

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from cytobridge_agent.runtime_v2.agent import CytoBridgeAgent
from cytobridge_agent.runtime_v2.graph import build_runtime_graph
from cytobridge_agent.runtime_v2.state import ensure_runtime_v2_state
from cytobridge_agent.runtime_v2.tool_registry import SingleAgentTools
from cytobridge_agent.tools.planner_file_tools import IMPLEMENTATION_REVIEW_POLICY_VERSION


class MinimalLLM:
    def bind_tools(self, tools):  # noqa: ANN001
        del tools
        return self

    def invoke(self, messages):  # noqa: ANN001
        raise AssertionError("invoke should not be called in this unit test")


class PostToolHookTests(unittest.TestCase):
    def test_agent_run_tolerates_none_update_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "none-payload-test",
                    "output_dir": tmpdir,
                }
            )
            agent = CytoBridgeAgent(MinimalLLM(), state, agent_role="planner", agent_id="planner")

            class DummyGraph:
                @staticmethod
                def stream(graph_state, stream_mode="updates", config=None):  # noqa: ANN001
                    del graph_state, stream_mode, config
                    yield {
                        "tools": None,
                        "agent": {
                            "messages": [
                                AIMessage(
                                    content="phase=final_answer\nCompleted without crashing.",
                                    additional_kwargs={"phase": "final_answer"},
                                )
                            ]
                        },
                    }

            agent.graph = DummyGraph()
            result = agent.run(user_instruction="test none payload handling")

            self.assertEqual(result["status"], "completed")
            self.assertIn("Completed without crashing.", result["content"])

    def test_graph_interrupts_after_tool_sets_needs_input(self) -> None:
        shared_state = {
            "planner_phase": "working",
            "planner_need": {},
        }
        agent_calls = {"count": 0}

        def request_review() -> str:
            shared_state["planner_phase"] = "needs_input"
            shared_state["planner_need"] = {
                "source": "algorithm_proposal",
                "question": "Please approve the algorithm proposal.",
                "reason": "algorithm_proposal_pending_review",
            }
            return "proposal_saved"

        def agent_node(graph_state):  # noqa: ANN001
            del graph_state
            agent_calls["count"] += 1
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "request_review",
                                "args": {},
                                "id": "call_request_review",
                                "type": "tool_call",
                            }
                        ],
                    )
                ],
                "commentary_retries": 0,
            }

        def post_tool_hook_node(graph_state):  # noqa: ANN001
            del graph_state
            if shared_state["planner_phase"] != "needs_input":
                return {}
            return {
                "messages": [
                    AIMessage(
                        content="phase=needs_input\nPlease approve the algorithm proposal.",
                        additional_kwargs={"phase": "needs_input"},
                    )
                ],
                "commentary_retries": 0,
            }

        graph = build_runtime_graph(
            agent_node,
            [
                StructuredTool.from_function(
                    func=request_review,
                    name="request_review",
                    description="Mark planner state as waiting for proposal review.",
                )
            ],
            post_tool_hook_node=post_tool_hook_node,
        )

        final_ai_messages = []
        for event in graph.stream(
            {
                "messages": [HumanMessage(content="Create a proposal.")],
                "commentary_retries": 0,
                "turn_system_prompt": "",
                "enable_multimodal": False,
            },
            stream_mode="updates",
            config={"configurable": {"thread_id": "post-tool-test", "checkpoint_ns": "post-tool-test"}},
        ):
            for node_payload in event.values():
                for message in node_payload.get("messages", []):
                    if isinstance(message, AIMessage):
                        final_ai_messages.append(message)

        self.assertEqual(agent_calls["count"], 1)
        self.assertTrue(final_ai_messages)
        self.assertEqual(final_ai_messages[-1].additional_kwargs.get("phase"), "needs_input")
        self.assertIn("Please approve the algorithm proposal.", str(final_ai_messages[-1].content))

    def test_existing_planner_need_is_preserved_on_needs_input_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "hook-test",
                    "output_dir": tmpdir,
                    "planner_phase": "needs_input",
                    "planner_need": {
                        "source": "algorithm_proposal",
                        "question": "Please approve the algorithm proposal.",
                        "reason": "algorithm_proposal_pending_review",
                        "algorithm_id": "custom_algo",
                    },
                }
            )
            agent = CytoBridgeAgent(MinimalLLM(), state, agent_role="planner", agent_id="planner")

            status = agent._synchronize_planner_phase("needs_input", "Generic fallback question")

            self.assertEqual(status, "needs_input")
            self.assertEqual(state["planner_need"]["algorithm_id"], "custom_algo")
            self.assertEqual(state["planner_need"]["question"], "Please approve the algorithm proposal.")

    def test_proposal_agent_review_action_runs_evaluator_and_returns_commentary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "hook-test",
                    "output_dir": tmpdir,
                    "algorithm_proposal_review_mode": "agent_decide",
                    "runtime_action": {
                        "kind": "proposal_agent_review",
                        "algorithm_id": "custom_algo",
                        "proposal_id": "proposal_123",
                        "proposal_path": str(tmpdir),
                    },
                }
            )
            agent = CytoBridgeAgent(MinimalLLM(), state, agent_role="planner", agent_id="planner")
            calls = {}

            def fake_run_sync(**kwargs):  # noqa: ANN003
                calls["run_sync"] = kwargs
                return {
                    "status": "completed",
                    "summary": "The recovery argument is mathematically incomplete.",
                    "findings": ["The proposal does not prove mass-fit from the stated objective."],
                    "artifact_refs": [],
                    "proposed_state_updates": {
                        "proposal_review": {
                            "decision": "revise",
                            "reviewer_feedback": "The mass / growth derivation is underspecified and must be tightened.",
                            "implementation_risks": [
                                "Particle weights may collapse numerically during long-horizon rollout.",
                                "The optimization may be sensitive to imbalance-ratio scaling.",
                            ],
                            "risk_assessment": (
                                "## Particle weight collapse\n\n"
                                "- Manifestation: TMV spikes during campaign trials.\n"
                                "- Diagnostics: inspect weight histograms and per-time total mass."
                            ),
                            "confidence": 0.88,
                        }
                    },
                    "needs_input_question": "",
                }

            def fake_review_algorithm_proposal(
                *,
                algorithm_id,
                decision,
                reviewer_feedback,
                proposal_id="",
                implementation_risks=None,
                risk_assessment="",
            ):
                calls["review"] = {
                    "algorithm_id": algorithm_id,
                    "proposal_id": proposal_id,
                    "decision": decision,
                    "reviewer_feedback": reviewer_feedback,
                    "implementation_risks": list(implementation_risks or []),
                    "risk_assessment": risk_assessment,
                }
                return "✅ applied"

            assert agent.tools_handler.subagent_manager is not None
            agent.tools_handler.subagent_manager.run_sync = fake_run_sync  # type: ignore[method-assign]
            agent.tools_handler.review_algorithm_proposal = fake_review_algorithm_proposal  # type: ignore[method-assign]

            payload = agent.post_tool_hook_node({})
            message = payload["messages"][0]

            self.assertEqual(calls["review"]["algorithm_id"], "custom_algo")
            self.assertEqual(calls["review"]["proposal_id"], "proposal_123")
            self.assertEqual(calls["review"]["decision"], "revise")
            self.assertIn("mass / growth derivation is underspecified", calls["review"]["reviewer_feedback"])
            self.assertIn("Particle weights may collapse numerically during long-horizon rollout.", calls["review"]["implementation_risks"])
            self.assertIn("TMV spikes", calls["review"]["risk_assessment"])
            self.assertEqual(state["runtime_action"], {})
            self.assertIn("Decision: revise", str(message.content))
            self.assertIn("Implementation risks:", str(message.content))
            self.assertIn("Risk assessment saved to risk.md:", str(message.content))
            self.assertIn("The proposal is marked revision_requested.", str(message.content))
            self.assertEqual(calls["run_sync"]["subagent_type"], "proposal_evaluator")

    def test_proposal_agent_review_retries_when_risk_contract_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "hook-retry-test",
                    "output_dir": tmpdir,
                    "algorithm_proposal_review_mode": "agent_decide",
                    "runtime_action": {
                        "kind": "proposal_agent_review",
                        "algorithm_id": "custom_algo",
                        "proposal_id": "proposal_123",
                        "proposal_path": str(tmpdir),
                    },
                }
            )
            events = []
            agent = CytoBridgeAgent(
                MinimalLLM(),
                state,
                agent_role="planner",
                agent_id="planner",
                event_callback=lambda event_type, payload: events.append((event_type, payload)),
            )
            calls = {"run_sync": []}

            def fake_run_sync(**kwargs):  # noqa: ANN003
                calls["run_sync"].append(kwargs)
                if len(calls["run_sync"]) == 1:
                    return {
                        "status": "completed",
                        "summary": "Approved but forgot risks.",
                        "findings": [],
                        "artifact_refs": [],
                        "proposed_state_updates": {
                            "proposal_review": {
                                "decision": "approve",
                                "reviewer_feedback": "The proposal is sound.",
                                "confidence": 0.9,
                            }
                        },
                        "needs_input_question": "",
                    }
                return {
                    "status": "completed",
                    "summary": "Approved with structured risks.",
                    "findings": [],
                    "artifact_refs": [],
                    "proposed_state_updates": {
                        "proposal_review": {
                            "decision": "approve",
                            "reviewer_feedback": "The proposal is sound.",
                            "implementation_risks": ["Solver memory may be high."],
                            "risk_assessment": "## Solver memory\n\nInspect memory and runtime.",
                            "confidence": 0.9,
                        }
                    },
                    "needs_input_question": "",
                }

            def fake_review_algorithm_proposal(**kwargs):  # noqa: ANN003
                calls["review"] = kwargs
                return "✅ applied"

            assert agent.tools_handler.subagent_manager is not None
            agent.tools_handler.subagent_manager.run_sync = fake_run_sync  # type: ignore[method-assign]
            agent.tools_handler.review_algorithm_proposal = fake_review_algorithm_proposal  # type: ignore[method-assign]

            payload = agent.post_tool_hook_node({})

            self.assertEqual(len(calls["run_sync"]), 2)
            self.assertIn("STRUCTURED OUTPUT CONTRACT VIOLATION", calls["run_sync"][1]["context_notes"])
            self.assertIn("Previous invalid subagent result to reuse", calls["run_sync"][1]["context_notes"])
            self.assertIn("Approved but forgot risks.", calls["run_sync"][1]["context_notes"])
            self.assertIn("Reuse the prior review analysis", calls["run_sync"][1]["success_criteria"][-1])
            self.assertEqual(calls["review"]["decision"], "approve")
            self.assertEqual(calls["review"]["implementation_risks"], ["Solver memory may be high."])
            self.assertIn("Solver memory", calls["review"]["risk_assessment"])
            self.assertTrue(any(event_type == "structured_subagent_result_retry" for event_type, _ in events))
            self.assertIn("Decision: approve", str(payload["messages"][0].content))

    def test_research_idea_agent_review_action_runs_evaluator_and_returns_commentary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "idea-hook-test",
                    "output_dir": tmpdir,
                    "idea_review_mode": "agent_decide",
                    "research_ideas": {
                        "idea_x": {
                            "idea_id": "idea_x",
                            "title": "Idea X",
                            "idea_path": str(tmpdir),
                            "idea_json_path": str(tmpdir),
                        }
                    },
                    "runtime_action": {
                        "kind": "research_idea_agent_review",
                        "idea_id": "idea_x",
                        "title": "Idea X",
                        "idea_path": str(tmpdir),
                    },
                }
            )
            agent = CytoBridgeAgent(MinimalLLM(), state, agent_role="planner", agent_id="planner")
            calls = {}

            def fake_run_sync(**kwargs):  # noqa: ANN003
                calls["run_sync"] = kwargs
                return {
                    "status": "completed",
                    "summary": "The problem is meaningful but the success criteria need tightening.",
                    "findings": ["The idea is sharp enough to keep, but the win condition should be more falsifiable."],
                    "artifact_refs": [],
                    "proposed_state_updates": {
                        "research_idea_review": {
                            "decision": "revise",
                            "reviewer_feedback": "The scientific object is clear, but the success criteria should be made more falsifiable.",
                            "confidence": 0.79,
                        }
                    },
                    "needs_input_question": "",
                }

            def fake_review_research_idea(*, idea_id, decision, reviewer_feedback):
                calls["review"] = {
                    "idea_id": idea_id,
                    "decision": decision,
                    "reviewer_feedback": reviewer_feedback,
                }
                return "✅ applied"

            assert agent.tools_handler.subagent_manager is not None
            agent.tools_handler.subagent_manager.run_sync = fake_run_sync  # type: ignore[method-assign]
            agent.tools_handler.review_research_idea = fake_review_research_idea  # type: ignore[method-assign]

            payload = agent.post_tool_hook_node({})
            message = payload["messages"][0]

            self.assertEqual(calls["review"]["idea_id"], "idea_x")
            self.assertEqual(calls["review"]["decision"], "revise")
            self.assertIn("success criteria should be made more falsifiable", calls["review"]["reviewer_feedback"])
            self.assertEqual(state["runtime_action"], {})
            self.assertIn("Decision: revise", str(message.content))
            self.assertIn("The idea is marked revise.", str(message.content))
            self.assertEqual(calls["run_sync"]["subagent_type"], "idea_evaluator")

    def test_inference_agent_review_action_records_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "inference-hook-test",
                    "output_dir": tmpdir,
                    "runtime_action": {
                        "kind": "inference_agent_review",
                        "algorithm_id": "custom_algo",
                        "proposal_id": "proposal_123",
                        "purpose": "campaign",
                        "review_hash": "hash_123",
                        "fingerprint_payload": {"has_simulation_hook": True},
                        "relevant_paths": [str(tmpdir)],
                    },
                }
            )
            agent = CytoBridgeAgent(MinimalLLM(), state, agent_role="planner", agent_id="planner")
            calls = {}

            def fake_run_sync(**kwargs):  # noqa: ANN003
                calls["run_sync"] = kwargs
                return {
                    "status": "completed",
                    "summary": "Inference is a valid t0 rollout.",
                    "findings": ["simulation_hook uses only the provided inference context."],
                    "artifact_refs": [],
                    "proposed_state_updates": {
                        "inference_review": {
                            "decision": "approve",
                            "reviewer_feedback": "No future-time truth is used to construct predictions.",
                            "blocking_issues": [],
                            "advisory_risks": ["Long rollouts may drift numerically."],
                        }
                    },
                    "needs_input_question": "",
                }

            assert agent.tools_handler.subagent_manager is not None
            agent.tools_handler.subagent_manager.run_sync = fake_run_sync  # type: ignore[method-assign]

            payload = agent.post_tool_hook_node({})
            message = payload["messages"][0]

            self.assertEqual(calls["run_sync"]["subagent_type"], "inference_evaluator")
            self.assertEqual(state["runtime_action"], {})
            record = agent.tools_handler.planner_file_tools.get_inference_review_record("custom_algo")
            self.assertEqual(record["status"], "approved")
            self.assertEqual(record["review_hash"], "hash_123")
            self.assertIn("Decision: approve", str(message.content))
            self.assertIn("Long rollouts may drift numerically.", str(message.content))

    def test_inference_agent_review_accepts_embedded_proposed_state_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "inference-hook-embedded-json-test",
                    "output_dir": tmpdir,
                    "runtime_action": {
                        "kind": "inference_agent_review",
                        "algorithm_id": "custom_algo",
                        "proposal_id": "proposal_123",
                        "purpose": "campaign",
                        "review_hash": "hash_embedded",
                        "fingerprint_payload": {"has_simulation_hook": True},
                        "relevant_paths": [str(tmpdir)],
                    },
                }
            )
            agent = CytoBridgeAgent(MinimalLLM(), state, agent_role="planner", agent_id="planner")

            def fake_run_sync(**kwargs):  # noqa: ANN003
                del kwargs
                return {
                    "status": "completed",
                    "summary": "Inference is a valid t0 rollout.",
                    "findings": [
                        (
                            'proposed_state_updates.inference_review={"decision":"approve",'
                            '"reviewer_feedback":"No future-time truth is used to construct predictions.",'
                            '"blocking_issues":[],"advisory_risks":["Compression should be sensitivity-checked."]}'
                        )
                    ],
                    "artifact_refs": [],
                    "proposed_state_updates": {},
                    "needs_input_question": "",
                }

            assert agent.tools_handler.subagent_manager is not None
            agent.tools_handler.subagent_manager.run_sync = fake_run_sync  # type: ignore[method-assign]

            payload = agent.post_tool_hook_node({})
            message = payload["messages"][0]

            self.assertEqual(state["runtime_action"], {})
            record = agent.tools_handler.planner_file_tools.get_inference_review_record("custom_algo")
            self.assertEqual(record["status"], "approved")
            self.assertEqual(record["review_hash"], "hash_embedded")
            self.assertEqual(record["advisory_risks"], ["Compression should be sensitivity-checked."])
            self.assertIn("Decision: approve", str(message.content))

    def test_inference_agent_review_accepts_embedded_proposed_state_json_with_spaced_equals(self) -> None:
        payload = SingleAgentTools._extract_review_payload(
            {
                "summary": "Inference is a valid t0 rollout.",
                "findings": [
                    (
                        'proposed_state_updates.inference_review = {"decision":"approve",'
                        '"reviewer_feedback":"No future-time truth is used to construct predictions.",'
                        '"blocking_issues":[],"advisory_risks":["Compression should be sensitivity-checked."]}'
                    )
                ],
                "proposed_state_updates": {},
            },
            "inference_review",
        )

        self.assertEqual(payload["decision"], "approve")
        self.assertEqual(payload["reviewer_feedback"], "No future-time truth is used to construct predictions.")
        self.assertEqual(payload["blocking_issues"], [])
        self.assertEqual(payload["advisory_risks"], ["Compression should be sensitivity-checked."])

    def test_implementation_agent_review_action_records_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "implementation-hook-test",
                    "output_dir": tmpdir,
                    "runtime_action": {
                        "kind": "implementation_agent_review",
                        "algorithm_id": "custom_algo",
                        "proposal_id": "proposal_123",
                        "purpose": "campaign",
                        "review_hash": "impl_hash_123",
                        "fingerprint_payload": {"proposal": {"proposal_id": "proposal_123"}},
                        "relevant_paths": [str(tmpdir)],
                        "campaign_id": "campaign_123",
                        "campaign_trial_count": 10,
                        "review_policy": "initial_or_proposal_semantics_change",
                        "due_reasons": ["proposal semantics changed since the approved implementation review"],
                    },
                }
            )
            agent = CytoBridgeAgent(MinimalLLM(), state, agent_role="planner", agent_id="planner")
            calls = {}

            def fake_run_sync(**kwargs):  # noqa: ANN003
                calls["run_sync"] = kwargs
                return {
                    "status": "completed",
                    "summary": "Implementation matches the approved proposal.",
                    "findings": ["The proposal pseudocode maps to algorithm.py and config.yaml."],
                    "artifact_refs": [],
                    "subagent_id": "subagent_impl",
                    "proposed_state_updates": {
                        "implementation_review": {
                            "decision": "approve",
                            "reviewer_feedback": "No proposal-required mechanism is omitted.",
                            "blocking_issues": [],
                            "advisory_risks": ["The UOT solve may need batching on larger datasets."],
                            "efficiency_recommendations": ["Move repeated pairwise cost computation to a batched GPU path."],
                            "generalization_shortcut_risks": ["No checkpoint warm-start shortcut was detected."],
                            "risk_assessment": "### UOT batching\n\nCompare chunked POT calls with builtin VGFM.",
                        }
                    },
                    "needs_input_question": "",
                }

            assert agent.tools_handler.subagent_manager is not None
            agent.tools_handler.subagent_manager.run_sync = fake_run_sync  # type: ignore[method-assign]

            payload = agent.post_tool_hook_node({})
            message = payload["messages"][0]

            self.assertEqual(calls["run_sync"]["subagent_type"], "implementation_evaluator")
            self.assertEqual(state["runtime_action"], {})
            record = agent.tools_handler.planner_file_tools.get_implementation_review_record("custom_algo")
            self.assertEqual(record["status"], "approved")
            self.assertEqual(record["review_hash"], "impl_hash_123")
            self.assertEqual(record["proposal_id"], "proposal_123")
            self.assertEqual(record["implementation_review_policy_version"], IMPLEMENTATION_REVIEW_POLICY_VERSION)
            self.assertEqual(record["campaign_trial_count"], 10)
            self.assertIn("batched GPU", record["efficiency_recommendations"][0])
            self.assertIn("No checkpoint warm-start", record["generalization_shortcut_risks"][0])
            self.assertIn("UOT batching", record["risk_assessment"])
            self.assertIn("Decision: approve", str(message.content))
            self.assertIn("UOT solve may need batching", str(message.content))
            self.assertIn("Efficiency recommendations", str(message.content))
            self.assertIn("Implementation risk notes saved to risk.md", str(message.content))

    def test_implementation_agent_review_records_flattened_findings_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "implementation-hook-flattened-test",
                    "output_dir": tmpdir,
                    "runtime_action": {
                        "kind": "implementation_agent_review",
                        "algorithm_id": "custom_algo",
                        "proposal_id": "proposal_123",
                        "purpose": "campaign",
                        "review_hash": "impl_hash_456",
                        "fingerprint_payload": {"proposal": {"proposal_id": "proposal_123"}},
                        "relevant_paths": [str(tmpdir)],
                        "campaign_id": "campaign_123",
                        "campaign_trial_count": 0,
                        "review_interval": 10,
                    },
                }
            )
            agent = CytoBridgeAgent(MinimalLLM(), state, agent_role="planner", agent_id="planner")

            def fake_run_sync(**kwargs):  # noqa: ANN003
                del kwargs
                return {
                    "status": "completed",
                    "summary": "Verdict: approve for campaign evidence.",
                    "findings": [
                        "implementation_review.decision=approve",
                        "implementation_review.reviewer_feedback=Approved for trusted campaign trials.",
                        "implementation_review.blocking_issues=[]",
                        "implementation_review.advisory_risks=[\"Monitor target residuals.\"]",
                        "implementation_review.efficiency_recommendations=[\"Batch dense UOT.\"]",
                        "implementation_review.generalization_shortcut_risks=[\"No warm-start shortcut found.\"]",
                    ],
                    "artifact_refs": [],
                    "subagent_id": "subagent_impl",
                    "proposed_state_updates": {},
                    "needs_input_question": "",
                }

            assert agent.tools_handler.subagent_manager is not None
            agent.tools_handler.subagent_manager.run_sync = fake_run_sync  # type: ignore[method-assign]

            payload = agent.post_tool_hook_node({})
            message = payload["messages"][0]

            record = agent.tools_handler.planner_file_tools.get_implementation_review_record("custom_algo")
            self.assertEqual(record["status"], "approved")
            self.assertEqual(record["review_hash"], "impl_hash_456")
            self.assertEqual(record["advisory_risks"], ["Monitor target residuals."])
            self.assertEqual(record["efficiency_recommendations"], ["Batch dense UOT."])
            self.assertEqual(record["generalization_shortcut_risks"], ["No warm-start shortcut found."])
            self.assertIn("Decision: approve", str(message.content))

    def test_implementation_review_approval_resumes_campaign_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "implementation-hook-resume-test",
                    "output_dir": tmpdir,
                    "runtime_action": {
                        "kind": "implementation_agent_review",
                        "algorithm_id": "custom_algo",
                        "proposal_id": "proposal_123",
                        "purpose": "campaign",
                        "review_hash": "impl_hash_123",
                        "fingerprint_payload": {"proposal": {"proposal_id": "proposal_123"}},
                        "relevant_paths": [str(tmpdir)],
                        "campaign_id": "campaign_123",
                        "campaign_trial_count": 0,
                        "review_interval": 10,
                        "resume_after_review": {
                            "tool": "run_campaign_trial",
                            "args": {
                                "campaign_id": "campaign_123",
                                "dataset_config_overrides": {"datasets": [{"dataset_id": "simulation_gene_2d"}]},
                            },
                        },
                    },
                }
            )
            agent = CytoBridgeAgent(MinimalLLM(), state, agent_role="planner", agent_id="planner")
            calls = {}

            def fake_run_sync(**kwargs):  # noqa: ANN003
                calls["run_sync"] = kwargs
                return {
                    "status": "completed",
                    "summary": "Implementation matches the approved proposal.",
                    "findings": [
                        (
                            'proposed_state_updates.implementation_review={"decision":"approve",'
                            '"reviewer_feedback":"No proposal-required mechanism is omitted.",'
                            '"blocking_issues":[],"advisory_risks":[],'
                            '"efficiency_recommendations":[],'
                            '"generalization_shortcut_risks":[]}'
                        )
                    ],
                    "artifact_refs": [],
                    "subagent_id": "subagent_impl",
                    "proposed_state_updates": {},
                    "needs_input_question": "",
                }

            def fake_run_campaign_trial(**kwargs):  # noqa: ANN003
                calls["run_campaign_trial"] = kwargs
                return json.dumps(
                    {
                        "status": "completed",
                        "campaign_id": kwargs.get("campaign_id"),
                        "trial_id": "trial_after_review",
                        "decision": "promote",
                    }
                )

            assert agent.tools_handler.subagent_manager is not None
            agent.tools_handler.subagent_manager.run_sync = fake_run_sync  # type: ignore[method-assign]
            agent.tools_handler.run_campaign_trial = fake_run_campaign_trial  # type: ignore[method-assign]

            payload = agent.post_tool_hook_node({})
            message = payload["messages"][0]

            self.assertEqual(calls["run_sync"]["subagent_type"], "implementation_evaluator")
            self.assertEqual(calls["run_campaign_trial"]["campaign_id"], "campaign_123")
            self.assertEqual(
                calls["run_campaign_trial"]["dataset_config_overrides"],
                {"datasets": [{"dataset_id": "simulation_gene_2d"}]},
            )
            self.assertIn("Automatically resumed `run_campaign_trial(...)`", str(message.content))
            self.assertIn("trial_after_review", str(message.content))

    def test_review_resume_chains_followup_inference_review_before_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            resume_after_review = {
                "tool": "run_campaign_trial",
                "args": {
                    "campaign_id": "campaign_123",
                    "dataset_config_overrides": {"datasets": [{"dataset_id": "weinreb_erythroid"}]},
                },
            }
            state = ensure_runtime_v2_state(
                {
                    "session_id": "implementation-inference-chain-test",
                    "output_dir": tmpdir,
                    "runtime_action": {
                        "kind": "implementation_agent_review",
                        "algorithm_id": "custom_algo",
                        "proposal_id": "proposal_123",
                        "purpose": "campaign",
                        "review_hash": "impl_hash_123",
                        "fingerprint_payload": {"proposal": {"proposal_id": "proposal_123"}},
                        "relevant_paths": [str(tmpdir)],
                        "campaign_id": "campaign_123",
                        "campaign_trial_count": 0,
                        "resume_after_review": resume_after_review,
                    },
                }
            )
            agent = CytoBridgeAgent(MinimalLLM(), state, agent_role="planner", agent_id="planner")
            calls = {"run_sync_types": [], "run_campaign_trial": []}

            def fake_run_sync(**kwargs):  # noqa: ANN003
                subagent_type = str(kwargs.get("subagent_type") or "")
                calls["run_sync_types"].append(subagent_type)
                if subagent_type == "implementation_evaluator":
                    return {
                        "status": "completed",
                        "summary": "Implementation matches the approved proposal.",
                        "findings": [],
                        "artifact_refs": [],
                        "proposed_state_updates": {
                            "implementation_review": {
                                "decision": "approve",
                                "reviewer_feedback": "No proposal-required mechanism is omitted.",
                                "blocking_issues": [],
                                "advisory_risks": [],
                                "efficiency_recommendations": [],
                                "generalization_shortcut_risks": [],
                            }
                        },
                        "needs_input_question": "",
                    }
                if subagent_type == "inference_evaluator":
                    return {
                        "status": "completed",
                        "summary": "Inference is a valid t0 rollout.",
                        "findings": [],
                        "artifact_refs": [],
                        "proposed_state_updates": {
                            "inference_review": {
                                "decision": "approve",
                                "reviewer_feedback": "No future-time truth is used to construct predictions.",
                                "blocking_issues": [],
                                "advisory_risks": [],
                            }
                        },
                        "needs_input_question": "",
                    }
                raise AssertionError(f"unexpected subagent_type: {subagent_type}")

            def fake_run_campaign_trial(**kwargs):  # noqa: ANN003
                calls["run_campaign_trial"].append(kwargs)
                if len(calls["run_campaign_trial"]) == 1:
                    state["runtime_action"] = {
                        "kind": "inference_agent_review",
                        "algorithm_id": "custom_algo",
                        "proposal_id": "proposal_123",
                        "purpose": "campaign",
                        "review_hash": "infer_hash_123",
                        "fingerprint_payload": {"has_simulation_hook": True},
                        "relevant_paths": [str(tmpdir)],
                        "resume_after_review": resume_after_review,
                    }
                    return json.dumps(
                        {
                            "status": "blocked",
                            "reason": "inference_review_required",
                            "campaign_id": kwargs.get("campaign_id"),
                        }
                    )
                return json.dumps(
                    {
                        "status": "completed",
                        "campaign_id": kwargs.get("campaign_id"),
                        "trial_id": "trial_after_inference",
                        "decision": "promote",
                    }
                )

            assert agent.tools_handler.subagent_manager is not None
            agent.tools_handler.subagent_manager.run_sync = fake_run_sync  # type: ignore[method-assign]
            agent.tools_handler.run_campaign_trial = fake_run_campaign_trial  # type: ignore[method-assign]

            payload = agent.post_tool_hook_node({})
            combined_content = "\n".join(str(message.content) for message in payload["messages"])

            self.assertEqual(calls["run_sync_types"], ["implementation_evaluator", "inference_evaluator"])
            self.assertEqual(len(calls["run_campaign_trial"]), 2)
            self.assertEqual(state["runtime_action"], {})
            self.assertIn("Implementation evaluator review completed", combined_content)
            self.assertIn("Inference evaluator review completed", combined_content)
            self.assertIn("trial_after_inference", combined_content)
