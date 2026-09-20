from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import MethodType
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cytobridge_agent.runtime_v2.agent import CytoBridgeAgent, _runtime_recursion_limit
from cytobridge_agent.runtime_v2.graph import build_runtime_graph
from cytobridge_agent.runtime_v2.paper_review_freshness import build_paper_review_manifest
from cytobridge_agent.runtime_v2.state import (
    DEFAULT_STOP_HOOK_MAX_TRIGGERS,
    DEFAULT_STOP_HOOK_MODE,
    DEFAULT_STOP_HOOK_PROMPT,
    ensure_runtime_v2_state,
)


class FinalAnswerLLM:
    def bind_tools(self, tools):  # noqa: ANN001
        del tools
        return self

    def invoke(self, messages):  # noqa: ANN001
        del messages
        return AIMessage(
            content="phase=final_answer\nAll work is complete.",
            additional_kwargs={"phase": "final_answer"},
        )


class StopHookVerdictTextLLM:
    def bind_tools(self, tools):  # noqa: ANN001
        del tools
        return self

    def invoke(self, messages):  # noqa: ANN001
        del messages
        return AIMessage(content='{"decision":"pass","reason":"internal verdict leaked"}')


class PaperDoneLLM:
    def bind_tools(self, tools):  # noqa: ANN001
        del tools
        return self

    def invoke(self, messages):  # noqa: ANN001
        del messages
        return AIMessage(
            content="phase=final_answer\n我已经完成 outputs/paper/main.tex 的重写和编译。",
            additional_kwargs={"phase": "final_answer"},
        )


class InvalidStopHookThenRepairLLM:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):  # noqa: ANN001
        del tools
        return self

    def invoke(self, messages):  # noqa: ANN001
        del messages
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content=(
                    "你问得很及时。这里是用户答案风格的错误输出。\n\n"
                    "```json\n"
                    '{"name":"claim_metric","direction":"greater"}\n'
                    "```"
                )
            )
        return AIMessage(content='{"decision":"pass","reason":"Repaired stop-hook verdict."}')


class CapturingStopHookLLM:
    def __init__(self) -> None:
        self.messages = []

    def bind_tools(self, tools):  # noqa: ANN001
        del tools
        return self

    def invoke(self, messages):  # noqa: ANN001
        self.messages.append(list(messages))
        return AIMessage(content='{"decision":"pass","reason":"Captured."}')


class StubStopHookSubagentManager:
    def __init__(self, result):  # noqa: ANN001
        self.result = result
        self.calls = []

    def run_sync(self, **kwargs):  # noqa: ANN003
        self.calls.append(dict(kwargs))
        return dict(self.result)


class SequenceStopHookSubagentManager:
    def __init__(self, results):  # noqa: ANN001
        self.results = list(results)
        self.calls = []

    def run_sync(self, **kwargs):  # noqa: ANN003
        self.calls.append(dict(kwargs))
        if self.results:
            return dict(self.results.pop(0))
        return {}


class StopHookTests(unittest.TestCase):
    def test_abort_interrupted_turn_marks_stop_and_resets_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "abort-interrupted-turn",
                    "output_dir": tmpdir,
                }
            )
            agent = CytoBridgeAgent(FinalAnswerLLM(), state, agent_role="planner", agent_id="planner")
            old_thread_id = agent.thread_id
            agent.chat_history = [
                HumanMessage(content="old question"),
                AIMessage(content="phase=final_answer\nold answer"),
                HumanMessage(content="current question"),
                AIMessage(
                    content="phase=commentary\nI will call a tool.",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"file_path": "stale.txt"},
                            "id": "call_stale",
                        }
                    ],
                ),
                ToolMessage(content="stale tool output", tool_call_id="call_stale"),
            ]

            agent.abort_interrupted_turn("Stopped by user.")

            self.assertNotEqual(agent.thread_id, old_thread_id)
            self.assertTrue(agent._seed_full_history_next_run)
            self.assertEqual(len(agent.chat_history), 6)
            self.assertIsInstance(agent.chat_history[0], HumanMessage)
            self.assertEqual(agent.chat_history[0].content, "old question")
            self.assertIsInstance(agent.chat_history[-1], AIMessage)
            self.assertIn("interrupted turn was stopped", agent.chat_history[-1].content)
            self.assertTrue(any(isinstance(message, ToolMessage) for message in agent.chat_history))
            self.assertEqual(len(state.get("messages") or []), 6)

    def test_runtime_recursion_limit_defaults_to_1000_and_ignores_max_turns(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_runtime_recursion_limit(), 10000)
            self.assertEqual(_runtime_recursion_limit(max_turns=1), 10000)
            self.assertEqual(_runtime_recursion_limit(max_turns=1, enforce_turn_bound=True), 10000)

    def test_stop_hook_defaults_to_prompt_lifecycle_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "stop-hook-defaults",
                    "output_dir": tmpdir,
                }
            )
            agent = CytoBridgeAgent(FinalAnswerLLM(), state, agent_role="planner", agent_id="planner")

            self.assertFalse(agent._stop_hook_enabled())
            state["stop_hook_enabled"] = True
            self.assertEqual(agent._stop_hook_mode(), DEFAULT_STOP_HOOK_MODE)
            self.assertIn("algorithm lifecycle", agent._stop_hook_prompt())
            self.assertIn("algorithm_lifecycle_status", agent._stop_hook_prompt())
            self.assertIn("final_regression", agent._stop_hook_prompt())
            self.assertIn("paper_reviewer", agent._stop_hook_prompt())
            self.assertIn("file-hash manifest", agent._stop_hook_prompt())
            self.assertEqual(agent._stop_hook_prompt(), DEFAULT_STOP_HOOK_PROMPT)
            self.assertEqual(agent._stop_hook_max_triggers(), DEFAULT_STOP_HOOK_MAX_TRIGGERS)

    def test_stop_hook_allows_high_lifecycle_trigger_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "stop-hook-high-budget",
                    "output_dir": tmpdir,
                    "stop_hook_max_triggers": 50,
                }
            )
            agent = CytoBridgeAgent(FinalAnswerLLM(), state, agent_role="planner", agent_id="planner")

            self.assertEqual(agent._stop_hook_max_triggers(), 50)

    def test_prompt_stop_hook_includes_campaign_stage_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = CapturingStopHookLLM()
            state = ensure_runtime_v2_state(
                {
                    "session_id": "stop-hook-campaign-snapshot",
                    "output_dir": tmpdir,
                    "stop_hook_enabled": True,
                    "stop_hook_mode": "prompt",
                    "active_algorithm_context": {
                        "algorithm_id": "algox",
                        "algorithm_lifecycle_status": "developing",
                        "algorithm_lifecycle_status_reason": "stage2 formal gate is blocked",
                        "proposal_id": "proposal_1",
                        "proposal_status": "approved",
                        "workspace_path": f"{tmpdir}/algox",
                        "active_snapshot_id": "snapshot_1",
                        "dirty_since_snapshot": False,
                        "dirty_paths": [],
                    },
                    "active_algorithm_campaign": {
                        "campaign_id": "campaign_1",
                        "algorithm_id": "algox",
                        "status": "active",
                        "algorithm_lifecycle_status": "developing",
                        "algorithm_lifecycle_status_reason": "stage2 formal gate is blocked",
                        "current_stage": "stage2_claim_validation",
                        "stage_order": [
                            "stage1_feasibility",
                            "stage2_claim_validation",
                            "stage3_tuning",
                            "final_regression",
                        ],
                        "stages": {
                            "stage1_feasibility": {
                                "status": "passed",
                                "gate_ready": True,
                                "gate_passed_trial_id": "trial_stage1",
                            },
                            "stage2_claim_validation": {
                                "status": "active",
                                "active_best_trial_id": "trial_stage2",
                                "last_gate_check": {"ok": False, "blockers": ["missing baseline claim metric"]},
                            },
                        },
                        "stage_statuses": {
                            "stage1_feasibility": {
                                "stage_status": "passed",
                                "stage_internal_status": "active_best_promoted",
                                "stage_gate_status": "passed",
                                "user_facing_status": "stage1_feasibility passed",
                            },
                            "stage2_claim_validation": {
                                "stage_status": "active",
                                "stage_internal_status": "active_best_promoted",
                                "stage_gate_status": "blocked_missing_baseline_claim_metric",
                                "user_facing_status": "Stage 2 trial promoted, formal gate blocked.",
                                "next_required_action": "compute baseline claim metric",
                            },
                        },
                    },
                }
            )
            agent = CytoBridgeAgent(llm, state, agent_role="planner", agent_id="planner")

            decision = agent._evaluate_prompt_stop_hook(
                messages=[
                    HumanMessage(content="Run the full algorithm lifecycle."),
                    AIMessage(
                        content="phase=final_answer\nStage 1 passed, so I am done.",
                        additional_kwargs={"phase": "final_answer"},
                    ),
                ],
                last_message=AIMessage(
                    content="phase=final_answer\nStage 1 passed, so I am done.",
                    additional_kwargs={"phase": "final_answer"},
                ),
                custom_prompt=DEFAULT_STOP_HOOK_PROMPT,
            )

            self.assertEqual(decision["decision"], "pass")
            request_text = str(llm.messages[0][-1].content)
            self.assertIn("Structured workflow/campaign snapshot", request_text)
            self.assertIn("stage2_claim_validation", request_text)
            self.assertIn("algorithm_lifecycle_status", request_text)
            self.assertIn("developing", request_text)
            self.assertIn("blocked_missing_baseline_claim_metric", request_text)
            self.assertIn("final_regression", request_text)
            self.assertIn("paper_reviewer", request_text)
            self.assertIn("lifecycle_state", request_text)
            self.assertIn('"status": "gate_blocked"', request_text)

    def test_stop_hook_json_extractor_handles_multiple_json_objects(self) -> None:
        payload = CytoBridgeAgent._extract_json_payload(
            'prefix\n{"decision":"pass","reason":"first valid verdict"}\n'
            '{"decision":"block","reason":"second object"}\ntrailer'
        )
        self.assertEqual(payload["decision"], "pass")
        self.assertEqual(payload["reason"], "first valid verdict")

    def test_stop_hook_decision_extractor_rejects_unrelated_json(self) -> None:
        with self.assertRaises(ValueError):
            CytoBridgeAgent._extract_stop_hook_decision_payload(
                '```json\n{"name":"claim_metric","direction":"greater"}\n```'
            )

    def test_prompt_stop_hook_repairs_non_verdict_response_internally(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = InvalidStopHookThenRepairLLM()
            state = ensure_runtime_v2_state(
                {
                    "session_id": "stop-hook-internal-repair",
                    "output_dir": tmpdir,
                    "stop_hook_enabled": True,
                    "stop_hook_mode": "prompt",
                    "stop_hook_prompt": "Pass only when the answer is complete.",
                }
            )
            agent = CytoBridgeAgent(llm, state, agent_role="planner", agent_id="planner")

            decision = agent._evaluate_prompt_stop_hook(
                messages=[
                    HumanMessage(content="What gaps remain?"),
                    AIMessage(
                        content="phase=final_answer\nHere is a complete framework gap analysis.",
                        additional_kwargs={"phase": "final_answer"},
                    ),
                ],
                last_message=AIMessage(
                    content="phase=final_answer\nHere is a complete framework gap analysis.",
                    additional_kwargs={"phase": "final_answer"},
                ),
                custom_prompt="Pass only when the answer is complete.",
            )

            self.assertEqual(llm.calls, 2)
            self.assertEqual(decision["decision"], "pass")
            self.assertEqual(decision["reason"], "Repaired stop-hook verdict.")

    def test_agent_node_guards_internal_stop_hook_verdict_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "stop-hook-verdict-leak-guard",
                    "output_dir": tmpdir,
                    "stop_hook_enabled": True,
                }
            )
            agent = CytoBridgeAgent(StopHookVerdictTextLLM(), state, agent_role="planner", agent_id="planner")

            payload = agent.agent_node(
                {
                    "messages": [HumanMessage(content="Give me a normal answer.")],
                    "commentary_retries": 0,
                    "turn_system_prompt": "",
                    "enable_multimodal": False,
                }
            )

            message = payload["messages"][-1]
            self.assertEqual(message.additional_kwargs["phase"], "commentary")
            self.assertIn("internal stop-hook verdict", str(message.content))
            self.assertNotIn('"decision":"pass"', str(message.content))

    def test_agent_node_blocks_paper_completion_without_reviewer_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "paper-reviewer-completion-guard",
                    "output_dir": tmpdir,
                }
            )
            agent = CytoBridgeAgent(PaperDoneLLM(), state, agent_role="planner", agent_id="planner")

            payload = agent.agent_node(
                {
                    "messages": [HumanMessage(content="请重写 outputs/paper/main.tex 这篇论文。")],
                    "commentary_retries": 0,
                    "turn_system_prompt": "",
                    "enable_multimodal": False,
                }
            )

            message = payload["messages"][-1]
            self.assertEqual(message.additional_kwargs["phase"], "commentary")
            self.assertEqual(payload["commentary_retries"], 0)
            self.assertIn("paper_reviewer", str(message.content))
            self.assertIn("approve", str(message.content))
            self.assertIn("file-hash manifest", str(message.content))

    def test_agent_node_allows_paper_completion_after_reviewer_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paper_dir = Path(tmpdir) / "outputs" / "paper"
            sections_dir = paper_dir / "sections"
            sections_dir.mkdir(parents=True)
            (paper_dir / "main.tex").write_text("\\input{sections/method}\n", encoding="utf-8")
            (sections_dir / "method.tex").write_text("\\section{Method}\n", encoding="utf-8")
            review_manifest = build_paper_review_manifest([str(paper_dir)])
            state = ensure_runtime_v2_state(
                {
                    "session_id": "paper-reviewer-completion-approved",
                    "output_dir": tmpdir,
                    "subagent_registry": {
                        "paper-reviewer-1": {
                            "subagent_type": "paper_reviewer",
                            "status": "completed",
                            "finished_at": "2026-05-02T10:00:00",
                            "last_result": {
                                "status": "completed",
                                "summary": "Verdict: approve.",
                                "proposed_state_updates": {
                                    "paper_review": {
                                        "decision": "approve",
                                        "reviewer_feedback": "No blocking issues.",
                                        "blocking_issues": [],
                                        "reviewed_file_hashes": review_manifest,
                                    }
                                },
                            },
                        }
                    },
                }
            )
            agent = CytoBridgeAgent(PaperDoneLLM(), state, agent_role="planner", agent_id="planner")

            payload = agent.agent_node(
                {
                    "messages": [HumanMessage(content="请重写 outputs/paper/main.tex 这篇论文。")],
                    "commentary_retries": 0,
                    "turn_system_prompt": "",
                    "enable_multimodal": False,
                }
            )

            message = payload["messages"][-1]
            self.assertEqual(message.additional_kwargs["phase"], "final_answer")
            self.assertIn("完成 outputs/paper/main.tex", str(message.content))

    def test_resume_rebuilds_graph_after_restoring_stop_hook_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_case = self
            checkpoint_data = {
                "agent_state": {
                    "session_id": "resume-stop-hook",
                    "user_goal": {"raw_question": "resume stop hook"},
                    "output_dir": tmpdir,
                    "stop_hook_enabled": True,
                    "stop_hook_mode": "prompt",
                },
                "metadata": {"output_dir": tmpdir},
                "messages": [],
                "agent_snapshots": {},
                "agent_histories": {},
                "events_log": [],
            }

            class FakeConversationStore:
                def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
                    del args, kwargs

                def generate_session_id(self) -> str:
                    return "fresh-session"

                def get_conversation(self, session_id: str):  # noqa: ANN201
                    test_case.assertEqual(session_id, "resume-stop-hook")
                    return checkpoint_data

            with patch("cytobridge_agent.conversation_store.ConversationStore", FakeConversationStore):
                from cytobridge_agent.interactive import InteractiveSession

                session = InteractiveSession.from_checkpoint("resume-stop-hook", FinalAnswerLLM())

            self.assertTrue(session.state["stop_hook_enabled"])
            self.assertIn("stop_hooks", session.planner.graph.get_graph().nodes)

    def test_graph_routes_final_answer_through_stop_hook_before_end(self) -> None:
        agent_calls = {"count": 0}
        stop_hook_calls = {"count": 0}

        def agent_node(graph_state):  # noqa: ANN001
            del graph_state
            agent_calls["count"] += 1
            if agent_calls["count"] == 1:
                return {
                    "messages": [
                    AIMessage(
                        content="Draft final answer.",
                        additional_kwargs={"phase": "final_answer"},
                    )
                    ],
                    "commentary_retries": 0,
                }
            return {
                "messages": [
                    AIMessage(
                        content="Final answer after stop-hook review.",
                        additional_kwargs={"phase": "final_answer"},
                    )
                ],
                "commentary_retries": 0,
            }

        def stop_hook_node(graph_state):  # noqa: ANN001
            stop_hook_calls["count"] += 1
            if stop_hook_calls["count"] == 1:
                last_message = graph_state["messages"][-1]
                self.assertIn("Draft final answer.", str(last_message.content))
                return {
                    "messages": [
                        AIMessage(
                            content="Do one more grounded check before ending.",
                            additional_kwargs={"phase": "commentary"},
                        )
                    ],
                    "commentary_retries": 0,
                }
            return {}

        graph = build_runtime_graph(agent_node, [], stop_hook_node=stop_hook_node)

        final_ai_messages = []
        for event in graph.stream(
            {
                "messages": [HumanMessage(content="Finish the task.")],
                "commentary_retries": 0,
                "turn_system_prompt": "",
                "enable_multimodal": False,
            },
            stream_mode="updates",
            config={"configurable": {"thread_id": "stop-hook-test", "checkpoint_ns": "stop-hook-test"}},
        ):
            for node_payload in event.values():
                if not isinstance(node_payload, dict):
                    continue
                for message in node_payload.get("messages", []):
                    if isinstance(message, AIMessage):
                        final_ai_messages.append(message)

        self.assertEqual(agent_calls["count"], 2)
        self.assertEqual(stop_hook_calls["count"], 2)
        self.assertEqual(str(final_ai_messages[-1].content), "Final answer after stop-hook review.")
        self.assertEqual(final_ai_messages[-1].additional_kwargs.get("phase"), "final_answer")
        self.assertTrue(any("Do one more grounded check" in str(msg.content) for msg in final_ai_messages))

    def test_terminal_commentary_routes_through_stop_hook_before_end(self) -> None:
        stop_hook_calls = {"count": 0}

        def agent_node(graph_state):  # noqa: ANN001
            del graph_state
            return {
                "messages": [
                    AIMessage(
                        content="phase=commentary\nI cannot continue.",
                        additional_kwargs={"phase": "commentary"},
                    )
                ],
                "commentary_retries": 2,
            }

        def stop_hook_node(graph_state):  # noqa: ANN001
            stop_hook_calls["count"] += 1
            self.assertIn("I cannot continue", str(graph_state["messages"][-1].content))
            return {"stop_hook_triggers": stop_hook_calls["count"]}

        graph = build_runtime_graph(agent_node, [], stop_hook_node=stop_hook_node)
        result = graph.invoke(
            {
                "messages": [HumanMessage(content="Run autonomously.")],
                "commentary_retries": 0,
                "stop_hook_triggers": 0,
                "turn_system_prompt": "",
                "enable_multimodal": False,
            },
            config={"configurable": {"thread_id": "stop-hook-commentary-route-test", "checkpoint_ns": "stop-hook-test"}},
        )

        self.assertEqual(stop_hook_calls["count"], 1)
        self.assertEqual(result["stop_hook_triggers"], 1)

    def test_graph_blocks_needs_input_and_continues_until_goal_complete(self) -> None:
        agent_calls = {"count": 0}
        stop_hook_calls = {"count": 0}

        def agent_node(graph_state):  # noqa: ANN001
            del graph_state
            agent_calls["count"] += 1
            if agent_calls["count"] == 1:
                return {
                    "messages": [
                        AIMessage(
                            content="Which benchmark should I use?",
                            additional_kwargs={"phase": "needs_input"},
                        )
                    ],
                    "commentary_retries": 0,
                }
            return {
                "messages": [
                    AIMessage(
                        content="The requested lifecycle is complete.",
                        additional_kwargs={"phase": "final_answer"},
                    )
                ],
                "commentary_retries": 0,
            }

        def stop_hook_node(graph_state):  # noqa: ANN001
            messages = graph_state["messages"]
            stop_hook_calls["count"] += 1
            last = messages[-1]
            if "Which benchmark" in str(last.content):
                return {
                    "messages": [
                        AIMessage(
                            content="Stop hook blocked needs_input because the agent must continue autonomously.",
                            additional_kwargs={"phase": "commentary"},
                        )
                    ],
                    "commentary_retries": 0,
                    "stop_hook_triggers": stop_hook_calls["count"],
                }
            return {"stop_hook_triggers": stop_hook_calls["count"]}

        graph = build_runtime_graph(agent_node, [], stop_hook_node=stop_hook_node)
        result = graph.invoke(
            {
                "messages": [HumanMessage(content="Run the algorithm lifecycle autonomously.")],
                "commentary_retries": 0,
                "stop_hook_triggers": 0,
                "turn_system_prompt": "",
                "enable_multimodal": False,
            },
            config={"configurable": {"thread_id": "stop-hook-needs-input-loop-test", "checkpoint_ns": "stop-hook-test"}},
        )

        self.assertEqual(agent_calls["count"], 2)
        self.assertEqual(stop_hook_calls["count"], 2)
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        self.assertIn("Which benchmark", str(ai_messages[0].content))
        self.assertIn("blocked needs_input", str(ai_messages[1].content))
        self.assertEqual(str(ai_messages[-1].content), "The requested lifecycle is complete.")
        self.assertEqual(ai_messages[-1].additional_kwargs.get("phase"), "final_answer")

    def test_planner_stop_hook_turns_final_answer_into_needs_input_when_state_requires_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "stop-hook-needs-input",
                    "output_dir": tmpdir,
                    "stop_hook_enabled": True,
                    "stop_hook_mode": "builtin",
                    "planner_phase": "needs_input",
                    "planner_need": {
                        "source": "algorithm_proposal",
                        "question": "Please approve the proposal before continuing.",
                        "reason": "algorithm_proposal_pending_review",
                    },
                }
            )
            agent = CytoBridgeAgent(FinalAnswerLLM(), state, agent_role="planner", agent_id="planner")

            result = agent.run(user_instruction="Can we finish now?")

            self.assertEqual(result["status"], "needs_input")
            self.assertEqual(result["phase"], "needs_input")
            self.assertIn("Please approve the proposal before continuing.", result["content"])
            self.assertEqual(state["planner_phase"], "needs_input")

    def test_stop_hook_can_be_disabled_per_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "stop-hook-disabled",
                    "output_dir": tmpdir,
                    "stop_hook_enabled": False,
                    "planner_phase": "needs_input",
                    "planner_need": {
                        "source": "algorithm_proposal",
                        "question": "Please approve the proposal before continuing.",
                        "reason": "algorithm_proposal_pending_review",
                    },
                }
            )
            agent = CytoBridgeAgent(FinalAnswerLLM(), state, agent_role="planner", agent_id="planner")

            result = agent.run(user_instruction="Can we finish now with stop hook disabled?")

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["phase"], "final_answer")
            self.assertEqual(result["content"], "All work is complete.")
            self.assertEqual(state["planner_phase"], "working")

    def test_stop_hook_blocks_finalization_when_runtime_action_is_still_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "stop-hook-runtime-action",
                    "output_dir": tmpdir,
                    "stop_hook_enabled": True,
                    "runtime_action": {
                        "kind": "custom_runtime_gate",
                    },
                }
            )
            agent = CytoBridgeAgent(FinalAnswerLLM(), state, agent_role="planner", agent_id="planner")

            payload = agent.stop_hook_node(
                {
                    "messages": [
                        HumanMessage(content="finish"),
                        AIMessage(
                            content="phase=final_answer\nDone.",
                            additional_kwargs={"phase": "final_answer"},
                        ),
                    ]
                }
            )

            self.assertEqual(state["runtime_action"], {})
            self.assertEqual(payload["commentary_retries"], 0)
            self.assertIn("pending runtime action (`custom_runtime_gate`)", str(payload["messages"][0].content))

    def test_prompt_mode_stop_hook_uses_session_prompt_and_can_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "stop-hook-prompt-mode",
                    "output_dir": tmpdir,
                    "stop_hook_enabled": True,
                    "stop_hook_mode": "prompt",
                    "stop_hook_prompt": "Block final answers that lack a grounded justification.",
                }
            )
            agent = CytoBridgeAgent(FinalAnswerLLM(), state, agent_role="planner", agent_id="planner")
            observed = {}

            def fake_prompt_eval(self, *, messages, last_message, custom_prompt):  # noqa: ANN001
                observed["message_count"] = len(messages)
                observed["custom_prompt"] = custom_prompt
                observed["last_message"] = str(last_message.content)
                return {
                    "decision": "block",
                    "reason": "The final answer is not grounded enough.",
                    "question": "",
                }

            agent._evaluate_prompt_stop_hook = MethodType(fake_prompt_eval, agent)
            payload = agent.stop_hook_node(
                {
                    "messages": [
                        HumanMessage(content="finish"),
                        AIMessage(
                            content="phase=final_answer\nAll work is complete.",
                            additional_kwargs={"phase": "final_answer"},
                        ),
                    ]
                }
            )

            self.assertEqual(observed["custom_prompt"], "Block final answers that lack a grounded justification.")
            self.assertIn("All work is complete.", observed["last_message"])
            self.assertEqual(payload["commentary_retries"], 0)
            self.assertEqual(payload["messages"][0].additional_kwargs["hook_mode"], "prompt")
            self.assertIn("not grounded enough", str(payload["messages"][0].content))

    def test_prompt_mode_stop_hook_reviews_needs_input_before_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "stop-hook-needs-input-prompt-mode",
                    "output_dir": tmpdir,
                    "stop_hook_enabled": True,
                    "stop_hook_mode": "prompt",
                    "stop_hook_prompt": "Keep running automatically unless real user input is unavoidable.",
                }
            )
            agent = CytoBridgeAgent(FinalAnswerLLM(), state, agent_role="planner", agent_id="planner")
            observed = {}

            def fake_prompt_eval(self, *, messages, last_message, custom_prompt):  # noqa: ANN001
                observed["custom_prompt"] = custom_prompt
                observed["last_message"] = str(last_message.content)
                return {
                    "decision": "block",
                    "reason": "The agent should continue autonomously instead of asking the user.",
                    "question": "",
                }

            agent._evaluate_prompt_stop_hook = MethodType(fake_prompt_eval, agent)
            payload = agent.stop_hook_node(
                {
                    "messages": [
                        HumanMessage(content="run autonomously"),
                        AIMessage(
                            content="phase=needs_input\nWhich benchmark should I use?",
                            additional_kwargs={"phase": "needs_input"},
                        ),
                    ]
                }
            )

            self.assertEqual(observed["custom_prompt"], "Keep running automatically unless real user input is unavoidable.")
            self.assertIn("Which benchmark", observed["last_message"])
            self.assertIsInstance(payload["messages"][0], HumanMessage)
            self.assertEqual(payload["messages"][0].additional_kwargs["phase"], "commentary")
            self.assertTrue(payload["messages"][0].additional_kwargs["is_meta"])
            self.assertNotEqual(state.get("planner_phase"), "needs_input")
            self.assertIn("Do not ask the user", str(payload["messages"][0].content))
            self.assertIn("continue autonomously", str(payload["messages"][0].content))

    def test_prompt_mode_stop_hook_converts_stale_planner_need_to_continue_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "stop-hook-stale-planner-need",
                    "output_dir": tmpdir,
                    "stop_hook_enabled": True,
                    "stop_hook_mode": "prompt",
                    "stop_hook_prompt": "Run autonomously until the task is complete.",
                    "planner_phase": "needs_input",
                    "planner_need": {
                        "source": "implementation_review",
                        "question": "Choose whether to simplify the proposal.",
                        "reason": "implementation_review_blocked",
                    },
                }
            )
            agent = CytoBridgeAgent(FinalAnswerLLM(), state, agent_role="planner", agent_id="planner")

            payload = agent.stop_hook_node(
                {
                    "messages": [
                        HumanMessage(content="run autonomously"),
                        AIMessage(
                            content="phase=final_answer\nI need the user to choose.",
                            additional_kwargs={"phase": "final_answer"},
                        ),
                    ],
                    "stop_hook_triggers": 0,
                }
            )

            self.assertIsInstance(payload["messages"][0], HumanMessage)
            self.assertEqual(payload["messages"][0].additional_kwargs["phase"], "commentary")
            self.assertEqual(state.get("planner_phase"), "working")
            self.assertEqual(state.get("planner_need"), {})
            self.assertIn("implementation_review_blocked", str(payload["messages"][0].content))
            self.assertIn("Do not ask the user", str(payload["messages"][0].content))

    def test_agent_mode_stop_hook_needs_input_verdict_forces_continue(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "stop-hook-agent-mode",
                    "output_dir": tmpdir,
                    "stop_hook_enabled": True,
                    "stop_hook_mode": "agent",
                    "stop_hook_prompt": "Run autonomously until the task is complete.",
                }
            )
            agent = CytoBridgeAgent(FinalAnswerLLM(), state, agent_role="planner", agent_id="planner")
            stub_manager = StubStopHookSubagentManager(
                {
                    "status": "completed",
                    "summary": "Need explicit user confirmation.",
                    "findings": [],
                    "artifact_refs": [],
                    "proposed_state_updates": {
                        "stop_hook": {
                            "decision": "needs_input",
                            "reason": "The task is not complete and the planner asked the user.",
                            "question": "Which option should I choose?",
                        }
                    },
                    "needs_input_question": "",
                }
            )
            agent.tools_handler.subagent_manager = stub_manager

            payload = agent.stop_hook_node(
                {
                    "messages": [
                        HumanMessage(content="finish"),
                        AIMessage(
                            content="phase=final_answer\nAll work is complete.",
                            additional_kwargs={"phase": "final_answer"},
                        ),
                    ]
                }
            )

            self.assertEqual(len(stub_manager.calls), 1)
            self.assertEqual(stub_manager.calls[0]["subagent_type"], "general")
            self.assertIn(
                "Run autonomously until the task is complete.",
                stub_manager.calls[0]["context_notes"],
            )
            self.assertIsInstance(payload["messages"][0], HumanMessage)
            self.assertEqual(payload["messages"][0].additional_kwargs["phase"], "commentary")
            self.assertNotEqual(state.get("planner_phase"), "needs_input")
            self.assertIn("not allowed for autonomous stop-hook mode", str(payload["messages"][0].content))
            self.assertIn("Do not ask the user", str(payload["messages"][0].content))

    def test_agent_mode_stop_hook_repairs_missing_structured_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "stop-hook-agent-mode-repair",
                    "output_dir": tmpdir,
                    "stop_hook_enabled": True,
                    "stop_hook_mode": "agent",
                    "stop_hook_prompt": "Run autonomously until the task is complete.",
                }
            )
            agent = CytoBridgeAgent(FinalAnswerLLM(), state, agent_role="planner", agent_id="planner")
            stub_manager = SequenceStopHookSubagentManager(
                [
                    {
                        "status": "completed",
                        "summary": "free-form only",
                        "findings": [],
                        "artifact_refs": [],
                        "proposed_state_updates": {},
                        "needs_input_question": "",
                    },
                    {
                        "status": "completed",
                        "summary": "structured pass",
                        "findings": [],
                        "artifact_refs": [],
                        "proposed_state_updates": {
                            "stop_hook": {
                                "decision": "pass",
                                "reason": "The planner answer is complete.",
                            }
                        },
                        "needs_input_question": "",
                    },
                ]
            )
            agent.tools_handler.subagent_manager = stub_manager

            payload = agent.stop_hook_node(
                {
                    "messages": [
                        HumanMessage(content="finish"),
                        AIMessage(
                            content="phase=final_answer\nAll work is complete.",
                            additional_kwargs={"phase": "final_answer"},
                        ),
                    ]
                }
            )

            self.assertEqual(len(stub_manager.calls), 2)
            self.assertIn("Previous invalid subagent result to reuse", stub_manager.calls[1]["context_notes"])
            self.assertIn("free-form only", stub_manager.calls[1]["context_notes"])
            self.assertEqual(payload.get("stop_hook_triggers"), 1)
            self.assertNotIn("messages", payload)

    def test_stop_hook_trigger_limit_converts_loop_into_needs_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "stop-hook-trigger-limit",
                    "output_dir": tmpdir,
                    "stop_hook_enabled": True,
                    "stop_hook_max_triggers": 1,
                    "runtime_action": {
                        "kind": "custom_runtime_gate",
                    },
                }
            )
            agent = CytoBridgeAgent(FinalAnswerLLM(), state, agent_role="planner", agent_id="planner")

            result = agent.run(user_instruction="Try to finalize.")

            self.assertEqual(result["status"], "needs_input")
            self.assertEqual(result["phase"], "needs_input")
            self.assertIn("Runtime stop-hook trigger limit reached (1)", result["content"])
            self.assertEqual(state["planner_phase"], "needs_input")
            self.assertEqual(int(state["planner_need"]["stop_hook_trigger_limit"]), 1)
            self.assertEqual(int(state["planner_need"]["stop_hook_trigger_count"]), 1)

    def test_prompt_stop_hook_stops_at_trigger_limit_without_extra_llm_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "stop-hook-prompt-limit",
                    "output_dir": tmpdir,
                    "stop_hook_enabled": True,
                    "stop_hook_mode": "prompt",
                    "stop_hook_max_triggers": 2,
                }
            )
            agent = CytoBridgeAgent(FinalAnswerLLM(), state, agent_role="planner", agent_id="planner")
            calls = {"count": 0}

            def fake_prompt_eval(_agent, *, messages, last_message, custom_prompt):  # noqa: ANN001
                del _agent, messages, last_message
                calls["count"] += 1
                self.assertIn("algorithm lifecycle", custom_prompt)
                return {
                    "decision": "block",
                    "reason": "The algorithm lifecycle is not complete.",
                    "question": "",
                }

            agent._evaluate_prompt_stop_hook = MethodType(fake_prompt_eval, agent)
            result = agent.run(user_instruction="Try to finalize before lifecycle completion.")

            self.assertEqual(calls["count"], 2)
            self.assertEqual(result["status"], "needs_input")
            self.assertEqual(result["phase"], "needs_input")
            self.assertIn("Runtime stop-hook trigger limit reached (2)", result["content"])
            self.assertEqual(int(state["planner_need"]["stop_hook_trigger_limit"]), 2)
            self.assertEqual(int(state["planner_need"]["stop_hook_trigger_count"]), 2)
