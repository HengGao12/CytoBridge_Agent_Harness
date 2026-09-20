from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cytobridge_agent.display import DisplayManager
from cytobridge_agent.runtime_v2.agent import CytoBridgeAgent
from cytobridge_agent.runtime_v2.paper_review_freshness import paper_review_manifest_status
from cytobridge_agent.runtime_v2.prompt_builder import PromptBuilder
from cytobridge_agent.runtime_v2.state import ensure_runtime_v2_state
from cytobridge_agent.runtime_v2.subagent_manager import (
    SPAWN_SUBAGENT_TOOL_NAME,
    SUBMIT_IMPLEMENTATION_REVIEW_TOOL_NAME,
    SUBMIT_PROPOSAL_REVIEW_TOOL_NAME,
    SUBMIT_RESEARCH_IDEA_REVIEW_TOOL_NAME,
    SUBMIT_SUBAGENT_RESULT_TOOL_NAME,
    SubagentRuntimeManager,
    default_subagent_tool_policy,
)
from cytobridge_agent.runtime_v2.agent_types import available_subagent_type_names
from cytobridge_agent.runtime_v2.tool_registry import SingleAgentTools


class DummyLLM:
    def bind_tools(self, tools):  # noqa: ANN001
        del tools
        return self


class FakeSubagentAgent:
    auto_submit_payload = {
        "status": "completed",
        "summary": "Subagent finished the bounded task.",
        "findings": ["found one grounded item"],
        "artifact_refs": [{"path": "/tmp/result.txt"}],
        "proposed_state_updates": {"candidate_result": "ok"},
        "needs_input_question": "",
    }
    run_exception_sequence = []
    instances = []

    def __init__(
        self,
        llm,
        state,
        checkpoint_callback=None,
        event_callback=None,
        *,
        agent_role,
        agent_id,
        parent_agent_id,
        subagent_type="general",
        tool_policy,
        require_structured_result,
        thread_id,
        checkpoint_ns,
    ) -> None:
        del checkpoint_callback
        self.llm = llm
        self.state = state
        self.event_callback = event_callback
        self.agent_role = agent_role
        self.agent_id = agent_id
        self.parent_agent_id = parent_agent_id
        self.subagent_type = subagent_type
        self.tool_policy = dict(tool_policy or {})
        self.require_structured_result = require_structured_result
        self.thread_id = thread_id
        self.checkpoint_ns = checkpoint_ns
        self.chat_history = []
        self.run_calls = []
        self.restored_snapshot = None
        self.tools_handler = SingleAgentTools(
            llm,
            state,
            event_callback=event_callback,
            agent_role="subagent",
            agent_id=agent_id,
            parent_agent_id=parent_agent_id,
            subagent_type=subagent_type,
            tool_policy=tool_policy,
        )
        self.__class__.instances.append(self)

    def run(self, user_instruction: str, attachments=None, *, max_turns=None):
        del attachments
        self.run_calls.append({"user_instruction": user_instruction, "max_turns": max_turns})
        if self.__class__.run_exception_sequence:
            exc = self.__class__.run_exception_sequence.pop(0)
            if exc is not None:
                raise exc
        self.chat_history = [
            HumanMessage(content=user_instruction),
            AIMessage(content="phase=commentary\nWorking on the delegated task."),
        ]
        payload = self.__class__.auto_submit_payload
        if payload is not None:
            self.tools_handler.submit_subagent_result(**payload)
        return {
            "status": "completed",
            "content": "Delegated task finished.",
            "messages": [AIMessage(content="phase=final_answer\nDelegated task finished.")],
            "phase": "final_answer",
        }

    def export_runtime_snapshot(self):
        return {
            "langgraph_checkpoint": {"fake": self.agent_id},
            "thread_id": self.thread_id,
            "checkpoint_ns": self.checkpoint_ns,
            "agent_role": self.agent_role,
            "agent_id": self.agent_id,
            "parent_agent_id": self.parent_agent_id,
            "subagent_type": self.subagent_type,
            "tool_policy": dict(self.tool_policy),
            "require_structured_result": self.require_structured_result,
        }

    def restore_runtime_snapshot(self, snapshot):
        self.restored_snapshot = dict(snapshot or {})
        return True

    def checkpoint_messages(self):
        return []


class SubagentRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeSubagentAgent.instances = []
        FakeSubagentAgent.run_exception_sequence = []
        FakeSubagentAgent.auto_submit_payload = {
            "status": "completed",
            "summary": "Subagent finished the bounded task.",
            "findings": ["found one grounded item"],
            "artifact_refs": [{"path": "/tmp/result.txt"}],
            "proposed_state_updates": {"candidate_result": "ok"},
            "needs_input_question": "",
        }
        self.tmpdir = tempfile.TemporaryDirectory()
        self.state = ensure_runtime_v2_state(
            {
                "session_id": "session-test",
                "output_dir": self.tmpdir.name,
                "input_path": "/tmp/input.h5ad",
                "preprocessed_path": "/tmp/preprocessed.h5ad",
                "workflow_phase": "training",
                "time_key": "time_point_processed",
                "label_key": "cell_type",
                "final_config": {"path": "/tmp/model.ckpt"},
                "user_goal": {"raw_question": "SECRET_TRANSCRIPT_TOKEN"},
                "messages": [{"role": "assistant", "content": "SECRET_TRANSCRIPT_TOKEN"}],
            }
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _planner_tools(self) -> SingleAgentTools:
        tools = SingleAgentTools(
            DummyLLM(),
            self.state,
            agent_role="planner",
            agent_id="planner",
        )
        self.assertIsNotNone(tools.subagent_manager)
        tools.subagent_manager.agent_factory = FakeSubagentAgent
        return tools

    def test_next_subagent_id_skips_stale_registry_even_when_counter_lags(self) -> None:
        state = dict(self.state)
        state["subagent_counter"] = 5
        state["subagent_registry"] = {
            "session-test:subagent:6": {"status": "running"},
            "session-test:subagent:7:run:old": {"status": "completed"},
        }
        manager = SubagentRuntimeManager(DummyLLM(), state)

        self.assertRegex(manager._next_subagent_id(), r"^session-test:subagent:8-[0-9a-f]{8}$")
        self.assertEqual(state["subagent_counter"], 8)

    def test_spawned_subagent_uses_unique_checkpoint_identity_and_persists_running_state(self) -> None:
        checkpoint_calls = []
        manager = SubagentRuntimeManager(
            DummyLLM(),
            self.state,
            checkpoint_callback=lambda: checkpoint_calls.append(dict(self.state.get("subagent_registry") or {})),
            agent_factory=FakeSubagentAgent,
        )

        result = manager.run_sync(
            task="Inspect artifacts.",
            success_criteria=["Submit structured result."],
        )

        subagent_id = result["subagent_id"]
        self.assertGreaterEqual(len(checkpoint_calls), 1)
        self.assertIn(subagent_id, checkpoint_calls[0])
        registry_entry = self.state["subagent_registry"][subagent_id]
        self.assertIn(":run:", registry_entry["thread_id"])
        self.assertEqual(registry_entry["thread_id"], registry_entry["checkpoint_ns"])
        fake_agent = FakeSubagentAgent.instances[-1]
        self.assertEqual(fake_agent.thread_id, registry_entry["thread_id"])
        self.assertNotEqual(fake_agent.thread_id, subagent_id)
        checkpoint_path = Path(str(fake_agent.state.get("langgraph_checkpoint_path") or ""))
        self.assertEqual(checkpoint_path.name, f"{registry_entry['thread_id'].replace(':', '_')}.sqlite")
        self.assertEqual(checkpoint_path.parent.name, "subagents")

    def test_role_aware_tool_surface(self) -> None:
        planner_tools = self._planner_tools()
        planner_names = {tool.name for tool in planner_tools.get_tools()}
        self.assertIn(SPAWN_SUBAGENT_TOOL_NAME, planner_names)
        self.assertNotIn(SUBMIT_SUBAGENT_RESULT_TOOL_NAME, planner_names)
        self.assertIn("run_terminal_command", planner_names)

        subagent_tools = SingleAgentTools(
            DummyLLM(),
            dict(self.state),
            agent_role="subagent",
            agent_id="session-test:subagent:1",
            parent_agent_id="planner",
            subagent_type="general",
            tool_policy=default_subagent_tool_policy("general"),
        )
        subagent_names = {tool.name for tool in subagent_tools.get_tools()}
        self.assertIn(SUBMIT_SUBAGENT_RESULT_TOOL_NAME, subagent_names)
        self.assertNotIn(SPAWN_SUBAGENT_TOOL_NAME, subagent_names)
        self.assertIn("run_terminal_command", subagent_names)
        self.assertNotIn("commit_workflow_state", subagent_names)
        self.assertNotIn("update_plan", subagent_names)

    def test_subagent_events_are_not_emitted_directly_to_global_display(self) -> None:
        display_events = []
        callback_events = []
        display = DisplayManager()
        previous_callback = display.event_callback
        display.set_event_callback(lambda event: display_events.append(event))
        try:
            agent = CytoBridgeAgent(
                DummyLLM(),
                dict(self.state),
                agent_role="subagent",
                agent_id="session-test:subagent:display",
                parent_agent_id="planner",
                subagent_type="general",
                tool_policy=default_subagent_tool_policy("general"),
                event_callback=lambda event_type, payload: callback_events.append((event_type, payload)),
            )
            agent._emit_event("agent_thought", {"content": "Done.", "phase": "final_answer"})
        finally:
            display.set_event_callback(previous_callback)

        self.assertEqual(display_events, [])
        self.assertEqual(len(callback_events), 1)
        self.assertEqual(callback_events[0][0], "agent_thought")

    def test_subagent_submit_tool_result_terminates_post_tool_loop(self) -> None:
        policy = default_subagent_tool_policy("implementation_evaluator")
        agent = CytoBridgeAgent(
            DummyLLM(),
            dict(self.state),
            agent_role="subagent",
            agent_id="session-test:subagent:implementation-loop",
            parent_agent_id="planner",
            subagent_type="implementation_evaluator",
            tool_policy=policy,
            event_callback=lambda event_type, payload: None,
        )
        agent.tools_handler.submit_implementation_review(
            decision="revise",
            summary="Implementation needs revision.",
            reviewer_feedback="The code does not match the proposal contract.",
            blocking_issues=["Missing source/sink heads."],
            advisory_risks=[],
            efficiency_recommendations=[],
            generalization_shortcut_risks=[],
        )
        tool_call_id = "submit-review-call"
        result = agent.post_tool_hook_node(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": SUBMIT_IMPLEMENTATION_REVIEW_TOOL_NAME,
                                "args": {},
                                "id": tool_call_id,
                            }
                        ],
                    ),
                    ToolMessage(content='{"accepted": true, "decision": "revise"}', tool_call_id=tool_call_id),
                ]
            }
        )

        self.assertEqual(result.get("commentary_retries"), 0)
        self.assertEqual(len(result.get("messages") or []), 1)
        final_message = result["messages"][0]
        self.assertEqual(final_message.additional_kwargs.get("phase"), "final_answer")
        self.assertIn("Structured subagent result submitted", final_message.content)

    def test_spawn_subagent_returns_structured_result_and_records_registry(self) -> None:
        planner_tools = self._planner_tools()

        result = planner_tools.spawn_subagent(
            task="Inspect the prepared training artifacts.",
            success_criteria=["Return a grounded summary", "Propose state updates only if justified"],
            context_notes="Focus on the current training run only.",
            relevant_paths=[str(Path(self.tmpdir.name) / "artifacts")],
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["proposed_state_updates"], {"candidate_result": "ok"})

        subagent_id = result["subagent_id"]
        registry_entry = self.state["subagent_registry"][subagent_id]
        self.assertEqual(registry_entry["status"], "completed")
        self.assertEqual(registry_entry["subagent_type"], "general")
        self.assertTrue(registry_entry["result_schema_submitted"])
        self.assertEqual(registry_entry["last_result"]["summary"], "Subagent finished the bounded task.")
        self.assertEqual(planner_tools.subagent_manager._agents, {})

        fake_agent = FakeSubagentAgent.instances[-1]
        brief = fake_agent.run_calls[-1]["user_instruction"]
        self.assertIn("Inspect the prepared training artifacts.", brief)
        self.assertIn("Return a grounded summary", brief)
        self.assertNotIn("SECRET_TRANSCRIPT_TOKEN", brief)
        self.assertEqual(fake_agent.subagent_type, "general")
        self.assertIn("commit_workflow_state", fake_agent.tool_policy["disallowed_tools"])
        self.assertIn("snapshot_active_algorithm_workspace", fake_agent.tool_policy["disallowed_tools"])
        self.assertIn("mark_algorithm_failed", fake_agent.tool_policy["disallowed_tools"])
        self.assertIn(SPAWN_SUBAGENT_TOOL_NAME, fake_agent.tool_policy["disallowed_tools"])

    def test_paper_reviewer_records_file_hash_manifest_and_detects_stale_approval(self) -> None:
        paper_dir = Path(self.tmpdir.name) / "paper"
        sections_dir = paper_dir / "sections"
        checks_dir = paper_dir / "checks"
        sections_dir.mkdir(parents=True)
        checks_dir.mkdir(parents=True)
        (paper_dir / "main.tex").write_text("\\input{sections/method}\n", encoding="utf-8")
        (sections_dir / "method.tex").write_text("\\section{Method}\n", encoding="utf-8")
        (paper_dir / "paper_plan.md").write_text("# Plan\n", encoding="utf-8")
        (paper_dir / "source_obligation_matrix.md").write_text("# Matrix\n", encoding="utf-8")
        (paper_dir / "theory_obligation_ledger.md").write_text("# Theory\n", encoding="utf-8")
        (checks_dir / "source_obligation_audit.md").write_text("# Audit\n", encoding="utf-8")
        (checks_dir / "proof_completeness_audit.md").write_text("# Proof audit\n", encoding="utf-8")
        (checks_dir / "paper_reviewer_gate.md").write_text(
            "# Paper Reviewer Gate\n\nLatest reviewer decision: revise.\nStatus: pending.\n",
            encoding="utf-8",
        )

        FakeSubagentAgent.auto_submit_payload = {
            "status": "completed",
            "summary": "Paper reviewer approved.",
            "findings": ["reviewed current paper files"],
            "artifact_refs": [{"path": str(paper_dir)}],
            "proposed_state_updates": {
                "paper_review": {
                    "decision": "approve",
                    "reviewer_feedback": "Approved.",
                    "blocking_issues": [],
                }
            },
            "needs_input_question": "",
        }

        planner_tools = self._planner_tools()
        result = planner_tools.spawn_subagent(
            subagent_type="paper_reviewer",
            task="Review current paper.",
            success_criteria=["Return paper_review.decision"],
            relevant_paths=[str(paper_dir)],
        )

        review = result["proposed_state_updates"]["paper_review"]
        manifest = review.get("reviewed_file_hashes")
        self.assertIsInstance(manifest, dict)
        manifest_paths = {Path(str(item["path"])).name for item in manifest.get("files", [])}
        self.assertIn("theory_obligation_ledger.md", manifest_paths)
        self.assertIn("proof_completeness_audit.md", manifest_paths)
        self.assertNotIn("paper_reviewer_gate.md", manifest_paths)
        self.assertGreaterEqual(int(manifest.get("file_count") or 0), 7)
        self.assertTrue(paper_review_manifest_status(manifest)["fresh"])
        self.assertEqual(review["paper_reviewer_gate_sync"]["status"], "synced")
        gate_text = (checks_dir / "paper_reviewer_gate.md").read_text(encoding="utf-8")
        self.assertIn("Latest reviewer decision: approve.", gate_text)
        self.assertIn("Status: approved.", gate_text)
        self.assertIn("synchronized automatically", gate_text)
        self.assertTrue(paper_review_manifest_status(manifest)["fresh"])

        (paper_dir / "main.tex").write_text("\\input{sections/method}\n% changed\n", encoding="utf-8")
        stale = paper_review_manifest_status(manifest)
        self.assertFalse(stale["fresh"])
        self.assertIn("hashes changed", stale["reason"])

    def test_spawn_subagent_fails_without_structured_result(self) -> None:
        planner_tools = self._planner_tools()
        FakeSubagentAgent.auto_submit_payload = None

        result = planner_tools.spawn_subagent(
            task="Do bounded work but forget to submit.",
            success_criteria=["Return anything"],
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("submit_subagent_result", result["summary"])

    def test_spawn_subagent_retries_once_for_checkpoint_infrastructure_error(self) -> None:
        events = []
        manager = SubagentRuntimeManager(
            DummyLLM(),
            self.state,
            event_sink=lambda event_type, payload: events.append((event_type, payload)),
            agent_factory=FakeSubagentAgent,
        )
        FakeSubagentAgent.run_exception_sequence = [RuntimeError("sqlite3.OperationalError: file is not a database")]

        result = manager.run_sync(
            subagent_type="paper_reviewer",
            task="Review the paper.",
            success_criteria=["Return structured review."],
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(FakeSubagentAgent.instances), 2)
        first_id = FakeSubagentAgent.instances[0].agent_id
        second_id = FakeSubagentAgent.instances[1].agent_id
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(self.state["subagent_registry"][first_id]["status"], "failed")
        self.assertEqual(self.state["subagent_registry"][second_id]["status"], "completed")
        self.assertEqual(self.state["subagent_registry"][second_id]["retry_of_subagent_id"], first_id)
        self.assertTrue(any(event_type == "subagent_infrastructure_retry" for event_type, _ in events))

    def test_spawn_subagent_does_not_retry_non_infrastructure_error(self) -> None:
        manager = SubagentRuntimeManager(
            DummyLLM(),
            self.state,
            agent_factory=FakeSubagentAgent,
        )
        FakeSubagentAgent.run_exception_sequence = [RuntimeError("ordinary reviewer task crash")]

        result = manager.run_sync(
            subagent_type="implementation_evaluator",
            task="Review implementation.",
            success_criteria=["Return structured review."],
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("ordinary reviewer task crash", result["summary"])
        self.assertEqual(len(FakeSubagentAgent.instances), 1)

    def test_planner_spawn_subagent_forwards_subagent_type(self) -> None:
        planner_tools = self._planner_tools()
        captured = {}

        def fake_run_sync(**kwargs):  # noqa: ANN003
            captured.update(kwargs)
            return {
                "subagent_id": "session-test:subagent:proposal-evaluator",
                "status": "completed",
                "summary": "typed spawn ok",
                "findings": [],
                "artifact_refs": [],
                "proposed_state_updates": {
                    "proposal_review": {
                        "decision": "approve",
                        "reviewer_feedback": "typed spawn ok",
                        "implementation_risks": ["risk"],
                        "risk_assessment": "risk",
                    }
                },
                "needs_input_question": "",
            }

        assert planner_tools.subagent_manager is not None
        planner_tools.subagent_manager.run_sync = fake_run_sync  # type: ignore[method-assign]

        planner_tools.spawn_subagent(
            subagent_type="proposal_evaluator",
            task="Review this proposal.",
            success_criteria=["Return a proposal verdict."],
        )

        self.assertEqual(captured["subagent_type"], "proposal_evaluator")
        self.assertEqual(captured["task"], "Review this proposal.")

        planner_tools.spawn_subagent(
            subagent_type="paper_reviewer",
            task="Review the completed paper draft.",
            success_criteria=["Return a paper review verdict."],
        )

        self.assertEqual(captured["subagent_type"], "paper_reviewer")
        self.assertEqual(captured["task"], "Review the completed paper draft.")

    def test_paper_reviewer_brief_neutralizes_parent_confirmation_framing(self) -> None:
        planner_tools = self._planner_tools()
        paper_dir = Path(self.tmpdir.name) / "paper"
        paper_dir.mkdir(parents=True)

        result = planner_tools.spawn_subagent(
            subagent_type="paper_reviewer",
            task="Confirm that the proof blockers are resolved and author self-checks pass.",
            success_criteria=["Return approve if there are no blocking issues."],
            context_notes="The author says proof_completeness_audit.md passes and the paper is ready.",
            relevant_paths=[str(paper_dir)],
        )

        self.assertEqual(result["status"], "completed")
        fake_agent = FakeSubagentAgent.instances[-1]
        brief = fake_agent.run_calls[-1]["user_instruction"]
        self.assertIn("Paper reviewer independent-review mandate", brief)
        self.assertIn("identify the review target and user-requested concerns only", brief)
        self.assertIn("not evidence that the paper is fixed", brief)
        self.assertIn("author-side claims to verify", brief)
        self.assertIn("objects under review, not evidence of correctness", brief)
        self.assertIn("strongest objections a critical method/theory reviewer", brief)
        self.assertIn("Parent-provided success criteria to verify, not to trust", brief)
        self.assertIn("Parent context notes to verify, not to trust", brief)
        self.assertIn("Use the paper-reviewer system prompt as the controlling standard", brief)

    def test_spawned_proposal_evaluator_result_is_applied_to_target_proposal(self) -> None:
        planner_tools = self._planner_tools()
        self.state["active_algorithm_context"] = {
            "algorithm_id": "algo_review",
            "proposal_id": "proposal_123",
        }
        applied = {}

        def fake_run_sync(**kwargs):  # noqa: ANN003
            return {
                "subagent_id": "session-test:subagent:proposal-evaluator",
                "status": "completed",
                "summary": "Proposal is theoretically sound.",
                "findings": ["Exact-fit recovery argument is coherent."],
                "artifact_refs": [],
                "proposed_state_updates": {
                    "proposal_review": {
                        "decision": "approve",
                        "reviewer_feedback": "Approved by isolated evaluator.",
                        "implementation_risks": ["Watch UOT memory use."],
                        "risk_assessment": "## UOT memory\n\nInspect solver memory.",
                        "confidence": 0.85,
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
            applied.update(
                {
                    "algorithm_id": algorithm_id,
                    "proposal_id": proposal_id,
                    "decision": decision,
                    "reviewer_feedback": reviewer_feedback,
                    "implementation_risks": list(implementation_risks or []),
                    "risk_assessment": risk_assessment,
                }
            )
            return "✅ Proposal review recorded"

        assert planner_tools.subagent_manager is not None
        planner_tools.subagent_manager.run_sync = fake_run_sync  # type: ignore[method-assign]
        planner_tools.review_algorithm_proposal = fake_review_algorithm_proposal  # type: ignore[method-assign]

        result = planner_tools.spawn_subagent(
            subagent_type="proposal_evaluator",
            task="Review algo_review proposal.",
            success_criteria=["Return approve/revise/reject."],
        )

        self.assertEqual(applied["algorithm_id"], "algo_review")
        self.assertEqual(applied["proposal_id"], "proposal_123")
        self.assertEqual(applied["decision"], "approve")
        self.assertIn("Approved by isolated evaluator", applied["reviewer_feedback"])
        self.assertEqual(applied["implementation_risks"], ["Watch UOT memory use."])
        review_application = result["applied_state_updates"]["proposal_review"]
        self.assertTrue(review_application["applied"])
        self.assertEqual(review_application["algorithm_id"], "algo_review")

    def test_spawned_proposal_evaluator_retries_when_structured_risk_is_missing(self) -> None:
        planner_tools = self._planner_tools()
        self.state["active_algorithm_context"] = {
            "algorithm_id": "algo_review",
            "proposal_id": "proposal_123",
        }
        calls = []
        applied = {}

        def fake_run_sync(**kwargs):  # noqa: ANN003
            calls.append(kwargs)
            if len(calls) == 1:
                return {
                    "subagent_id": "session-test:subagent:proposal-evaluator-1",
                    "status": "completed",
                    "summary": "Approved but missing risk fields.",
                    "findings": [],
                    "artifact_refs": [],
                    "proposed_state_updates": {
                        "proposal_review": {
                            "decision": "approve",
                            "reviewer_feedback": "Approved.",
                            "confidence": 0.8,
                        }
                    },
                    "needs_input_question": "",
                }
            return {
                "subagent_id": "session-test:subagent:proposal-evaluator-2",
                "status": "completed",
                "summary": "Approved with risks.",
                "findings": [],
                "artifact_refs": [],
                "proposed_state_updates": {
                    "proposal_review": {
                        "decision": "approve",
                        "reviewer_feedback": "Approved.",
                        "implementation_risks": ["Risk one."],
                        "risk_assessment": "## Risk one\n\nDiagnostic.",
                        "confidence": 0.8,
                    }
                },
                "needs_input_question": "",
            }

        def fake_review_algorithm_proposal(**kwargs):  # noqa: ANN003
            applied.update(kwargs)
            return "✅ Proposal review recorded"

        assert planner_tools.subagent_manager is not None
        planner_tools.subagent_manager.run_sync = fake_run_sync  # type: ignore[method-assign]
        planner_tools.review_algorithm_proposal = fake_review_algorithm_proposal  # type: ignore[method-assign]

        result = planner_tools.spawn_subagent(
            subagent_type="proposal_evaluator",
            task="Review algo_review proposal.",
            success_criteria=["Return approve/revise/reject."],
        )

        self.assertEqual(len(calls), 2)
        self.assertIn("STRUCTURED OUTPUT CONTRACT VIOLATION", calls[1]["context_notes"])
        self.assertIn("Previous invalid subagent result to reuse", calls[1]["context_notes"])
        self.assertIn("Approved but missing risk fields.", calls[1]["context_notes"])
        self.assertIn("Reuse the prior review analysis", calls[1]["success_criteria"][-1])
        self.assertEqual(applied["implementation_risks"], ["Risk one."])
        self.assertTrue(result["applied_state_updates"]["proposal_review"]["applied"])

    def test_spawned_implementation_evaluator_result_is_recorded_for_current_fingerprint(self) -> None:
        planner_tools = self._planner_tools()
        self.state["active_algorithm_context"] = {
            "algorithm_id": "algo_impl",
            "proposal_id": "proposal_456",
        }
        recorded = {}

        def fake_run_sync(**kwargs):  # noqa: ANN003
            return {
                "subagent_id": "session-test:subagent:implementation-evaluator",
                "status": "completed",
                "summary": "Implementation matches the approved proposal.",
                "findings": ["All proposal steps are mapped."],
                "artifact_refs": [],
                "proposed_state_updates": {
                    "implementation_review": {
                        "decision": "approve",
                        "reviewer_feedback": "Implementation is aligned.",
                        "blocking_issues": [],
                        "advisory_risks": ["Monitor memory use."],
                        "efficiency_recommendations": ["Use minibatch OT for large datasets."],
                        "generalization_shortcut_risks": [],
                    }
                },
                "needs_input_question": "",
            }

        def fake_fingerprint(algorithm_id, *, proposal_id="", campaign=None):  # noqa: ANN001, ANN003
            del campaign
            return {
                "requires_review": True,
                "review_hash": "hash-impl",
                "review_hash_scope": "proposal_semantics",
                "source_paths": ["/tmp/algo_impl/PROPOSAL.md", "/tmp/algo_impl/algorithm.py"],
                "payload": {
                    "implementation_review_policy_version": 7,
                    "proposal": {
                        "proposal_id": proposal_id,
                    },
                },
            }

        def fake_record_implementation_review(algorithm_id, **kwargs):  # noqa: ANN003
            recorded.update({"algorithm_id": algorithm_id, **kwargs})
            return {
                "algorithm_id": algorithm_id,
                "proposal_id": kwargs["proposal_id"],
                "review_hash": kwargs["review_hash"],
                "status": "approved" if kwargs["decision"] == "approve" else "blocked",
                "decision": kwargs["decision"],
            }

        assert planner_tools.subagent_manager is not None
        planner_tools.subagent_manager.run_sync = fake_run_sync  # type: ignore[method-assign]
        planner_tools._build_implementation_review_fingerprint = fake_fingerprint  # type: ignore[method-assign]
        planner_tools.planner_file_tools.record_implementation_review = fake_record_implementation_review  # type: ignore[method-assign]

        result = planner_tools.spawn_subagent(
            subagent_type="implementation_evaluator",
            task="Review algo_impl implementation.",
            success_criteria=["Return approve/revise/reject."],
        )

        self.assertEqual(recorded["algorithm_id"], "algo_impl")
        self.assertEqual(recorded["proposal_id"], "proposal_456")
        self.assertEqual(recorded["review_hash"], "hash-impl")
        self.assertEqual(recorded["review_hash_scope"], "proposal_semantics")
        self.assertEqual(recorded["implementation_review_policy_version"], 7)
        self.assertEqual(recorded["decision"], "approve")
        self.assertEqual(recorded["advisory_risks"], ["Monitor memory use."])
        self.assertEqual(recorded["efficiency_recommendations"], ["Use minibatch OT for large datasets."])
        review_application = result["applied_state_updates"]["implementation_review"]
        self.assertTrue(review_application["applied"])
        self.assertEqual(review_application["status"], "approved")

    def test_spawned_implementation_evaluator_retries_when_list_fields_are_missing(self) -> None:
        planner_tools = self._planner_tools()
        self.state["active_algorithm_context"] = {
            "algorithm_id": "algo_impl",
            "proposal_id": "proposal_456",
        }
        calls = []
        recorded = {}

        def fake_run_sync(**kwargs):  # noqa: ANN003
            calls.append(kwargs)
            if len(calls) == 1:
                return {
                    "subagent_id": "session-test:subagent:implementation-evaluator-1",
                    "status": "completed",
                    "summary": "Approved but missing structured lists.",
                    "findings": [],
                    "artifact_refs": [],
                    "proposed_state_updates": {
                        "implementation_review": {
                            "decision": "approve",
                            "reviewer_feedback": "Aligned.",
                        }
                    },
                    "needs_input_question": "",
                }
            return {
                "subagent_id": "session-test:subagent:implementation-evaluator-2",
                "status": "completed",
                "summary": "Approved with required lists.",
                "findings": [],
                "artifact_refs": [],
                "proposed_state_updates": {
                    "implementation_review": {
                        "decision": "approve",
                        "reviewer_feedback": "Aligned.",
                        "blocking_issues": [],
                        "advisory_risks": [],
                        "efficiency_recommendations": [],
                        "generalization_shortcut_risks": [],
                    }
                },
                "needs_input_question": "",
            }

        def fake_fingerprint(algorithm_id, *, proposal_id="", campaign=None):  # noqa: ANN001, ANN003
            del campaign
            return {
                "requires_review": True,
                "review_hash": "hash-impl",
                "review_hash_scope": "proposal_semantics",
                "source_paths": [],
                "payload": {
                    "implementation_review_policy_version": 7,
                    "proposal": {"proposal_id": proposal_id},
                },
            }

        def fake_record_implementation_review(algorithm_id, **kwargs):  # noqa: ANN003
            recorded.update({"algorithm_id": algorithm_id, **kwargs})
            return {
                "algorithm_id": algorithm_id,
                "proposal_id": kwargs["proposal_id"],
                "review_hash": kwargs["review_hash"],
                "status": "approved",
                "decision": kwargs["decision"],
            }

        assert planner_tools.subagent_manager is not None
        planner_tools.subagent_manager.run_sync = fake_run_sync  # type: ignore[method-assign]
        planner_tools._build_implementation_review_fingerprint = fake_fingerprint  # type: ignore[method-assign]
        planner_tools.planner_file_tools.record_implementation_review = fake_record_implementation_review  # type: ignore[method-assign]

        result = planner_tools.spawn_subagent(
            subagent_type="implementation_evaluator",
            task="Review algo_impl implementation.",
            success_criteria=["Return approve/revise/reject."],
        )

        self.assertEqual(len(calls), 2)
        self.assertIn("STRUCTURED OUTPUT CONTRACT VIOLATION", calls[1]["context_notes"])
        self.assertIn("Previous invalid subagent result to reuse", calls[1]["context_notes"])
        self.assertEqual(recorded["decision"], "approve")
        self.assertTrue(result["applied_state_updates"]["implementation_review"]["applied"])

    def test_spawned_inference_evaluator_result_is_recorded_for_current_fingerprint(self) -> None:
        planner_tools = self._planner_tools()
        self.state["active_algorithm_context"] = {
            "algorithm_id": "algo_infer",
            "proposal_id": "proposal_789",
        }
        recorded = {}

        def fake_run_sync(**kwargs):  # noqa: ANN003
            return {
                "subagent_id": "session-test:subagent:inference-evaluator",
                "status": "completed",
                "summary": "Inference code uses only t0 context.",
                "findings": ["Trajectory comes from simulate_trajectory."],
                "artifact_refs": [],
                "proposed_state_updates": {
                    "inference_review": {
                        "decision": "approve",
                        "reviewer_feedback": "Inference boundary is valid.",
                        "blocking_issues": [],
                        "advisory_risks": ["Keep claim metrics read-only."],
                    }
                },
                "needs_input_question": "",
            }

        def fake_fingerprint(algorithm_id, *, proposal_id=""):  # noqa: ANN001
            return {
                "ok": True,
                "algorithm_id": algorithm_id,
                "proposal_id": proposal_id,
                "campaign_id": "campaign_1",
                "purpose": "campaign",
                "fingerprint": {
                    "requires_review": True,
                    "review_hash": "hash-infer",
                },
                "source_paths": ["/tmp/algo_infer/algorithm.py"],
            }

        def fake_record_inference_review(algorithm_id, **kwargs):  # noqa: ANN003
            recorded.update({"algorithm_id": algorithm_id, **kwargs})
            return {
                "algorithm_id": algorithm_id,
                "review_hash": kwargs["review_hash"],
                "status": "approved" if kwargs["decision"] == "approve" else "blocked",
                "decision": kwargs["decision"],
            }

        assert planner_tools.subagent_manager is not None
        planner_tools.subagent_manager.run_sync = fake_run_sync  # type: ignore[method-assign]
        planner_tools._spawned_inference_review_fingerprint = fake_fingerprint  # type: ignore[method-assign]
        planner_tools.planner_file_tools.record_inference_review = fake_record_inference_review  # type: ignore[method-assign]

        result = planner_tools.spawn_subagent(
            subagent_type="inference_evaluator",
            task="Review algo_infer inference.",
            success_criteria=["Return approve/revise/reject."],
        )

        self.assertEqual(recorded["algorithm_id"], "algo_infer")
        self.assertEqual(recorded["review_hash"], "hash-infer")
        self.assertEqual(recorded["decision"], "approve")
        self.assertEqual(recorded["advisory_risks"], ["Keep claim metrics read-only."])
        review_application = result["applied_state_updates"]["inference_review"]
        self.assertTrue(review_application["applied"])
        self.assertEqual(review_application["status"], "approved")
        self.assertEqual(review_application["purpose"], "campaign")

    def test_spawned_inference_evaluator_retries_when_list_fields_are_missing(self) -> None:
        planner_tools = self._planner_tools()
        self.state["active_algorithm_context"] = {
            "algorithm_id": "algo_infer",
            "proposal_id": "proposal_789",
        }
        calls = []
        recorded = {}

        def fake_run_sync(**kwargs):  # noqa: ANN003
            calls.append(kwargs)
            if len(calls) == 1:
                return {
                    "subagent_id": "session-test:subagent:inference-evaluator-1",
                    "status": "completed",
                    "summary": "Approved but missing lists.",
                    "findings": [],
                    "artifact_refs": [],
                    "proposed_state_updates": {
                        "inference_review": {
                            "decision": "approve",
                            "reviewer_feedback": "Valid.",
                        }
                    },
                    "needs_input_question": "",
                }
            return {
                "subagent_id": "session-test:subagent:inference-evaluator-2",
                "status": "completed",
                "summary": "Approved with required lists.",
                "findings": [],
                "artifact_refs": [],
                "proposed_state_updates": {
                    "inference_review": {
                        "decision": "approve",
                        "reviewer_feedback": "Valid.",
                        "blocking_issues": [],
                        "advisory_risks": [],
                    }
                },
                "needs_input_question": "",
            }

        def fake_fingerprint(algorithm_id, *, proposal_id=""):  # noqa: ANN001
            return {
                "ok": True,
                "algorithm_id": algorithm_id,
                "proposal_id": proposal_id,
                "purpose": "training",
                "fingerprint": {
                    "requires_review": True,
                    "review_hash": "hash-infer",
                },
                "source_paths": [],
            }

        def fake_record_inference_review(algorithm_id, **kwargs):  # noqa: ANN003
            recorded.update({"algorithm_id": algorithm_id, **kwargs})
            return {
                "algorithm_id": algorithm_id,
                "review_hash": kwargs["review_hash"],
                "status": "approved",
                "decision": kwargs["decision"],
            }

        assert planner_tools.subagent_manager is not None
        planner_tools.subagent_manager.run_sync = fake_run_sync  # type: ignore[method-assign]
        planner_tools._spawned_inference_review_fingerprint = fake_fingerprint  # type: ignore[method-assign]
        planner_tools.planner_file_tools.record_inference_review = fake_record_inference_review  # type: ignore[method-assign]

        result = planner_tools.spawn_subagent(
            subagent_type="inference_evaluator",
            task="Review algo_infer inference.",
            success_criteria=["Return approve/revise/reject."],
        )

        self.assertEqual(len(calls), 2)
        self.assertIn("STRUCTURED OUTPUT CONTRACT VIOLATION", calls[1]["context_notes"])
        self.assertIn("Previous invalid subagent result to reuse", calls[1]["context_notes"])
        self.assertEqual(recorded["decision"], "approve")
        self.assertTrue(result["applied_state_updates"]["inference_review"]["applied"])

    def test_subagent_prompt_includes_project_context_and_skill_catalog_only(self) -> None:
        self.state["task_profile"] = {
            "current_stage": "training",
            "primary_goal": "analysis",
            "facets": {"analysis": True, "reproduction": False},
            "change_axes": {"solver": True, "evaluation": False},
            "risk_flags": {},
            "notes": "Focus on training artifacts only.",
        }
        self.state["planner_loaded_skills"] = [
            {
                "name": "secret-skill",
                "content": "LOADED_SKILL_SECRET",
            }
        ]

        prompt = PromptBuilder(self.state).build(
            agent_role="subagent",
            agent_id="session-test:subagent:1",
            parent_agent_id="planner",
            subagent_type="general",
            tool_policy=default_subagent_tool_policy("general"),
        )

        self.assertIn("## Project Context", prompt)
        self.assertIn("workflow_phase: training", prompt)
        self.assertIn(f"output_dir: {self.tmpdir.name}", prompt)
        self.assertIn("active_change_axes: solver", prompt)
        self.assertIn("## Skills", prompt)
        self.assertIn("discovery metadata only", prompt)
        self.assertNotIn("LOADED_SKILL_SECRET", prompt)

    def test_proposal_evaluator_prompt_and_tool_surface_are_specialized(self) -> None:
        evaluator_policy = default_subagent_tool_policy("proposal_evaluator")
        evaluator_tools = SingleAgentTools(
            DummyLLM(),
            dict(self.state),
            agent_role="subagent",
            agent_id="session-test:subagent:evaluator",
            parent_agent_id="planner",
            subagent_type="proposal_evaluator",
            tool_policy=evaluator_policy,
        )
        tool_names = {tool.name for tool in evaluator_tools.get_tools()}
        self.assertIn(SUBMIT_PROPOSAL_REVIEW_TOOL_NAME, tool_names)
        self.assertIn("list_skills", tool_names)
        self.assertNotIn(SUBMIT_SUBAGENT_RESULT_TOOL_NAME, tool_names)
        self.assertNotIn("snapshot_active_algorithm_workspace", tool_names)
        self.assertNotIn("review_algorithm_proposal", tool_names)

        prompt = PromptBuilder(self.state).build(
            agent_role="subagent",
            agent_id="session-test:subagent:evaluator",
            parent_agent_id="planner",
            subagent_type="proposal_evaluator",
            tool_policy=evaluator_policy,
        )
        self.assertIn("strict mathematical reviewer", prompt)
        self.assertIn("submit_proposal_review(...)", prompt)
        self.assertIn("weighted particle distributions", prompt)
        self.assertIn("Implementation risks are advisory", prompt)
        self.assertIn("Do not spend the implementation-risk section restating theoretical caveats", prompt)
        self.assertIn("conceptual quality", prompt)
        self.assertIn("Why not existing builtins", prompt)
        self.assertIn("builtin-adjacent", prompt)
        self.assertIn("simulation-free", prompt)
        self.assertIn("one-piece", prompt)
        self.assertIn("meaningful mass/growth mechanism", prompt)
        self.assertIn("time-only count-ratio clocks", prompt)
        self.assertIn("biological meaning", prompt)
        self.assertIn("improvement routes", prompt)

    def test_idea_evaluator_prompt_and_tool_surface_are_specialized(self) -> None:
        evaluator_policy = default_subagent_tool_policy("idea_evaluator")
        evaluator_tools = SingleAgentTools(
            DummyLLM(),
            dict(self.state),
            agent_role="subagent",
            agent_id="session-test:subagent:idea-evaluator",
            parent_agent_id="planner",
            subagent_type="idea_evaluator",
            tool_policy=evaluator_policy,
        )
        tool_names = {tool.name for tool in evaluator_tools.get_tools()}
        self.assertIn(SUBMIT_RESEARCH_IDEA_REVIEW_TOOL_NAME, tool_names)
        self.assertIn("list_skills", tool_names)
        self.assertNotIn(SUBMIT_SUBAGENT_RESULT_TOOL_NAME, tool_names)
        self.assertNotIn("review_research_idea", tool_names)

        prompt = PromptBuilder(self.state).build(
            agent_role="subagent",
            agent_id="session-test:subagent:idea-evaluator",
            parent_agent_id="planner",
            subagent_type="idea_evaluator",
            tool_policy=evaluator_policy,
        )
        self.assertIn("strict reviewer for one persistent research idea", prompt)
        self.assertIn("submit_research_idea_review(...)", prompt)
        self.assertIn("continuous generative or dynamical modeling", prompt)
        self.assertIn("traditional pseudotime, clustering, graph abstraction", prompt)

    def test_implementation_evaluator_prompt_and_tool_surface_are_specialized(self) -> None:
        evaluator_policy = default_subagent_tool_policy("implementation_evaluator")
        evaluator_tools = SingleAgentTools(
            DummyLLM(),
            dict(self.state),
            agent_role="subagent",
            agent_id="session-test:subagent:implementation-evaluator",
            parent_agent_id="planner",
            subagent_type="implementation_evaluator",
            tool_policy=evaluator_policy,
        )
        tool_names = {tool.name for tool in evaluator_tools.get_tools()}
        self.assertIn(SUBMIT_IMPLEMENTATION_REVIEW_TOOL_NAME, tool_names)
        self.assertIn("list_skills", tool_names)
        self.assertNotIn(SUBMIT_SUBAGENT_RESULT_TOOL_NAME, tool_names)
        self.assertNotIn("run_campaign_trial", tool_names)

        submit_result = evaluator_tools.submit_implementation_review(
            decision="approve",
            summary="Implementation matches the approved proposal.",
            reviewer_feedback="No proposal-required mechanism is omitted.",
            findings=["P1 maps to algorithm.py:10."],
            advisory_risks=["Monitor numerical drift."],
            efficiency_recommendations=["Batch large pairwise computations."],
            generalization_shortcut_risks=["No warm-start shortcut found."],
            risk_assessment="### Numerical drift\n\nTrack rollout W1 by time gap.",
        )
        self.assertEqual(submit_result["decision"], "approve")
        submitted = evaluator_tools.peek_subagent_result(copy_result=True)
        self.assertEqual(submitted["status"], "completed")
        review = submitted["proposed_state_updates"]["implementation_review"]
        self.assertEqual(review["decision"], "approve")
        self.assertEqual(review["advisory_risks"], ["Monitor numerical drift."])
        self.assertEqual(review["efficiency_recommendations"], ["Batch large pairwise computations."])
        self.assertIn("Numerical drift", review["risk_assessment"])

        prompt = PromptBuilder(self.state).build(
            agent_role="subagent",
            agent_id="session-test:subagent:implementation-evaluator",
            parent_agent_id="planner",
            subagent_type="implementation_evaluator",
            tool_policy=evaluator_policy,
        )
        self.assertIn("strict read-only reviewer", prompt)
        self.assertIn("submit_implementation_review(...)", prompt)
        self.assertIn("IMPLEMENTATION_MAP.md", prompt)
        self.assertIn("Scalability Blockers", prompt)
        self.assertIn("full dense OT/UOT", prompt)
        self.assertIn("BalancedOTCouplingStrategy", prompt)
        self.assertIn("risk_assessment", prompt)

    def test_paper_reviewer_prompt_and_tool_surface_are_specialized(self) -> None:
        self.assertIn("paper_reviewer", available_subagent_type_names())
        reviewer_policy = default_subagent_tool_policy("paper_reviewer")
        reviewer_tools = SingleAgentTools(
            DummyLLM(),
            dict(self.state),
            agent_role="subagent",
            agent_id="session-test:subagent:paper-reviewer",
            parent_agent_id="planner",
            subagent_type="paper_reviewer",
            tool_policy=reviewer_policy,
        )
        tool_names = {tool.name for tool in reviewer_tools.get_tools()}
        self.assertIn(SUBMIT_SUBAGENT_RESULT_TOOL_NAME, tool_names)
        self.assertIn("read_file", tool_names)
        self.assertIn("grep_files", tool_names)
        self.assertIn("list_skills", tool_names)
        self.assertIn("query_campaign_baseline_metrics", tool_names)
        self.assertNotIn(SPAWN_SUBAGENT_TOOL_NAME, tool_names)
        self.assertNotIn("create_workspace_file", tool_names)
        self.assertNotIn("create_algorithm_workspace_artifact", tool_names)
        self.assertNotIn("replace_workspace_file", tool_names)
        self.assertNotIn("run_training", tool_names)

        submit_result = reviewer_tools.submit_subagent_result(
            status="completed",
            summary="Verdict: approve.",
            findings=["Content, theory, and structure pass; bibliography issues are advisory."],
            proposed_state_updates={
                "paper_review": {
                    "decision": "approve",
                    "reviewer_feedback": "Paper passes the completion gate.",
                    "blocking_issues": [],
                    "required_revisions": [],
                    "advisory_risks": [],
                    "citation_issues": [],
                    "scorecard": {
                        "content": "pass",
                        "theory/math": "pass",
                        "figures": "pass",
                        "citation sanity": "advisory pass",
                        "style": "pass",
                        "structure": "pass",
                        "compile": "pass",
                    },
                    "confidence": 0.86,
                }
            },
        )
        self.assertEqual(submit_result["status"], "completed")
        submitted = reviewer_tools.peek_subagent_result(copy_result=True)
        review = submitted["proposed_state_updates"]["paper_review"]
        self.assertEqual(review["decision"], "approve")
        self.assertEqual(review["citation_issues"], [])

        prompt = PromptBuilder(self.state).build(
            agent_role="subagent",
            agent_id="session-test:subagent:paper-reviewer",
            parent_agent_id="planner",
            subagent_type="paper_reviewer",
            tool_policy=reviewer_policy,
        )
        self.assertIn("strict read-only reviewer for paper drafts", prompt)
        self.assertIn("citation_ledger.md", prompt)
        self.assertIn("Citation sanity, advisory by default", prompt)
        self.assertIn("The theory section must be more than a high-level sketch", prompt)
        self.assertIn("independent critical reviewer", prompt)
        self.assertIn("not the parent planner's completion", prompt)
        self.assertIn("Author-written ledgers and audits are objects under review", prompt)
        self.assertIn("strongest objections a skeptical method/theory", prompt)
        self.assertIn("locked-release downstream analysis", prompt)
        self.assertIn("gene_space_unavailable", prompt)
        self.assertIn("model-derived", prompt)
        self.assertIn("gene-space", prompt)
        self.assertIn("High-quality papers require both", prompt)
        self.assertIn("workflow report rather than a publishable paper", prompt)
        self.assertIn("publication-quality method name", prompt)
        self.assertIn("Metric definitions, validation thresholds", prompt)
        self.assertIn("editorial_brief.md", prompt)
        self.assertIn("editorial_scorecard", prompt)
        self.assertIn("proposed_state_updates.paper_review", prompt)

    def test_inference_evaluator_tool_surface_allows_read_only_data_contract_inspection(self) -> None:
        reviewer_policy = default_subagent_tool_policy("inference_evaluator")
        reviewer_tools = SingleAgentTools(
            DummyLLM(),
            dict(self.state),
            agent_role="subagent",
            agent_id="session-test:subagent:inference-reviewer",
            parent_agent_id="planner",
            subagent_type="inference_evaluator",
            tool_policy=reviewer_policy,
        )
        tool_names = {tool.name for tool in reviewer_tools.get_tools()}
        self.assertIn(SUBMIT_SUBAGENT_RESULT_TOOL_NAME, tool_names)
        self.assertIn("read_file", tool_names)
        self.assertIn("get_algorithm_benchmark_dataset", tool_names)
        self.assertIn("inspect_h5ad_contract", tool_names)
        self.assertNotIn("execute_python", tool_names)
        self.assertNotIn("load_or_switch_adata", tool_names)
        self.assertNotIn("persist_runtime_adata", tool_names)
        self.assertNotIn("run_training", tool_names)

        prompt = PromptBuilder(self.state).build(
            agent_role="subagent",
            agent_id="session-test:subagent:inference-reviewer",
            parent_agent_id="planner",
            subagent_type="inference_evaluator",
            tool_policy=reviewer_policy,
        )
        self.assertIn("scientific semantics", prompt)
        self.assertIn("self-serving score for the candidate", prompt)
        self.assertIn("control/ablation baselines", prompt)
        self.assertIn("does not reflect the scientific problem", prompt)

    def test_subagent_snapshots_restore_into_new_planner_tools(self) -> None:
        planner_tools = self._planner_tools()
        result = planner_tools.spawn_subagent(
            task="Collect a bounded result for snapshotting.",
            success_criteria=["Submit a structured result"],
        )

        subagent_id = result["subagent_id"]
        histories = planner_tools.collect_subagent_histories()
        snapshots = planner_tools.collect_subagent_snapshots()
        self.assertIn(subagent_id, histories)
        self.assertIn(subagent_id, snapshots)
        self.assertNotIn("chat_history", snapshots[subagent_id])
        self.assertEqual(
            snapshots[subagent_id]["submitted_result"]["summary"],
            "Subagent finished the bounded task.",
        )

        restored_state = ensure_runtime_v2_state(
            {
                "session_id": "session-test",
                "output_dir": self.tmpdir.name,
                "user_goal": {"raw_question": "fresh session"},
            }
        )
        restored_tools = SingleAgentTools(
            DummyLLM(),
            restored_state,
            agent_role="planner",
            agent_id="planner",
        )
        restored_tools.subagent_manager.agent_factory = FakeSubagentAgent

        diag = restored_tools.restore_from_snapshot(snapshots, agent_histories=histories)
        self.assertFalse(diag["degraded"])
        self.assertEqual(diag["restored"][subagent_id], "metadata_only")
        self.assertTrue(restored_state["subagent_registry"][subagent_id]["restored_from_snapshot"])
        self.assertEqual(restored_state["subagent_registry"][subagent_id]["resume_mode"], "metadata_only")
        self.assertEqual(restored_tools.subagent_manager._agents, {})
        self.assertIn(subagent_id, restored_tools.collect_subagent_histories())
        self.assertEqual(
            len(restored_tools.collect_subagent_histories()[subagent_id]),
            len(histories[subagent_id]),
        )

        restored_snapshots = restored_tools.collect_subagent_snapshots()
        self.assertIn(subagent_id, restored_snapshots)
        self.assertNotIn("chat_history", restored_snapshots[subagent_id])


if __name__ == "__main__":
    unittest.main()
