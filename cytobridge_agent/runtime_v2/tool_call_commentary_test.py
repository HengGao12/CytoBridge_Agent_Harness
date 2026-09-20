from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cytobridge_agent.runtime_v2.agent import CytoBridgeAgent
from cytobridge_agent.runtime_v2.state import ensure_runtime_v2_state
from cytobridge_agent.tools.turn_phase import strip_tool_call_transcript


class ToolCallingLLM:
    def bind_tools(self, tools):  # noqa: ANN001
        del tools
        return self

    def invoke(self, messages):  # noqa: ANN001
        del messages
        return AIMessage(
            content="I will inspect the file before deciding.",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"path": "/tmp/example.txt"},
                    "id": "call_read_file",
                    "type": "tool_call",
                }
            ],
        )


class PseudoToolTranscriptLLM:
    def bind_tools(self, tools):  # noqa: ANN001
        del tools
        return self

    def invoke(self, messages):  # noqa: ANN001
        del messages
        return AIMessage(
            content='phase=commentary\nto=functions.create_workspace_file\n{"path": "out/paper/main.tex"}',
            tool_calls=[
                {
                    "name": "create_workspace_file",
                    "args": {"path": "out/paper/main.tex", "content": "..."},
                    "id": "call_create_workspace_file",
                    "type": "tool_call",
                }
            ],
        )


class PseudoMultiToolTranscriptLLM:
    def bind_tools(self, tools):  # noqa: ANN001
        del tools
        return self

    def invoke(self, messages):  # noqa: ANN001
        del messages
        return AIMessage(
            content=(
                "phase=commentary\n"
                'to=multi_tool_use.parallel { "tool_uses": ['
                '{ "recipient_name": "functions.read_file", "parameters": {"file_path": "README.md"} }'
                "] }"
            ),
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": "README.md"},
                    "id": "call_read_file",
                    "type": "tool_call",
                }
            ],
        )


class PlannerReviewerSubmitToolLLM:
    def bind_tools(self, tools):  # noqa: ANN001
        del tools
        return self

    def invoke(self, messages):  # noqa: ANN001
        del messages
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_proposal_review",
                    "args": {"decision": "approve", "summary": "stale review"},
                    "id": "call_submit_review",
                    "type": "tool_call",
                }
            ],
        )


class ToolCallCommentaryTest(unittest.TestCase):
    def _skill_loader_patches(self, tmpdir: str):  # noqa: ANN202
        user_root = Path(tmpdir).resolve() / "skills"
        builtin_root = Path(tmpdir).resolve() / "builtin_skills"
        return (
            patch("cytobridge_agent.tools.skills_loader.get_cellcompass_skills_root", return_value=user_root),
            patch("cytobridge_agent.tools.skills_loader.get_builtin_skills_root", return_value=builtin_root),
            patch("cytobridge_agent.tools.skills_loader.ensure_cellcompass_skills_migrated", return_value=user_root),
        )

    def test_tool_call_response_text_is_visible_and_preserved_in_history(self) -> None:
        events = []

        def event_callback(event_type, payload):  # noqa: ANN001
            events.append((event_type, dict(payload or {})))

        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "tool-call-commentary",
                    "output_dir": tmpdir,
                }
            )
            patches = self._skill_loader_patches(tmpdir)
            with patches[0], patches[1], patches[2]:
                agent = CytoBridgeAgent(
                    ToolCallingLLM(),
                    state,
                    agent_role="planner",
                    agent_id="planner",
                    event_callback=event_callback,
                )

                result = agent.agent_node(
                    {
                        "messages": [HumanMessage(content="inspect")],
                        "commentary_retries": 0,
                        "turn_system_prompt": "",
                        "enable_multimodal": False,
                    }
                )

        thought_events = [payload for event_type, payload in events if event_type == "agent_thought"]
        self.assertEqual(len(thought_events), 1)
        self.assertEqual(thought_events[0]["source"], "tool_call_response_text")
        self.assertIn("I will inspect the file", thought_events[0]["content"])

        messages = result["messages"]
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0], AIMessage)
        self.assertEqual(messages[0].additional_kwargs.get("phase"), "commentary")
        self.assertIn("I will inspect the file", messages[0].content)
        self.assertEqual(messages[0].tool_calls[0]["name"], "read_file")

    def test_pseudo_tool_call_transcript_is_not_shown_as_commentary(self) -> None:
        events = []

        def event_callback(event_type, payload):  # noqa: ANN001
            events.append((event_type, dict(payload or {})))

        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "pseudo-tool-transcript",
                    "output_dir": tmpdir,
                }
            )
            patches = self._skill_loader_patches(tmpdir)
            with patches[0], patches[1], patches[2]:
                agent = CytoBridgeAgent(
                    PseudoToolTranscriptLLM(),
                    state,
                    agent_role="planner",
                    agent_id="planner",
                    event_callback=event_callback,
                )

                result = agent.agent_node(
                    {
                        "messages": [HumanMessage(content="write paper")],
                        "commentary_retries": 0,
                        "turn_system_prompt": "",
                        "enable_multimodal": False,
                    }
                )

        thought_events = [payload for event_type, payload in events if event_type == "agent_thought"]
        self.assertEqual(thought_events, [])

        messages = result["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, "")
        self.assertEqual(messages[0].additional_kwargs.get("phase"), "commentary")
        self.assertEqual(messages[0].tool_calls[0]["name"], "create_workspace_file")

    def test_pseudo_multi_tool_transcript_is_not_shown_as_commentary(self) -> None:
        events = []

        def event_callback(event_type, payload):  # noqa: ANN001
            events.append((event_type, dict(payload or {})))

        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "pseudo-multi-tool-transcript",
                    "output_dir": tmpdir,
                }
            )
            patches = self._skill_loader_patches(tmpdir)
            with patches[0], patches[1], patches[2]:
                agent = CytoBridgeAgent(
                    PseudoMultiToolTranscriptLLM(),
                    state,
                    agent_role="planner",
                    agent_id="planner",
                    event_callback=event_callback,
                )

                result = agent.agent_node(
                    {
                        "messages": [HumanMessage(content="read docs")],
                        "commentary_retries": 0,
                        "turn_system_prompt": "",
                        "enable_multimodal": False,
                    }
                )

        thought_events = [payload for event_type, payload in events if event_type == "agent_thought"]
        self.assertEqual(thought_events, [])

        messages = result["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, "")
        self.assertEqual(messages[0].additional_kwargs.get("phase"), "commentary")
        self.assertEqual(messages[0].tool_calls[0]["name"], "read_file")

    def test_final_answer_after_pseudo_tool_transcript_is_preserved(self) -> None:
        text = (
            'to=functions.create_workspace_file\n{"path": "out/paper/main.tex"}\n'
            "phase=final_answer\n"
            "main.tex 已经写好。"
        )

        self.assertEqual(strip_tool_call_transcript(text), "main.tex 已经写好。")

    def test_final_answer_after_pseudo_multi_tool_transcript_is_preserved(self) -> None:
        text = (
            'to=multi_tool_use.parallel {"tool_uses": []}\n'
            "phase=completed\n"
            "已完成文档读取。"
        )

        self.assertEqual(strip_tool_call_transcript(text), "已完成文档读取。")

    def test_planner_reviewer_submit_tool_call_is_guarded(self) -> None:
        events = []

        def event_callback(event_type, payload):  # noqa: ANN001
            events.append((event_type, dict(payload or {})))

        with tempfile.TemporaryDirectory() as tmpdir:
            state = ensure_runtime_v2_state(
                {
                    "session_id": "planner-review-submit-guard",
                    "output_dir": tmpdir,
                }
            )
            patches = self._skill_loader_patches(tmpdir)
            with patches[0], patches[1], patches[2]:
                agent = CytoBridgeAgent(
                    PlannerReviewerSubmitToolLLM(),
                    state,
                    agent_role="planner",
                    agent_id="planner",
                    event_callback=event_callback,
                )

                result = agent.agent_node(
                    {
                        "messages": [HumanMessage(content="answer the current question")],
                        "commentary_retries": 0,
                        "turn_system_prompt": "",
                        "enable_multimodal": False,
                    }
                )

        guard_events = [payload for event_type, payload in events if event_type == "internal_subagent_tool_call_guarded"]
        self.assertEqual(len(guard_events), 1)
        self.assertEqual(guard_events[0]["tool_names"], ["submit_proposal_review"])
        self.assertFalse(getattr(result["messages"][-1], "tool_calls", None))
        self.assertEqual(result["messages"][-1].additional_kwargs.get("phase"), "commentary")

    def test_completed_answer_after_pseudo_tool_transcript_is_preserved(self) -> None:
        text = (
            'phase=commentary to=functions.commit_workflow_state\n{"phase": "reporting"}\n'
            "|« token stats »|\n"
            "phase=completed\n"
            "已完成 paper，并生成 main.pdf。"
        )

        self.assertEqual(strip_tool_call_transcript(text), "已完成 paper，并生成 main.pdf。")


if __name__ == "__main__":
    unittest.main()
