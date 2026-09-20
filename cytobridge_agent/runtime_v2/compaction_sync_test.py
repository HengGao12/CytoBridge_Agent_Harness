from __future__ import annotations

import os
import tempfile
import threading
import unittest

from langchain_core.load import dumpd, load
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from cytobridge_agent.conversation_store import ConversationStore
from cytobridge_agent.interactive import InteractiveSession
from cytobridge_agent.runtime_v2 import agent as agent_mod
from cytobridge_agent.runtime_v2.middleware import RuntimeMiddleware
from cytobridge_agent.runtime_v2.state import ensure_runtime_v2_state
from cytobridge_agent.tools import context_compaction as compact_mod
from cytobridge_agent.tools.context_compaction import (
    NO_TOOLS_PREAMBLE,
    NO_TOOLS_TRAILER,
    compact_history,
    estimate_tokens,
    format_compact_summary,
    get_compact_prompt,
    get_context_policy,
    is_compact_boundary_message,
    is_compact_rehydrate_message,
    is_compact_summary_message,
    _build_isolated_compaction_llm,
    microcompact_history,
    project_compacted_history,
)


class _FakeLLM:
    def bind_tools(self, tools):
        return self


class _CopyTrackingLLM:
    def __init__(self, session_id: str = "source") -> None:
        self.session_id = session_id
        self.model_copy_calls = []

    def model_copy(self, *, deep: bool = False):
        self.model_copy_calls.append({"deep": deep})
        return _CopyTrackingLLM(session_id=self.session_id)


class CompactionSyncTest(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_agent_invoke = agent_mod.invoke_with_retry
        self._orig_compact_invoke = compact_mod.invoke_with_retry
        self._counter = 0

        def _fake_agent_invoke(llm, messages, logger, label):
            self._counter += 1
            return AIMessage(content=f"phase=final_answer\nresponse {self._counter}")

        agent_mod.invoke_with_retry = _fake_agent_invoke

    def tearDown(self) -> None:
        agent_mod.invoke_with_retry = self._orig_agent_invoke
        compact_mod.invoke_with_retry = self._orig_compact_invoke

    def test_compact_prompt_uses_claude_shape_with_cytobridge_adaptation(self) -> None:
        prompt = get_compact_prompt()
        self.assertIn(NO_TOOLS_PREAMBLE.strip(), prompt)
        self.assertIn(NO_TOOLS_TRAILER.strip(), prompt)
        self.assertIn("<analysis>", prompt)
        self.assertIn("<summary>", prompt)
        self.assertIn("Files, Artifacts, and State", prompt)
        self.assertIn("active workflow phase", prompt)
        self.assertIn("active data paths", prompt)

    def test_format_compact_summary_strips_analysis_and_keeps_summary(self) -> None:
        raw = "<analysis>scratch</analysis>\n<summary>\n1. Primary Request and Intent:\n- item\n</summary>"
        formatted = format_compact_summary(raw)
        self.assertNotIn("scratch", formatted)
        self.assertIn("Summary:", formatted)
        self.assertIn("1. Primary Request and Intent:", formatted)

    def test_compaction_llm_clone_uses_shallow_copy(self) -> None:
        llm = _CopyTrackingLLM(session_id="planner-session")
        clone = _build_isolated_compaction_llm(llm)

        self.assertIsNot(clone, llm)
        self.assertEqual(llm.model_copy_calls, [{"deep": False}])
        self.assertEqual(llm.session_id, "planner-session")
        self.assertTrue(str(clone.session_id).startswith("planner-session-compaction-"))

    def test_bad_context_policy_is_normalized_to_sane_defaults(self) -> None:
        state = ensure_runtime_v2_state(
            {
                "session_id": "bad-policy",
                "context_policy": {
                    "enabled": True,
                    "context_window": 50,
                    "trigger_ratio": 0.1,
                    "keep_last_turns": 2,
                    "max_tool_chars": 1500,
                    "microcompact_enabled": True,
                    "llm_compact_enabled": False,
                    "llm_compact_input_max_chars": 24000,
                },
            }
        )
        normalized_state_policy = dict(state.get("context_policy") or {})
        self.assertEqual(normalized_state_policy.get("context_window"), 200000)
        self.assertEqual(normalized_state_policy.get("context_window_cap"), 256000)
        self.assertEqual(normalized_state_policy.get("effective_context_window"), 200000)
        self.assertFalse(normalized_state_policy.get("allow_large_context_window"))
        self.assertEqual(normalized_state_policy.get("trigger_ratio"), 0.82)

        normalized_runtime_policy = get_context_policy(
            {
                "enabled": True,
                "context_window": 50,
                "trigger_ratio": 0.1,
                "keep_last_turns": 2,
                "max_tool_chars": 1500,
                "microcompact_enabled": True,
                "llm_compact_enabled": False,
                "llm_compact_input_max_chars": 24000,
            }
        )
        self.assertEqual(normalized_runtime_policy.get("context_window"), 200000)
        self.assertEqual(normalized_runtime_policy.get("context_window_cap"), 256000)
        self.assertEqual(normalized_runtime_policy.get("effective_context_window"), 200000)
        self.assertFalse(normalized_runtime_policy.get("allow_large_context_window"))
        self.assertEqual(normalized_runtime_policy.get("trigger_ratio"), 0.82)

    def test_large_context_window_is_capped_for_compaction_by_default(self) -> None:
        capped = get_context_policy(
            {
                "context_window": 1000000,
                "trigger_ratio": 0.82,
            }
        )
        self.assertEqual(capped["context_window"], 1000000)
        self.assertEqual(capped["context_window_cap"], 256000)
        self.assertEqual(capped["effective_context_window"], 256000)
        self.assertFalse(capped["allow_large_context_window"])

        uncapped = get_context_policy(
            {
                "context_window": 1000000,
                "trigger_ratio": 0.82,
                "allow_large_context_window": True,
            }
        )
        self.assertEqual(uncapped["context_window"], 1000000)
        self.assertEqual(uncapped["effective_context_window"], 1000000)
        self.assertTrue(uncapped["allow_large_context_window"])

    def test_large_snapshot_defaults_to_full_resume_not_event_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConversationStore(base_dir=tmpdir)
            old_limit = store._RESUME_SNAPSHOT_MAX_BYTES
            store._RESUME_SNAPSHOT_MAX_BYTES = 256
            try:
                session_id = "large-resume"
                store.save_conversation(
                    session_id,
                    metadata={"title": "Large Resume"},
                    messages=[{"role": "user", "content": "keep full state"}],
                    agent_state={"important_checkpoint_state": "x" * 2048},
                    agent_snapshots={"planner": {"langgraph_checkpoint": {"checkpoint": "present"}}},
                )
                store.append_event(session_id, "user_message", {"content": "event-only preview"})

                full = store.get_conversation(session_id)
                self.assertIsNotNone(full)
                assert full is not None
                self.assertEqual(full["agent_state"]["important_checkpoint_state"], "x" * 2048)
                self.assertEqual(
                    full["agent_snapshots"]["planner"]["langgraph_checkpoint"]["checkpoint"],
                    "present",
                )
                self.assertFalse(full["resume_diagnostics"]["degraded"])

                preview = store.get_conversation(session_id, degrade_large_snapshot=True)
                self.assertIsNotNone(preview)
                assert preview is not None
                self.assertTrue(preview["resume_diagnostics"]["degraded"])
                self.assertEqual(preview["agent_state"], {})
            finally:
                store._RESUME_SNAPSHOT_MAX_BYTES = old_limit

    def test_compact_history_summarizes_full_history_and_keeps_recent_user_messages(self) -> None:
        def _fake_compact_invoke(llm, messages, logger, label):
            return AIMessage(
                content=(
                    "<analysis>ignore me</analysis>\n"
                    "<summary>\n1. Primary Request and Intent:\n- keep theory context\n</summary>"
                )
            )

        compact_mod.invoke_with_retry = _fake_compact_invoke
        messages = [
            HumanMessage(content="u1"),
            AIMessage(content="a1"),
            HumanMessage(content="u2"),
            AIMessage(content="a2"),
            HumanMessage(content="u3"),
            AIMessage(content="a3"),
        ]
        compacted, info = compact_history(
            messages,
            keep_last_turns=1,
            llm=_FakeLLM(),
            llm_enabled=True,
            state={"workflow_phase": "training", "task_profile": {"current_stage": "training"}},
            force_full=True,
            microcompact_enabled=False,
        )
        updates = list(info.get("history_updates") or [])
        self.assertTrue(any(is_compact_boundary_message(msg) for msg in updates))
        self.assertTrue(any(is_compact_summary_message(msg) for msg in updates))
        self.assertTrue(any(is_compact_rehydrate_message(msg) for msg in updates))
        summary_msg = next(msg for msg in updates if is_compact_summary_message(msg))
        self.assertNotIn("<analysis>", summary_msg.content)
        self.assertIn("Summary:", summary_msg.content)
        self.assertIn("Recent messages are preserved verbatim.", summary_msg.content)
        self.assertEqual(info.get("mode"), "llm")
        self.assertEqual(info.get("preserved_tail_count"), 1)

        full_history = messages + updates
        projected, meta = project_compacted_history(full_history)
        self.assertTrue(meta.get("has_boundary"))
        projected_contents = [getattr(msg, "content", "") for msg in projected]
        self.assertIn("u3", projected_contents)
        self.assertNotIn("a3", projected_contents)
        self.assertEqual(meta.get("preserved_tail_count"), 1)
        self.assertLess(len(projected), len(full_history))

    def test_microcompact_trims_old_tool_payloads(self) -> None:
        messages = [
            SystemMessage(content="sys"),
            HumanMessage(content="older question"),
            ToolMessage(content="A" * 5000, tool_call_id="old-call"),
            HumanMessage(content="middle"),
            AIMessage(content="recent assistant"),
            HumanMessage(content="recent user"),
        ]
        compacted, info = microcompact_history(
            messages,
            keep_last_turns=1,
            max_tool_chars=200,
            preserve_leading_system=True,
        )
        self.assertEqual(info.get("mode"), "microcompact")
        self.assertGreater(int(info.get("tokens_saved", 0)), 0)
        old_tool = next(msg for msg in compacted if isinstance(msg, ToolMessage))
        self.assertIn("...[microcompacted", old_tool.content)

    def test_project_without_boundary_returns_full_history(self) -> None:
        messages = [HumanMessage(content="u1"), AIMessage(content="a1")]
        projected, meta = project_compacted_history(messages)
        self.assertFalse(meta.get("has_boundary"))
        self.assertEqual(projected, messages)

    def test_middleware_skips_counting_noop_compaction(self) -> None:
        state = ensure_runtime_v2_state(
            {
                "session_id": "noop-compact",
                "context_policy": {
                    "enabled": True,
                    "context_window": 1000,
                    "trigger_ratio": 0.5,
                    "keep_last_turns": 6,
                    "max_tool_chars": 200,
                    "microcompact_enabled": True,
                    "llm_compact_enabled": False,
                    "llm_compact_input_max_chars": 24000,
                },
            }
        )
        middleware = RuntimeMiddleware(state, _FakeLLM())
        prompt_messages = [
            SystemMessage(content="sys"),
            HumanMessage(content="u" * 1200),
            AIMessage(content="a" * 1200),
        ]
        prepared = middleware.prepare_messages(prompt_messages)
        self.assertNotIn("compaction_stats", state)
        self.assertEqual(prepared.history_updates, [])
        self.assertIn(prepared.compaction_meta.get("mode"), {None, "none"})

    def test_middleware_does_not_microcompact_before_full_compact_threshold(self) -> None:
        state = ensure_runtime_v2_state(
            {
                "session_id": "microcompact-first",
                "context_policy": {
                    "enabled": True,
                    "context_window": 100000,
                    "trigger_ratio": 0.9,
                    "keep_last_turns": 1,
                    "max_tool_chars": 120,
                    "microcompact_enabled": True,
                    "llm_compact_enabled": False,
                    "llm_compact_input_max_chars": 24000,
                },
            }
        )
        middleware = RuntimeMiddleware(state, _FakeLLM())
        prompt_messages = [
            SystemMessage(content="sys"),
            HumanMessage(content="use tool"),
            AIMessage(
                content="tool call",
                tool_calls=[{"id": "tc1", "name": "read_file", "args": {"path": "/tmp/a.txt"}}],
            ),
            ToolMessage(content="T" * 8000, tool_call_id="tc1"),
            HumanMessage(content="recent question"),
            AIMessage(content="recent answer"),
        ]

        self.assertLess(
            estimate_tokens(prompt_messages),
            int(state["context_policy"]["context_window"] * state["context_policy"]["trigger_ratio"]),
        )

        prepared = middleware.prepare_messages(prompt_messages)
        self.assertIn(prepared.compaction_meta.get("mode"), {None, "none"})
        self.assertEqual(prepared.history_updates, [])
        self.assertIsNone(prepared.persisted_history)
        self.assertNotIn("compaction_stats", state)

    def test_middleware_uses_provider_usage_total_for_compaction_pressure(self) -> None:
        state = ensure_runtime_v2_state(
            {
                "session_id": "usage-total-pressure",
                "context_policy": {
                    "enabled": True,
                    "context_window": 10000,
                    "trigger_ratio": 0.5,
                    "keep_last_turns": 1,
                    "max_tool_chars": 120,
                    "microcompact_enabled": True,
                    "llm_compact_enabled": False,
                    "llm_compact_input_max_chars": 24000,
                },
                "llm_context_usage": {
                    "total_tokens": 9000,
                    "prompt_tokens": 0,
                    "source": "usage_total_tokens",
                },
            }
        )
        middleware = RuntimeMiddleware(state, _FakeLLM())
        prompt_messages = [
            SystemMessage(content="sys"),
            HumanMessage(content="short"),
            AIMessage(content="short answer"),
        ]
        self.assertLess(estimate_tokens(prompt_messages), 5000)

        prepared = middleware.prepare_messages(prompt_messages)

        self.assertTrue(prepared.history_updates)
        self.assertEqual(prepared.compaction_meta.get("budget_token_source"), "usage_total_tokens")
        self.assertEqual(prepared.compaction_meta.get("before_tokens"), 9000)
        self.assertEqual(state["llm_context_usage"]["source"], "estimate_after_full_compact")
        self.assertLess(state["llm_context_usage"]["budget_tokens"], 9000)

    def test_middleware_uses_effective_context_window_cap_for_large_models(self) -> None:
        state = ensure_runtime_v2_state(
            {
                "session_id": "large-model-cap",
                "context_policy": {
                    "enabled": True,
                    "context_window": 1000000,
                    "trigger_ratio": 0.82,
                    "keep_last_turns": 1,
                    "max_tool_chars": 120,
                    "microcompact_enabled": True,
                    "llm_compact_enabled": False,
                    "llm_compact_input_max_chars": 24000,
                },
                "llm_context_usage": {
                    "total_tokens": 220000,
                    "source": "usage_total_tokens",
                },
            }
        )
        middleware = RuntimeMiddleware(state, _FakeLLM())
        prompt_messages = [
            SystemMessage(content="sys"),
            HumanMessage(content="short"),
            AIMessage(content="short answer"),
        ]

        prepared = middleware.prepare_messages(prompt_messages)

        self.assertTrue(prepared.history_updates)
        self.assertEqual(prepared.compaction_meta.get("context_window"), 1000000)
        self.assertEqual(prepared.compaction_meta.get("effective_context_window"), 256000)
        self.assertEqual(prepared.compaction_meta.get("context_window_cap"), 256000)
        self.assertFalse(prepared.compaction_meta.get("allow_large_context_window"))

    def test_middleware_can_explicitly_allow_large_context_window(self) -> None:
        state = ensure_runtime_v2_state(
            {
                "session_id": "large-model-allowed",
                "context_policy": {
                    "enabled": True,
                    "context_window": 1000000,
                    "allow_large_context_window": True,
                    "trigger_ratio": 0.82,
                    "keep_last_turns": 1,
                    "max_tool_chars": 120,
                    "microcompact_enabled": True,
                    "llm_compact_enabled": False,
                    "llm_compact_input_max_chars": 24000,
                },
                "llm_context_usage": {
                    "total_tokens": 220000,
                    "source": "usage_total_tokens",
                },
            }
        )
        middleware = RuntimeMiddleware(state, _FakeLLM())
        prompt_messages = [
            SystemMessage(content="sys"),
            HumanMessage(content="short"),
            AIMessage(content="short answer"),
        ]

        prepared = middleware.prepare_messages(prompt_messages)

        self.assertEqual(prepared.history_updates, [])
        self.assertIn(prepared.compaction_meta.get("mode"), {None, "none"})
        self.assertEqual(prepared.compaction_meta.get("context_window"), 1000000)
        self.assertEqual(prepared.compaction_meta.get("effective_context_window"), 1000000)
        self.assertTrue(prepared.compaction_meta.get("allow_large_context_window"))

    def test_microcompact_persists_rewritten_history_to_avoid_repeat_on_next_turn(self) -> None:
        state = ensure_runtime_v2_state(
            {
                "session_id": "persist-microcompact",
                "context_policy": {
                    "enabled": True,
                    "context_window": 3000,
                    "trigger_ratio": 0.5,
                    "keep_last_turns": 1,
                    "max_tool_chars": 120,
                    "microcompact_enabled": True,
                    "llm_compact_enabled": False,
                    "llm_compact_input_max_chars": 24000,
                },
            }
        )
        middleware = RuntimeMiddleware(state, _FakeLLM())
        messages = [
            SystemMessage(content="sys"),
            HumanMessage(content="use tool"),
            AIMessage(
                content="tool call",
                tool_calls=[{"id": "tc1", "name": "read_file", "args": {"path": "/tmp/a.txt"}}],
            ),
            ToolMessage(content="T" * 8000, tool_call_id="tc1"),
            HumanMessage(content="use second tool"),
            AIMessage(
                content="tool call 2",
                tool_calls=[{"id": "tc2", "name": "read_file", "args": {"path": "/tmp/b.txt"}}],
            ),
            ToolMessage(content="U" * 8000, tool_call_id="tc2"),
            HumanMessage(content="followup 2"),
            AIMessage(content="answer 2"),
        ]

        prepared_first = middleware.prepare_messages(messages)
        self.assertEqual(prepared_first.compaction_meta.get("mode"), "microcompact")
        self.assertEqual(prepared_first.history_updates, [])
        self.assertIsNotNone(prepared_first.persisted_history)
        self.assertEqual(int(state["compaction_stats"]["microcompact_count"]), 1)

        persisted = [SystemMessage(content="sys")] + list(prepared_first.persisted_history or [])
        prepared_second = middleware.prepare_messages(persisted)
        self.assertEqual(prepared_second.history_updates, [])
        self.assertIn(prepared_second.compaction_meta.get("mode"), {None, "none"})
        self.assertEqual(int(state["compaction_stats"]["microcompact_count"]), 1)

    def test_runtime_auto_microcompact_persists_to_checkpoint_and_avoids_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                session = InteractiveSession(
                    input_path=None,
                    user_goal={"raw_question": "runtime microcompact"},
                    llm=_FakeLLM(),
                )
                session.state["context_policy"] = {
                    "enabled": True,
                    "context_window": 6000,
                    "trigger_ratio": 0.5,
                    "keep_last_turns": 1,
                    "max_tool_chars": 120,
                    "microcompact_enabled": True,
                    "llm_compact_enabled": False,
                    "llm_compact_input_max_chars": 24000,
                }
                seed_history = [
                    HumanMessage(content="use tool"),
                    AIMessage(
                        content="tool call",
                        tool_calls=[{"id": "tc1", "name": "read_file", "args": {"path": "/tmp/a.txt"}}],
                    ),
                    ToolMessage(content="T" * 8000, tool_call_id="tc1"),
                    HumanMessage(content="use second tool"),
                    AIMessage(
                        content="tool call 2",
                        tool_calls=[{"id": "tc2", "name": "read_file", "args": {"path": "/tmp/b.txt"}}],
                    ),
                    ToolMessage(content="U" * 8000, tool_call_id="tc2"),
                    HumanMessage(content="followup 2"),
                    AIMessage(content="answer 2"),
                ]
                session.planner.replace_chat_history(seed_history, reset_checkpoint=True)

                reply = session.run_turn("continue after microcompact seed")
                self.assertIn("response", reply)
                self.assertEqual(int(session.state["compaction_stats"]["microcompact_count"]), 1)

                checkpoint_after_first = session.planner.checkpoint_messages()
                self.assertTrue(checkpoint_after_first)
                self.assertEqual(len(session.planner.chat_history), len(checkpoint_after_first))

                before_second = int(session.state["compaction_stats"]["microcompact_count"])
                second_reply = session.run_turn("follow up after persisted microcompact")
                self.assertIn("response", second_reply)
                self.assertEqual(int(session.state["compaction_stats"]["microcompact_count"]), before_second)
                self.assertEqual(
                    len(session.planner.chat_history),
                    len(session.planner.checkpoint_messages()),
                )
            finally:
                if previous_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = previous_home

    def test_full_compact_drops_below_threshold_to_avoid_immediate_recompact(self) -> None:
        state = ensure_runtime_v2_state(
            {
                "session_id": "repeat-compact",
                "workflow_phase": "training",
                "context_policy": {
                    "enabled": True,
                    "context_window": 4000,
                    "trigger_ratio": 0.5,
                    "keep_last_turns": 2,
                    "max_tool_chars": 200,
                    "microcompact_enabled": True,
                    "llm_compact_enabled": False,
                    "llm_compact_input_max_chars": 24000,
                },
            }
        )
        middleware = RuntimeMiddleware(state, _FakeLLM())
        prompt_messages = [SystemMessage(content="sys")]
        for idx in range(20):
            prompt_messages.append(HumanMessage(content=f"user {idx} " + ("x" * 300)))
            prompt_messages.append(AIMessage(content=f"assistant {idx} " + ("y" * 300)))

        prepared_first = middleware.prepare_messages(prompt_messages)
        self.assertGreater(len(prepared_first.history_updates), 0)
        self.assertIsNotNone(prepared_first.persisted_history)
        self.assertEqual(state["compaction_stats"]["count"], 1)

        full_history = list(prepared_first.persisted_history or [])
        prepared_second = middleware.prepare_messages([SystemMessage(content="sys")] + full_history)
        self.assertEqual(state["compaction_stats"]["count"], 1)
        self.assertEqual(prepared_second.history_updates, [])
        self.assertIsNone(prepared_second.persisted_history)
        self.assertLess(
            estimate_tokens(prepared_first.prompt_messages),
            int(state["context_policy"]["context_window"] * state["context_policy"]["trigger_ratio"]),
        )

    def test_runtime_auto_full_compact_persists_across_resume_without_immediate_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                session = InteractiveSession(
                    input_path=None,
                    user_goal={"raw_question": "runtime full compact"},
                    llm=_FakeLLM(),
                )
                session.state["context_policy"] = {
                    "enabled": True,
                    "context_window": 10000,
                    "trigger_ratio": 0.5,
                    "keep_last_turns": 2,
                    "max_tool_chars": 200,
                    "microcompact_enabled": True,
                    "llm_compact_enabled": False,
                    "llm_compact_input_max_chars": 24000,
                }

                for idx in range(12):
                    session.run_turn(f"turn {idx} " + ("x" * 1000))
                    if int((session.state.get("compaction_stats") or {}).get("count", 0)) >= 1:
                        break

                self.assertEqual(int(session.state["compaction_stats"]["count"]), 1)
                self.assertTrue(
                    any(is_compact_boundary_message(msg) for msg in session.planner.chat_history)
                )
                self.assertTrue(
                    any(is_compact_boundary_message(msg) for msg in session.planner.checkpoint_messages())
                )
                self.assertEqual(
                    len(session.planner.chat_history),
                    len(session.planner.checkpoint_messages()),
                )

                restored = InteractiveSession.from_checkpoint(session.session_id, _FakeLLM())
                self.assertTrue(
                    any(is_compact_boundary_message(msg) for msg in restored.planner.chat_history)
                )
                self.assertTrue(
                    any(is_compact_boundary_message(msg) for msg in restored.planner.checkpoint_messages())
                )
                before_second = int((restored.state.get("compaction_stats") or {}).get("count", 0))
                reply = restored.run_turn("post-resume followup")
                self.assertIn("response", reply)
                self.assertEqual(int(restored.state["compaction_stats"]["count"]), before_second)
                self.assertEqual(
                    len(restored.planner.chat_history),
                    len(restored.planner.checkpoint_messages()),
                )
            finally:
                if previous_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = previous_home

    def test_interactive_manual_compact_appends_boundary_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                session = InteractiveSession(
                    input_path=None,
                    user_goal={"raw_question": "debug compact"},
                    llm=_FakeLLM(),
                )
                session.state["context_policy"] = {
                    "enabled": True,
                    "context_window": 200000,
                    "trigger_ratio": 0.82,
                    "keep_last_turns": 2,
                    "max_tool_chars": 200,
                    "microcompact_enabled": True,
                    "llm_compact_enabled": False,
                    "llm_compact_input_max_chars": 24000,
                }
                for idx in range(8):
                    session.run_turn(f"turn {idx}")

                before_history_len = len(session.planner.chat_history)
                before_conv_len = len(session.conversation_history)
                compact_info = session.compact_context()
                after_history_len = len(session.planner.chat_history)

                self.assertGreater(compact_info["removed_messages"], 0)
                self.assertEqual(len(session.conversation_history), before_conv_len)
                self.assertLess(after_history_len, before_history_len)
                self.assertTrue(any(is_compact_boundary_message(msg) for msg in session.planner.chat_history))

                projected, meta = project_compacted_history(session.planner.chat_history)
                self.assertTrue(meta.get("has_boundary"))
                self.assertGreater(len(projected), len(session.planner.chat_history))
                self.assertLess(len(projected), before_history_len)

                reply = session.run_turn("post compact turn")
                self.assertIn("response", reply)
                self.assertEqual(
                    len(session.planner.chat_history),
                    len(session.planner.checkpoint_messages()),
                )
            finally:
                if previous_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = previous_home

    def test_interactive_manual_compact_normalizes_malformed_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                session = InteractiveSession(
                    input_path=None,
                    user_goal={"raw_question": "debug malformed compact policy"},
                    llm=_FakeLLM(),
                )
                session.state["context_policy"] = {
                    "enabled": True,
                    "context_window": 200000,
                    "trigger_ratio": 0.82,
                    "keep_last_turns": "oops",
                    "max_tool_chars": "bad",
                    "microcompact_enabled": True,
                    "llm_compact_enabled": False,
                    "llm_compact_input_max_chars": "bad",
                }
                for idx in range(8):
                    session.run_turn(f"turn {idx}")

                compact_info = session.compact_context()
                self.assertTrue(compact_info["changed"])
                self.assertEqual(session.state["context_policy"]["keep_last_turns"], 6)
                self.assertEqual(session.state["context_policy"]["max_tool_chars"], 1500)
                self.assertEqual(session.state["context_policy"]["llm_compact_input_max_chars"], 24000)
            finally:
                if previous_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = previous_home

    def test_checkpoint_snapshot_omits_duplicate_chat_history_and_ignores_stale_planner_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                session = InteractiveSession(
                    input_path=None,
                    user_goal={"raw_question": "checkpoint dedupe"},
                    llm=_FakeLLM(),
                )
                session.state["context_policy"] = {
                    "enabled": False,
                    "context_window": 200000,
                    "trigger_ratio": 0.82,
                    "keep_last_turns": 2,
                    "max_tool_chars": 200,
                    "microcompact_enabled": True,
                    "llm_compact_enabled": False,
                    "llm_compact_input_max_chars": 24000,
                }
                session.run_turn("turn 0")
                session.run_turn("turn 1")

                earlier_saved = session.store.get_conversation(session.session_id)
                earlier_planner_snapshot = dict(((earlier_saved or {}).get("agent_snapshots") or {}).get("planner") or {})
                self.assertNotIn("chat_history", earlier_planner_snapshot)

                session.run_turn("turn 2")
                latest_saved = session.store.get_conversation(session.session_id)
                latest_hist_raw = list(((latest_saved or {}).get("agent_histories") or {}).get("planner") or [])
                latest_history = [load(msg) for msg in latest_hist_raw]
                self.assertTrue(latest_history)

                stale_snapshot = dict(earlier_planner_snapshot)
                stale_snapshot["history_revision"] = int((earlier_saved or {}).get("history_revision", 0))

                mutated = dict(latest_saved or {})
                mutated_snapshots = dict(mutated.get("agent_snapshots") or {})
                mutated_snapshots["planner"] = stale_snapshot
                mutated["agent_snapshots"] = mutated_snapshots
                session.store._write_json_atomic(session.store._get_conversation_path(session.session_id), mutated)

                restored = InteractiveSession.from_checkpoint(session.session_id, _FakeLLM())
                self.assertEqual(len(restored.planner.checkpoint_messages()), 0)
                self.assertEqual(len(restored.planner.chat_history), len(latest_history))
                self.assertEqual(
                    getattr(restored.planner.chat_history[-1], "content", ""),
                    getattr(latest_history[-1], "content", ""),
                )
            finally:
                if previous_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = previous_home

    def test_offline_compact_preserves_top_level_transcript_and_updates_planner_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConversationStore(base_dir=tmpdir)
            session_id = store.generate_session_id()
            planner_messages = [
                HumanMessage(content=f"user {idx}") if idx % 2 == 0 else AIMessage(content=f"assistant {idx}")
                for idx in range(8)
            ]
            snapshot = {
                "schema_version": 2,
                "session_id": session_id,
                "metadata": {},
                "messages": [{"role": "user", "content": f"top {idx}"} for idx in range(8)],
                "agent_state": {
                    "session_id": session_id,
                    "context_policy": {
                        "enabled": True,
                        "context_window": 200000,
                        "trigger_ratio": 0.82,
                        "keep_last_turns": 2,
                        "max_tool_chars": 200,
                        "microcompact_enabled": True,
                        "llm_compact_enabled": False,
                        "llm_compact_input_max_chars": 24000,
                    },
                },
                "agent_histories": {
                    "planner": [dumpd(msg) for msg in planner_messages],
                },
                "agent_snapshots": {
                    "planner": {
                        "chat_history": [dumpd(msg) for msg in planner_messages],
                        "langgraph_checkpoint": {
                            "format": "langgraph_inmemory_v1",
                            "payload_b64": "gASVFgAAAAAAAAB9lCiMB3N0b3JhZ2WUfZSMBndyaXRlc5R9lIwFYmxvYnOUfZR1Lg==",
                        },
                    }
                },
                "events_log": [],
                "history_revision": 0,
                "compaction_stats": {},
            }
            store._write_json_atomic(store._get_conversation_path(session_id), snapshot)

            result = store.compact_conversation(session_id, keep_last_turns=2)
            self.assertTrue(result["ok"])

            saved = store.get_conversation(session_id)
            self.assertEqual(len(saved.get("messages") or []), 8)
            planner_snapshot = (saved.get("agent_snapshots") or {}).get("planner") or {}
            planner_history_raw = (saved.get("agent_histories") or {}).get("planner") or []
            planner_history = [load(msg) for msg in planner_history_raw]
            self.assertIsNone(planner_snapshot.get("langgraph_checkpoint"))
            self.assertNotIn("chat_history", planner_snapshot)
            self.assertTrue(any(is_compact_boundary_message(msg) for msg in planner_history))

    def test_offline_compact_normalizes_malformed_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConversationStore(base_dir=tmpdir)
            session_id = store.generate_session_id()
            planner_messages = [
                HumanMessage(content=f"user {idx}") if idx % 2 == 0 else AIMessage(content=f"assistant {idx}")
                for idx in range(20)
            ]
            snapshot = {
                "schema_version": 2,
                "session_id": session_id,
                "metadata": {},
                "messages": [{"role": "user", "content": f"top {idx}"} for idx in range(8)],
                "agent_state": {
                    "session_id": session_id,
                    "context_policy": {
                        "enabled": True,
                        "context_window": 200000,
                        "trigger_ratio": 0.82,
                        "keep_last_turns": "oops",
                        "max_tool_chars": "bad",
                        "microcompact_enabled": True,
                        "llm_compact_enabled": False,
                        "llm_compact_input_max_chars": "bad",
                    },
                },
                "agent_histories": {
                    "planner": [dumpd(msg) for msg in planner_messages],
                },
                "agent_snapshots": {
                    "planner": {
                        "chat_history": [dumpd(msg) for msg in planner_messages],
                        "langgraph_checkpoint": None,
                    }
                },
                "events_log": [],
                "history_revision": 0,
                "compaction_stats": {},
            }
            store._write_json_atomic(store._get_conversation_path(session_id), snapshot)

            result = store.compact_conversation(session_id, keep_last_turns=2)
            self.assertTrue(result["ok"])
            self.assertTrue(result["changed"])

            saved = store.get_conversation(session_id)
            normalized = dict((saved.get("agent_state") or {}).get("context_policy") or {})
            planner_snapshot = (saved.get("agent_snapshots") or {}).get("planner") or {}
            self.assertEqual(normalized.get("keep_last_turns"), 6)
            self.assertEqual(normalized.get("max_tool_chars"), 1500)
            self.assertEqual(normalized.get("llm_compact_input_max_chars"), 24000)
            self.assertNotIn("chat_history", planner_snapshot)

    def test_manual_compact_is_rejected_while_turn_is_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                started = threading.Event()
                release = threading.Event()

                def _slow_invoke(llm, messages, logger, label):
                    started.set()
                    release.wait(timeout=5.0)
                    return AIMessage(content="phase=final_answer\nslow response")

                agent_mod.invoke_with_retry = _slow_invoke
                session = InteractiveSession(
                    input_path=None,
                    user_goal={"raw_question": "concurrent compact"},
                    llm=_FakeLLM(),
                )
                session.state["context_policy"] = {
                    "enabled": True,
                    "context_window": 50,
                    "trigger_ratio": 0.1,
                    "keep_last_turns": 2,
                    "max_tool_chars": 200,
                    "microcompact_enabled": True,
                    "llm_compact_enabled": False,
                    "llm_compact_input_max_chars": 24000,
                }
                for idx in range(4):
                    session.run_turn(f"turn {idx}")

                turn_result = {"reply": None}

                def _run_turn() -> None:
                    turn_result["reply"] = session.run_turn("running turn")

                worker = threading.Thread(target=_run_turn)
                worker.start()
                self.assertTrue(started.wait(timeout=2.0))

                with self.assertRaisesRegex(RuntimeError, "Cannot compact context while an agent turn is still running"):
                    session.compact_context()

                release.set()
                worker.join()
                self.assertEqual(turn_result["reply"], "slow response")
                self.assertEqual(
                    len(session.planner.chat_history),
                    len(session.planner.checkpoint_messages()),
                )
            finally:
                if previous_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = previous_home


if __name__ == "__main__":
    unittest.main()
