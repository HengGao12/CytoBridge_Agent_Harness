from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from langchain_core.load import dumpd
from langchain_core.messages import AIMessage, HumanMessage

from cytobridge_agent.conversation_store import ConversationStore
from cytobridge_agent.interactive import InteractiveSession
from cytobridge_agent.runtime_v2 import agent as agent_mod
from cytobridge_agent.runtime_v2.resume import SQLITE_CHECKPOINT_FORMAT, create_runtime_checkpointer


class _FakeLLM:
    def bind_tools(self, tools):
        return self


class RuntimeCheckpointerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_agent_invoke = agent_mod.invoke_with_retry
        self._counter = 0

        def _fake_agent_invoke(llm, messages, logger, label):
            self._counter += 1
            return AIMessage(content=f"phase=final_answer\ncheckpoint response {self._counter}")

        agent_mod.invoke_with_retry = _fake_agent_invoke

    def tearDown(self) -> None:
        agent_mod.invoke_with_retry = self._orig_agent_invoke

    def test_interactive_session_uses_persistent_sqlite_checkpointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                session = InteractiveSession(
                    input_path=None,
                    user_goal={"raw_question": "checkpoint sqlite"},
                    llm=_FakeLLM(),
                )
                reply = session.run_turn("hello")
                self.assertIn("checkpoint response", reply)

                saved = session.store.get_conversation(session.session_id)
                planner_snapshot = dict(((saved or {}).get("agent_snapshots") or {}).get("planner") or {})
                checkpoint_snapshot = dict(planner_snapshot.get("langgraph_checkpoint") or {})
                self.assertEqual(checkpoint_snapshot.get("format"), SQLITE_CHECKPOINT_FORMAT)
                self.assertNotIn("payload_b64", checkpoint_snapshot)

                db_path = Path(str(checkpoint_snapshot.get("path") or ""))
                self.assertTrue(db_path.exists())
                self.assertTrue(session.planner.checkpoint_messages())

                restored = InteractiveSession.from_checkpoint(session.session_id, _FakeLLM())
                restored_snapshot = restored.planner.export_runtime_snapshot()["langgraph_checkpoint"]
                self.assertEqual(restored_snapshot.get("path"), checkpoint_snapshot.get("path"))
                self.assertTrue(restored.planner.checkpoint_messages())
                self.assertEqual(
                    len(restored.planner.chat_history),
                    len(restored.planner.checkpoint_messages()),
                )
            finally:
                if previous_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = previous_home

    def test_reset_checkpoint_namespace_does_not_reuse_old_sqlite_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                session = InteractiveSession(
                    input_path=None,
                    user_goal={"raw_question": "checkpoint reset"},
                    llm=_FakeLLM(),
                )
                session.run_turn("before reset")
                self.assertTrue(session.planner.checkpoint_messages())

                session.planner.replace_chat_history([HumanMessage(content="rewritten history")], reset_checkpoint=True)
                self.assertEqual(session.planner.checkpoint_messages(), [])

                session.planner.run("")
                checkpoint_contents = [getattr(msg, "content", "") for msg in session.planner.checkpoint_messages()]
                self.assertIn("rewritten history", checkpoint_contents)
            finally:
                if previous_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = previous_home

    def test_resume_reseeds_graph_when_sqlite_checkpoint_has_no_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            try:
                store = ConversationStore()
                session_id = store.generate_session_id()
                checkpoint_path = (
                    Path(tmpdir)
                    / ".cellcompass"
                    / "conversations"
                    / "checkpoints"
                    / f"{session_id}.sqlite"
                )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.touch()
                planner_messages = [
                    HumanMessage(content="old durable question"),
                    AIMessage(content="old durable answer"),
                ]
                snapshot = {
                    "schema_version": 2,
                    "session_id": session_id,
                    "metadata": {},
                    "messages": [
                        {"role": "user", "content": "old durable question"},
                        {"role": "assistant", "content": "old durable answer"},
                    ],
                    "agent_state": {
                        "session_id": session_id,
                        "user_goal": {"raw_question": "resume durable history"},
                    },
                    "agent_histories": {"planner": [dumpd(msg) for msg in planner_messages]},
                    "agent_snapshots": {
                        "planner": {
                            "thread_id": session_id,
                            "checkpoint_ns": "runtime_v2",
                            "history_revision": 0,
                            "langgraph_checkpoint": {
                                "format": SQLITE_CHECKPOINT_FORMAT,
                                "backend": "sqlite",
                                "path": str(checkpoint_path),
                            },
                        }
                    },
                    "events_log": [],
                    "history_revision": 0,
                    "compaction_stats": {},
                }
                store._write_json_atomic(store._get_conversation_path(session_id), snapshot)

                restored = InteractiveSession.from_checkpoint(session_id, _FakeLLM())
                self.assertEqual(
                    [getattr(msg, "content", "") for msg in restored.planner.chat_history],
                    ["old durable question", "old durable answer"],
                )
                self.assertEqual(restored.planner.checkpoint_messages(), [])
                self.assertTrue(restored.planner._seed_full_history_next_run)

                restored.run_turn("new followup")
                checkpoint_contents = [
                    getattr(msg, "content", "")
                    for msg in restored.planner.checkpoint_messages()
                ]
                self.assertIn("old durable question", checkpoint_contents)
                self.assertIn("new followup", checkpoint_contents)
            finally:
                if previous_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = previous_home

    def test_corrupt_sqlite_checkpoint_is_quarantined_and_recreated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "bad.sqlite"
            checkpoint_path.write_text("this is not sqlite", encoding="utf-8")

            checkpointer = create_runtime_checkpointer(
                {"session_id": "bad-sqlite", "langgraph_checkpoint_path": str(checkpoint_path)}
            )

            self.assertEqual(getattr(checkpointer, "_cytobridge_checkpoint_backend", ""), "sqlite")
            self.assertTrue(checkpoint_path.exists())
            self.assertGreater(checkpoint_path.stat().st_size, 0)
            quarantines = list(Path(tmpdir).glob("quarantine_bad_sqlite_*"))
            self.assertEqual(len(quarantines), 1)
            self.assertTrue((quarantines[0] / "bad.sqlite").exists())


if __name__ == "__main__":
    unittest.main()
