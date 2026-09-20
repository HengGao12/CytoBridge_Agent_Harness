from __future__ import annotations

import unittest

from cytobridge_agent.cli_runtime import (
    CLIEventSink,
    SessionOpenSpec,
    open_or_create_session,
    resolve_resume_session_id,
    user_goal_from_spec,
)
from cytobridge_agent.cli_terminal import (
    choose_session_interactively,
    filter_sessions,
    format_session_detail,
    format_session_listing,
)


class _FakeSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id


class _FakeController:
    def __init__(self) -> None:
        self.created = []
        self.resumed = []
        self.sessions = [{"session_id": "latest123"}, {"session_id": "older456"}]

    def latest_session_id(self):
        return self.sessions[0]["session_id"] if self.sessions else None

    def list_sessions(self, limit=20):
        return self.sessions[:limit]

    def resume_session(self, session_id):
        self.resumed.append(session_id)
        return _FakeSession(session_id)

    def new_session(self, **kwargs):
        self.created.append(kwargs)
        return _FakeSession("new123")


class CLIRuntimeHelpersTest(unittest.TestCase):
    def test_open_or_create_creates_shared_user_goal(self) -> None:
        controller = _FakeController()
        session, resumed = open_or_create_session(
            controller,
            SessionOpenSpec(input_path="data.h5ad", question="analyze", output_dir="out", device="cpu"),
        )
        self.assertFalse(resumed)
        self.assertEqual(session.session_id, "new123")
        self.assertEqual(controller.created[0]["user_goal"], user_goal_from_spec(SessionOpenSpec(question="analyze", device="cpu")))

    def test_user_goal_overrides_extend_shared_goal(self) -> None:
        goal = user_goal_from_spec(
            SessionOpenSpec(
                question="analyze",
                device="cpu",
                user_goal_overrides={"benchmark_strict": True, "requested_analyses": ["velocity"]},
            )
        )
        self.assertTrue(goal["benchmark_strict"])
        self.assertEqual(goal["requested_analyses"], ["velocity"])
        self.assertEqual(goal["raw_question"], "analyze")

    def test_open_or_create_resumes_when_session_id_present(self) -> None:
        controller = _FakeController()
        session, resumed = open_or_create_session(controller, SessionOpenSpec(session_id="abc"))
        self.assertTrue(resumed)
        self.assertEqual(session.session_id, "abc")
        self.assertEqual(controller.resumed, ["abc"])

    def test_resolve_latest_matching_session_id(self) -> None:
        controller = _FakeController()
        controller.sessions = [
            {"session_id": "latest123", "metadata": {"title": "other"}},
            {"session_id": "older456", "metadata": {"title": "packer analysis"}},
        ]
        self.assertEqual(resolve_resume_session_id(controller, SessionOpenSpec(last=True, query="packer")), "older456")

    def test_event_sink_records_runtime_envelope(self) -> None:
        sink = CLIEventSink()
        sink.emit("session_started", {"session_id": "abc"})
        self.assertEqual(sink.events[0]["type"], "session_started")
        self.assertEqual(sink.events[0]["session_id"], "abc")
        self.assertIn("event_id", sink.events[0])


class CLITerminalHelpersTest(unittest.TestCase):
    def _sessions(self):
        return [
            {
                "session_id": "abcdef12",
                "metadata": {"title": "First analysis", "updated_at": "2026-05-20T08:30:00Z", "cwd": "/tmp/project"},
                "preview": "preview",
            },
            {
                "session_id": "12345678",
                "metadata": {"title": "Second analysis", "updated_at": "2026-05-19T01:02:03Z"},
                "preview": "",
            },
        ]

    def test_format_session_listing_is_numbered_and_searchable(self) -> None:
        text = format_session_listing(self._sessions())
        self.assertIn("1. abcdef12", text)
        self.assertIn("First analysis", text)
        self.assertIn("[/tmp/project]", text)

    def test_choose_session_accepts_number_prefix_and_cancel(self) -> None:
        self.assertEqual(
            choose_session_interactively(self._sessions(), input_fn=lambda _: "2", output_fn=lambda _: None),
            "12345678",
        )
        self.assertEqual(
            choose_session_interactively(self._sessions(), input_fn=lambda _: "abc", output_fn=lambda _: None),
            "abcdef12",
        )
        self.assertIsNone(choose_session_interactively(self._sessions(), input_fn=lambda _: "q", output_fn=lambda _: None))

    def test_filter_sessions_matches_title_and_location(self) -> None:
        self.assertEqual([item["session_id"] for item in filter_sessions(self._sessions(), query="second")], ["12345678"])
        self.assertEqual([item["session_id"] for item in filter_sessions(self._sessions(), cwd="/tmp/project")], ["abcdef12"])

    def test_format_session_detail_includes_recent_events(self) -> None:
        text = format_session_detail(
            {"status": "success", "session_id": "abc", "metadata": {"title": "Analysis"}, "preview": "agent reply"},
            [{"type": "turn_complete", "timestamp": "2026-05-20T08:31:00Z", "data": {"response": "done"}}],
        )
        self.assertIn("Session: abc", text)
        self.assertIn("Recent events:", text)


if __name__ == "__main__":
    unittest.main()
