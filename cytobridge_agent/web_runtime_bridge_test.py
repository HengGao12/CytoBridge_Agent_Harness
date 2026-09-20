from __future__ import annotations

import asyncio
import unittest

from cytobridge_agent import web_server


class _FakePlanner:
    def __init__(self) -> None:
        self.stop_check = None


class _FakeSession:
    def __init__(self) -> None:
        self.session_id = "web123"
        self.input_path = "input.h5ad"
        self.output_dir = "out"
        self.state = {"conversation_turn": 3}
        self.planner = _FakePlanner()


class WebRuntimeBridgeTest(unittest.TestCase):
    def tearDown(self) -> None:
        web_server.session_state.controller = None
        web_server.session_state.active_session = None
        web_server.session_state.stop_requested = False
        web_server.session_state.events_log.clear()
        if web_server._matplotlib_patch_state.get("patched"):
            asyncio.run(web_server.shutdown_event())

    def test_adopt_web_session_exposes_shared_controller_status(self) -> None:
        session = _FakeSession()
        runtime_cfg = {
            "model": "mimo-v2.5-pro",
            "provider": "xiaomi",
            "base_url": "https://example/v1",
            "api_key": "key",
            "auth_mode": "api_key",
            "profile_id": None,
            "thinking_level": "high",
        }
        web_server._adopt_web_session(session, runtime_cfg)

        controller = web_server.session_state.controller
        self.assertIsNotNone(controller)
        self.assertIs(controller.active_session, session)
        self.assertEqual(controller.status()["session_id"], "web123")
        self.assertEqual(controller.status()["provider"], "xiaomi")
        self.assertEqual(controller.status()["snapshot"]["session"]["session_id"], "web123")
        self.assertTrue(callable(session.planner.stop_check))

    def test_active_runtime_status_uses_shared_snapshot(self) -> None:
        session = _FakeSession()
        runtime_cfg = {
            "model": "mimo-v2.5-pro",
            "provider": "xiaomi",
            "base_url": "https://example/v1",
            "api_key": "key",
            "auth_mode": "api_key",
            "profile_id": None,
            "thinking_level": "high",
        }
        web_server._adopt_web_session(session, runtime_cfg)

        status = web_server._active_runtime_status()

        self.assertEqual(status["session_id"], "web123")
        self.assertEqual(status["snapshot"]["session"]["session_id"], "web123")
        self.assertEqual(status["snapshot"]["adapter"]["name"], "web")

    def test_record_session_event_handles_controller_only_session(self) -> None:
        session = _FakeSession()
        runtime_cfg = {
            "model": "mimo-v2.5-pro",
            "provider": "xiaomi",
            "base_url": "https://example/v1",
            "api_key": "key",
            "auth_mode": "api_key",
            "profile_id": None,
            "thinking_level": "high",
        }
        web_server._adopt_web_session(session, runtime_cfg)

        event = web_server._record_session_event({"type": "unit_event", "data": {"ok": True}})

        self.assertEqual(event["type"], "unit_event")
        self.assertEqual(web_server.session_state.events_log[-1]["type"], "unit_event")

    def test_lifespan_startup_and_shutdown_manage_runtime_hooks(self) -> None:
        async def run_cycle() -> None:
            await web_server.startup_event()
            self.assertIsNotNone(web_server.server_loop)
            self.assertTrue(web_server._matplotlib_patch_state.get("patched"))
            await web_server.shutdown_event()

        asyncio.run(run_cycle())
        self.assertIsNone(web_server.server_loop)
        self.assertFalse(web_server._matplotlib_patch_state)


if __name__ == "__main__":
    unittest.main()
