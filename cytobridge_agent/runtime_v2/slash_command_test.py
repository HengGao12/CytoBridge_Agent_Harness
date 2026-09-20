from __future__ import annotations

import asyncio
import threading
import unittest
from unittest.mock import patch

from cytobridge_agent import web_server


class SlashCommandTests(unittest.TestCase):
    def tearDown(self) -> None:
        thread = web_server.session_state.agent_thread
        if thread and thread.is_alive():
            thread.join(timeout=1)
        web_server.session_state.active_session = None
        web_server.session_state.events_log = []
        web_server.session_state.initial_config = {}
        web_server.session_state.agent_thread = None
        web_server.session_state.active_turn_id = None
        web_server.session_state.stop_requested = False
        web_server.session_state.recent_chat_requests.clear()

    def test_parse_slash_command_aliases(self) -> None:
        command, args, rest = web_server._parse_slash_command("/thinking high")
        self.assertEqual(command, "reasoning")
        self.assertEqual(args, ["high"])
        self.assertEqual(rest, "high")

        command, args, rest = web_server._parse_slash_command("/hooks on max 5")
        self.assertEqual(command, "hooks")
        self.assertEqual(args, ["on", "max", "5"])
        self.assertEqual(rest, "on max 5")

    def test_help_command_is_handled_without_agent_turn(self) -> None:
        result = asyncio.run(web_server._handle_slash_command("/help"))

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["command_handled"])
        self.assertIn("/reasoning", result["message"])
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["type"], "system")
        self.assertEqual(web_server.session_state.events_log[-1]["type"], "system")

    def test_unknown_command_returns_command_error(self) -> None:
        result = asyncio.run(web_server._handle_slash_command("/does_not_exist"))

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["command_handled"])
        self.assertIn("Unknown slash command", result["message"])

    def test_stop_command_works_without_active_session_as_command(self) -> None:
        result = asyncio.run(web_server._handle_slash_command("/stop"))

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["command_handled"])
        self.assertIn("Failed to stop agent", result["message"])

    def test_resume_requires_session_id(self) -> None:
        result = asyncio.run(web_server._handle_slash_command("/resume"))

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["command_handled"])
        self.assertIn("Usage: /resume SESSION_ID", result["message"])

    def test_new_command_returns_clear_timeline_payload(self) -> None:
        result = asyncio.run(web_server._handle_slash_command("/new"))

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["command_handled"])
        self.assertTrue(result["clear_timeline"])
        self.assertEqual(result["events"][0]["type"], "system")

    def test_provider_api_key_payload_preserves_legacy_provider_key(self) -> None:
        with patch(
            "cytobridge_agent.web_server.get_saved_config",
            return_value={"llm_provider": "xiaomi", "openai_api_key": "sk-xiaomi"},
        ):
            payload = web_server._provider_key_save_payload("deepseek", "sk-deepseek")

        self.assertEqual(payload["provider_api_keys"]["xiaomi"], "sk-xiaomi")
        self.assertEqual(payload["provider_api_keys"]["deepseek"], "sk-deepseek")
        self.assertNotIn("openai_api_key", payload)

    def test_provider_switch_does_not_reuse_previous_provider_base_url(self) -> None:
        web_server.session_state.initial_config = {
            "llm_provider": "xiaomi",
            "llm_model": "mimo-v2.5-pro",
            "llm_base_url": "https://token-plan-sgp.xiaomimimo.com/v1",
            "llm_auth_mode": "api_key",
            "provider_api_keys": {"xiaomi": "sk-xiaomi", "deepseek": "sk-deepseek"},
        }
        with patch(
            "cytobridge_agent.web_server.get_saved_config",
            return_value={
                "llm_provider": "xiaomi",
                "llm_model": "mimo-v2.5-pro",
                "llm_base_url": "https://token-plan-sgp.xiaomimimo.com/v1",
                "llm_auth_mode": "api_key",
                "provider_api_keys": {"xiaomi": "sk-xiaomi", "deepseek": "sk-deepseek"},
            },
        ):
            cfg = web_server._resolve_llm_runtime_config(
                web_server.ModelSwitchRequest(
                    llm_provider="deepseek",
                    llm_model="deepseek-v4-pro",
                    llm_thinking_level="xhigh",
                )
            )

        self.assertEqual(cfg["provider"], "deepseek")
        self.assertEqual(cfg["model"], "deepseek-v4-pro")
        self.assertIsNone(cfg["base_url"])
        self.assertEqual(cfg["api_key"], "sk-deepseek")

    def test_provider_switch_uses_provider_specific_saved_base_url(self) -> None:
        web_server.session_state.initial_config = {
            "llm_provider": "deepseek",
            "llm_model": "deepseek-v4-pro",
            "llm_auth_mode": "api_key",
            "provider_api_keys": {"xiaomi": "sk-xiaomi", "deepseek": "sk-deepseek"},
            "provider_base_urls": {"xiaomi": "https://token-plan-sgp.xiaomimimo.com/v1"},
        }
        with patch(
            "cytobridge_agent.web_server.get_saved_config",
            return_value={
                "llm_provider": "deepseek",
                "llm_model": "deepseek-v4-pro",
                "llm_base_url": "https://token-plan-sgp.xiaomimimo.com/v1",
                "llm_auth_mode": "api_key",
                "provider_api_keys": {"xiaomi": "sk-xiaomi", "deepseek": "sk-deepseek"},
                "provider_base_urls": {"xiaomi": "https://token-plan-sgp.xiaomimimo.com/v1"},
            },
        ):
            cfg = web_server._resolve_llm_runtime_config(
                web_server.ModelSwitchRequest(
                    llm_provider="xiaomi",
                    llm_model="mimo-v2.5-pro",
                    llm_thinking_level="high",
                )
            )

        self.assertEqual(cfg["provider"], "xiaomi")
        self.assertEqual(cfg["model"], "mimo-v2.5-pro")
        self.assertEqual(cfg["base_url"], "https://token-plan-sgp.xiaomimimo.com/v1")
        self.assertEqual(cfg["api_key"], "sk-xiaomi")
        self.assertEqual(cfg["thinking_level"], "high")

    def test_xiaomi_provider_switch_without_explicit_thinking_defaults_off(self) -> None:
        web_server.session_state.initial_config = {
            "llm_provider": "deepseek",
            "llm_model": "deepseek-v4-pro",
            "llm_auth_mode": "api_key",
            "llm_thinking_level": "high",
            "provider_api_keys": {"xiaomi": "sk-xiaomi", "deepseek": "sk-deepseek"},
            "provider_base_urls": {"xiaomi": "https://token-plan-sgp.xiaomimimo.com/v1"},
        }
        with patch(
            "cytobridge_agent.web_server.get_saved_config",
            return_value={
                "llm_provider": "deepseek",
                "llm_model": "deepseek-v4-pro",
                "llm_auth_mode": "api_key",
                "llm_thinking_level": "high",
                "provider_api_keys": {"xiaomi": "sk-xiaomi", "deepseek": "sk-deepseek"},
                "provider_base_urls": {"xiaomi": "https://token-plan-sgp.xiaomimimo.com/v1"},
            },
        ):
            cfg = web_server._resolve_llm_runtime_config(
                web_server.ModelSwitchRequest(
                    llm_provider="xiaomi",
                    llm_model="mimo-v2.5-pro",
                )
            )

        self.assertEqual(cfg["provider"], "xiaomi")
        self.assertEqual(cfg["thinking_level"], "off")

    def test_hooks_on_rebuilds_runtime_graph(self) -> None:
        class DummyPlanner:
            def __init__(self) -> None:
                self.refresh_count = 0

            def refresh_tooling(self) -> None:
                self.refresh_count += 1

        class DummySession:
            def __init__(self) -> None:
                self.state = {"stop_hook_enabled": False}
                self.planner = DummyPlanner()

        session = DummySession()
        web_server.session_state.active_session = session

        with patch("cytobridge_agent.web_server.save_config"):
            updates = web_server._update_stop_hook_settings(["on"], "on")

        self.assertEqual(updates["stop_hook_enabled"], True)
        self.assertEqual(session.planner.refresh_count, 1)

    def test_chat_client_request_id_dedupes_duplicate_turns(self) -> None:
        started = threading.Event()
        finish = threading.Event()

        class DummySession:
            session_id = "dedupe-session"

            def run_turn(self, user_message, attachments=None):
                del user_message, attachments
                started.set()
                finish.wait(timeout=1)
                return "ok"

            def _save_conversation(self):
                return None

        web_server.session_state.active_session = DummySession()

        first = web_server._start_agent_turn("hello", client_request_id="req-1")
        self.assertEqual(first["status"], "success")
        self.assertTrue(started.wait(timeout=1))

        duplicate = web_server._start_agent_turn("hello", client_request_id="req-1")
        self.assertEqual(duplicate["status"], "success")
        self.assertTrue(duplicate["duplicate"])

        finish.set()
        thread = web_server.session_state.agent_thread
        if thread:
            thread.join(timeout=1)

    def test_agent_stop_requested_aborts_partial_history_before_saving(self) -> None:
        class DummySession:
            session_id = "stop-cleanup-session"

            def __init__(self) -> None:
                self.aborted_reasons = []
                self.saved = 0
                self.conversation_history = [{"role": "user", "content": "hello"}]

            def run_turn(self, user_message, attachments=None):
                del user_message, attachments
                raise web_server.AgentStopRequested()

            def abort_interrupted_turn(self, reason):
                self.aborted_reasons.append(reason)
                if self.conversation_history and self.conversation_history[-1].get("role") == "user":
                    self.conversation_history.append({"role": "assistant", "content": reason})

            def _save_conversation(self):
                self.saved += 1

        session = DummySession()
        web_server.session_state.active_session = session

        result = web_server._start_agent_turn("hello", client_request_id="stop-cleanup")
        self.assertEqual(result["status"], "success")
        thread = web_server.session_state.agent_thread
        if thread:
            thread.join(timeout=1)

        self.assertEqual(session.aborted_reasons, ["Stopped by user."])
        self.assertEqual(
            session.conversation_history,
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Stopped by user."}],
        )
        self.assertEqual(session.saved, 1)
        self.assertIsNone(web_server.session_state.agent_thread)

    def test_stop_hook_settings_endpoint_does_not_rebuild_llm(self) -> None:
        class DummyPlanner:
            def __init__(self) -> None:
                self.refresh_count = 0

            def refresh_tooling(self) -> None:
                self.refresh_count += 1

        class DummySession:
            def __init__(self) -> None:
                self.state = {"stop_hook_enabled": True}
                self.planner = DummyPlanner()
                self.saved = 0

            def _save_conversation(self):
                self.saved += 1

        session = DummySession()
        web_server.session_state.active_session = session

        with patch("cytobridge_agent.web_server.save_config") as save_config, patch(
            "cytobridge_agent.web_server._instantiate_runtime_llm"
        ) as instantiate:
            result = asyncio.run(
                web_server.update_stop_hook_settings(
                    web_server.StopHookSettingsRequest(stop_hook_enabled=False)
                )
            )

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["stop_hook_enabled"])
        self.assertEqual(session.planner.refresh_count, 1)
        self.assertEqual(session.saved, 1)
        save_config.assert_called_once()
        instantiate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
