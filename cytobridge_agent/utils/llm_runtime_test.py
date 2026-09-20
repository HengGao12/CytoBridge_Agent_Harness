from __future__ import annotations

import logging
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage

from cytobridge_agent.utils.llm_factory import instantiate_llm
from cytobridge_agent.utils.llm_providers import (
    build_reasoning_kwargs,
    get_llm_provider,
    provider_model_ids,
)
from cytobridge_agent.utils.llm_runtime import get_llm_runtime_settings, invoke_with_retry


_ENV_KEYS = [
    "CYTOBRIDGE_LLM_MAX_RETRIES",
    "CYTOBRIDGE_LLM_SDK_MAX_RETRIES",
    "CYTOBRIDGE_LLM_RETRY_BACKOFF_SEC",
    "CYTOBRIDGE_LLM_RETRY_JITTER_SEC",
    "CYTOBRIDGE_LLM_RATE_LIMIT_RETRY_SEC",
]


class _FlakyModel:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            raise self.error
        return AIMessage(content="ok")


class _RateLimitError(RuntimeError):
    status_code = 429

    def __init__(self, retry_after: str) -> None:
        super().__init__("rate limit")
        self.response = SimpleNamespace(headers={"Retry-After": retry_after})


class LLMRuntimeRetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = {key: os.environ.get(key) for key in _ENV_KEYS}
        for key in _ENV_KEYS:
            os.environ.pop(key, None)
        self.logger = logging.getLogger("llm_runtime_test")

    def tearDown(self) -> None:
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_chatopenai_sdk_retries_default_to_zero(self) -> None:
        settings = get_llm_runtime_settings()
        self.assertEqual(settings["sdk_max_retries"], 0)

        llm, mode = instantiate_llm(
            model="mimo-v2.5-pro",
            base_url="http://localhost:1/v1",
            api_key="sk-test",
            auth_mode="api_key",
        )

        self.assertEqual(mode, "openai_compatible")
        self.assertEqual(getattr(llm, "max_retries", None), 0)

    def test_chatopenai_sdk_retries_are_configurable(self) -> None:
        os.environ["CYTOBRIDGE_LLM_SDK_MAX_RETRIES"] = "3"

        llm, _ = instantiate_llm(
            model="mimo-v2.5-pro",
            base_url="http://localhost:1/v1",
            api_key="sk-test",
            auth_mode="api_key",
        )

        self.assertEqual(getattr(llm, "max_retries", None), 3)

    def test_codex_oauth_accepts_gpt_5_6_sol_with_xhigh_reasoning(self) -> None:
        self.assertIn("gpt-5.6-sol", provider_model_ids("openai-codex"))
        self.assertEqual(get_llm_provider("openai-codex").default_model, "gpt-5.6-sol")

        llm, mode = instantiate_llm(
            model="gpt-5.6-sol",
            base_url=None,
            api_key=None,
            auth_mode="codex_oauth",
            provider="openai-codex",
            preferred_profile_id="openai-codex:test@example.com",
            thinking_level="xhigh",
        )

        self.assertEqual(mode, "codex_oauth")
        self.assertEqual(getattr(llm, "model_name", None), "gpt-5.6-sol")
        self.assertEqual(getattr(llm, "thinking_level", None), "xhigh")

    def test_xiaomi_roundtrips_reasoning_content_for_tool_followups(self) -> None:
        instantiate_llm(
            model="mimo-v2.5-pro",
            base_url="http://localhost:1/v1",
            api_key="sk-test",
            auth_mode="api_key",
            provider="xiaomi",
        )

        import langchain_openai.chat_models.base as openai_base

        message = openai_base._convert_dict_to_message(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "inspect tool outcome before continuing",
                "tool_calls": [],
            }
        )
        self.assertEqual(
            message.additional_kwargs.get("reasoning_content"),
            "inspect tool outcome before continuing",
        )

        payload = openai_base._convert_message_to_dict(message)
        self.assertEqual(
            payload.get("reasoning_content"),
            "inspect tool outcome before continuing",
        )
        self.assertNotIn("_cytobridge_reasoning_content_provider", payload)

    def test_reasoning_content_roundtrip_is_provider_scoped(self) -> None:
        instantiate_llm(
            model="mimo-v2.5-pro",
            base_url="http://localhost:1/v1",
            api_key="sk-test",
            auth_mode="api_key",
            provider="xiaomi",
        )
        import langchain_openai.chat_models.base as openai_base

        xiaomi_message = openai_base._convert_dict_to_message(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "xiaomi trace",
                "tool_calls": [],
            }
        )
        self.assertEqual(xiaomi_message.additional_kwargs.get("reasoning_content"), "xiaomi trace")

        instantiate_llm(
            model="deepseek-v4-pro",
            base_url="http://localhost:1/v1",
            api_key="sk-test",
            auth_mode="api_key",
            provider="deepseek",
        )
        deepseek_message = openai_base._convert_dict_to_message(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "deepseek trace",
                "tool_calls": [],
            }
        )
        self.assertEqual(deepseek_message.additional_kwargs.get("reasoning_content"), "deepseek trace")
        self.assertNotIn("reasoning_content", openai_base._convert_message_to_dict(xiaomi_message))
        self.assertEqual(
            openai_base._convert_message_to_dict(deepseek_message).get("reasoning_content"),
            "deepseek trace",
        )

    def test_xiaomi_uses_deepseek_style_thinking_payload(self) -> None:
        self.assertEqual(
            build_reasoning_kwargs("xiaomi", "mimo-v2.5-pro", "off"),
            {"extra_body": {"thinking": {"type": "disabled"}}},
        )
        self.assertEqual(
            build_reasoning_kwargs("xiaomi", "mimo-v2.5-pro", "high"),
            {"extra_body": {"thinking": {"type": "enabled", "reasoning_effort": "high"}}},
        )

    def test_outer_retry_uses_configured_backoff_without_jitter(self) -> None:
        os.environ["CYTOBRIDGE_LLM_MAX_RETRIES"] = "2"
        os.environ["CYTOBRIDGE_LLM_RETRY_BACKOFF_SEC"] = "1.5"
        os.environ["CYTOBRIDGE_LLM_RETRY_JITTER_SEC"] = "0"
        model = _FlakyModel(TimeoutError("request timed out"))

        with patch("cytobridge_agent.utils.llm_runtime.time.sleep") as sleep:
            response = invoke_with_retry(model, [("human", "hi")], self.logger, "test")

        self.assertEqual(response.content, "ok")
        self.assertEqual(model.calls, 2)
        sleep.assert_called_once_with(1.5)

    def test_outer_retry_honors_retry_after_header(self) -> None:
        os.environ["CYTOBRIDGE_LLM_MAX_RETRIES"] = "2"
        os.environ["CYTOBRIDGE_LLM_RETRY_JITTER_SEC"] = "0"
        os.environ["CYTOBRIDGE_LLM_RATE_LIMIT_RETRY_SEC"] = "60"
        model = _FlakyModel(_RateLimitError("2"))

        with patch("cytobridge_agent.utils.llm_runtime.time.sleep") as sleep:
            response = invoke_with_retry(model, [("human", "hi")], self.logger, "test")

        self.assertEqual(response.content, "ok")
        self.assertEqual(model.calls, 2)
        sleep.assert_called_once_with(2.0)


if __name__ == "__main__":
    unittest.main()
