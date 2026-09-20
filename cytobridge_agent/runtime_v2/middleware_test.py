from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from cytobridge_agent.runtime_v2.middleware import RuntimeMiddleware
from cytobridge_agent.utils.llm_factory import instantiate_llm


class DummyThinkingLLM:
    extra_body = {"thinking": {"type": "enabled", "reasoning_effort": "high"}}


class DummyPlainLLM:
    extra_body = {}


class RuntimeMiddlewareReasoningTests(unittest.TestCase):
    def test_repairs_missing_tool_result_in_interrupted_history(self) -> None:
        middleware = RuntimeMiddleware({"llm_provider": "deepseek"}, DummyPlainLLM())
        messages = [
            HumanMessage(content="analyze the data"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "inspect", "args": {}, "id": "call_a", "type": "tool_call"},
                    {"name": "report", "args": {}, "id": "call_b", "type": "tool_call"},
                ],
            ),
            ToolMessage(content="inspection complete", name="inspect", tool_call_id="call_a"),
            HumanMessage(content="continue"),
        ]

        repaired, repair_count = middleware._sanitize_dangling_tool_results(messages)

        self.assertGreater(repair_count, 0)
        self.assertEqual([type(message) for message in repaired], [HumanMessage, AIMessage, ToolMessage, ToolMessage, HumanMessage])
        self.assertEqual(repaired[2].tool_call_id, "call_a")
        self.assertEqual(repaired[3].tool_call_id, "call_b")
        self.assertIn("prior run ended", str(repaired[3].content))

        repaired_again, second_repair_count = middleware._sanitize_dangling_tool_results(repaired)
        self.assertEqual(second_repair_count, 0)
        self.assertEqual(repaired_again, repaired)

    def test_prepare_messages_persists_repaired_history(self) -> None:
        middleware = RuntimeMiddleware(
            {"llm_provider": "deepseek", "context_policy": {"enabled": False}},
            DummyPlainLLM(),
        )
        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="question"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "inspect", "args": {}, "id": "call_a", "type": "tool_call"},
                ],
            ),
            HumanMessage(content="continue"),
        ]

        prepared = middleware.prepare_messages(messages)

        self.assertIsNotNone(prepared.persisted_history)
        self.assertEqual(
            [type(message) for message in prepared.prompt_messages],
            [SystemMessage, HumanMessage, AIMessage, ToolMessage, HumanMessage],
        )
        self.assertEqual(prepared.prompt_messages[3].tool_call_id, "call_a")

    def test_orders_all_tool_results_before_synthetic_media(self) -> None:
        middleware = RuntimeMiddleware({"llm_provider": "deepseek"}, DummyPlainLLM())
        media = HumanMessage(
            content="image",
            additional_kwargs={"synthetic_tool_media": True, "tool_call_id": "call_a"},
        )
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_a", "args": {}, "id": "call_a", "type": "tool_call"},
                    {"name": "read_b", "args": {}, "id": "call_b", "type": "tool_call"},
                ],
            ),
            ToolMessage(content="A", name="read_a", tool_call_id="call_a"),
            media,
            ToolMessage(content="B", name="read_b", tool_call_id="call_b"),
            HumanMessage(content="next question"),
        ]

        repaired, repair_count = middleware._sanitize_dangling_tool_results(messages)

        self.assertGreater(repair_count, 0)
        self.assertEqual([type(message) for message in repaired], [AIMessage, ToolMessage, ToolMessage, HumanMessage, HumanMessage])
        self.assertEqual([repaired[1].tool_call_id, repaired[2].tool_call_id], ["call_a", "call_b"])
        self.assertIs(repaired[3], media)

    def test_thinking_provider_patches_legacy_tool_call_reasoning_content(self) -> None:
        middleware = RuntimeMiddleware({"llm_provider": "xiaomi"}, DummyThinkingLLM())
        messages = [
            HumanMessage(content="use a tool"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "inspect",
                        "args": {},
                        "id": "call_inspect",
                        "type": "tool_call",
                    }
                ],
            ),
        ]

        patched, added = middleware._ensure_tool_call_reasoning_content(messages)

        self.assertEqual(added, 1)
        self.assertEqual(
            patched[1].additional_kwargs.get("reasoning_content"),
            "Legacy tool-call message predates provider reasoning_content capture.",
        )
        self.assertEqual(
            patched[1].additional_kwargs.get("_cytobridge_reasoning_content_provider"),
            "xiaomi",
        )

    def test_xiaomi_legacy_reasoning_placeholder_survives_openai_payload_conversion(self) -> None:
        instantiate_llm(
            model="mimo-v2.5-pro",
            base_url="http://localhost:1/v1",
            api_key="sk-test",
            auth_mode="api_key",
            provider="xiaomi",
            thinking_level="high",
        )
        import langchain_openai.chat_models.base as openai_base

        middleware = RuntimeMiddleware({"llm_provider": "xiaomi"}, DummyThinkingLLM())
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "inspect",
                        "args": {},
                        "id": "call_inspect",
                        "type": "tool_call",
                    }
                ],
            ),
        ]

        patched, added = middleware._ensure_tool_call_reasoning_content(messages)
        payload = openai_base._convert_message_to_dict(patched[0])

        self.assertEqual(added, 1)
        self.assertEqual(
            payload.get("reasoning_content"),
            "Legacy tool-call message predates provider reasoning_content capture.",
        )
        self.assertNotIn("_cytobridge_reasoning_content_provider", payload)

    def test_xiaomi_empty_reasoning_content_survives_openai_payload_conversion(self) -> None:
        instantiate_llm(
            model="mimo-v2.5-pro",
            base_url="http://localhost:1/v1",
            api_key="sk-test",
            auth_mode="api_key",
            provider="xiaomi",
            thinking_level="high",
        )
        import langchain_openai.chat_models.base as openai_base

        message = AIMessage(
            content="",
            additional_kwargs={
                "reasoning_content": "",
                "_cytobridge_reasoning_content_provider": "xiaomi",
            },
            tool_calls=[
                {
                    "name": "inspect",
                    "args": {},
                    "id": "call_inspect",
                    "type": "tool_call",
                }
            ],
        )

        payload = openai_base._convert_message_to_dict(message)

        self.assertIn("reasoning_content", payload)
        self.assertEqual(payload["reasoning_content"], "")
        self.assertNotIn("_cytobridge_reasoning_content_provider", payload)

    def test_thinking_provider_patches_raw_additional_tool_calls(self) -> None:
        middleware = RuntimeMiddleware({"llm_provider": "xiaomi"}, DummyThinkingLLM())
        messages = [
            AIMessage(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": "call_raw",
                            "type": "function",
                            "function": {"name": "inspect", "arguments": "{}"},
                        }
                    ]
                },
            ),
        ]

        patched, added = middleware._ensure_tool_call_reasoning_content(messages)

        self.assertEqual(added, 1)
        self.assertEqual(
            patched[0].additional_kwargs.get("reasoning_content"),
            "Legacy tool-call message predates provider reasoning_content capture.",
        )

    def test_thinking_provider_patches_legacy_plain_assistant_reasoning_content(self) -> None:
        middleware = RuntimeMiddleware({"llm_provider": "xiaomi"}, DummyThinkingLLM())
        messages = [AIMessage(content="previous answer without captured thinking trace")]

        patched, added = middleware._ensure_tool_call_reasoning_content(messages)

        self.assertEqual(added, 1)
        self.assertEqual(
            patched[0].additional_kwargs.get("reasoning_content"),
            "Legacy assistant message predates provider reasoning_content capture.",
        )
        self.assertEqual(
            patched[0].additional_kwargs.get("_cytobridge_reasoning_content_provider"),
            "xiaomi",
        )

    def test_deepseek_thinking_provider_patches_legacy_tool_call_reasoning_content(self) -> None:
        middleware = RuntimeMiddleware({"llm_provider": "deepseek"}, DummyThinkingLLM())
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "inspect",
                        "args": {},
                        "id": "call_inspect",
                        "type": "tool_call",
                    }
                ],
            ),
        ]

        patched, added = middleware._ensure_tool_call_reasoning_content(messages)

        self.assertEqual(added, 1)
        self.assertEqual(
            patched[0].additional_kwargs.get("reasoning_content"),
            "Legacy tool-call message predates provider reasoning_content capture.",
        )

    def test_plain_provider_does_not_add_reasoning_content(self) -> None:
        middleware = RuntimeMiddleware({"llm_provider": "xiaomi"}, DummyPlainLLM())
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "inspect",
                        "args": {},
                        "id": "call_inspect",
                        "type": "tool_call",
                    }
                ],
            ),
        ]

        patched, added = middleware._ensure_tool_call_reasoning_content(messages)

        self.assertEqual(added, 0)
        self.assertNotIn("reasoning_content", patched[0].additional_kwargs)


if __name__ == "__main__":
    unittest.main()
