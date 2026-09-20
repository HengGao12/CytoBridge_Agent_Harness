from __future__ import annotations

import os
import time
import unittest

from langchain_core.messages import AIMessage, ToolMessage

from cytobridge_agent.runtime_v2.tool_isolation import run_tool_in_subprocess
from cytobridge_agent.runtime_v2.tool_node import build_media_aware_tool_node
from cytobridge_agent.runtime_v2.tool_registry import SingleAgentTools


def _raise_memory_error() -> None:
    raise MemoryError("simulated memory pressure")


def _exit_without_payload() -> None:
    os._exit(137)


def _sleep_forever() -> None:
    time.sleep(30)


def _return_value() -> dict[str, object]:
    return {"ok": True, "value": 3}


def _return_large_payload() -> dict[str, object]:
    return {"blob": "x" * (2 * 1024 * 1024)}


def _return_tool_timeout(timeout: float) -> dict[str, object]:
    return {"tool_timeout": timeout}


class _ExplodingTool:
    name = "explode"

    def invoke(self, args):  # noqa: ANN001
        raise MemoryError("tool-node memory error")


class ToolIsolationTest(unittest.TestCase):
    def test_memory_error_is_returned_as_data(self) -> None:
        result = run_tool_in_subprocess(_raise_memory_error, timeout=5)

        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "MemoryError")
        self.assertIn("simulated memory pressure", result.error or "")

    def test_process_crash_is_returned_as_data(self) -> None:
        result = run_tool_in_subprocess(_exit_without_payload, timeout=5)

        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ToolProcessCrashed")
        self.assertEqual(result.returncode, 137)

    def test_timeout_kills_child_and_returns_data(self) -> None:
        result = run_tool_in_subprocess(_sleep_forever, timeout=0.2)

        self.assertFalse(result.ok)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.error_type, "TimeoutError")

    def test_successful_result_survives_round_trip(self) -> None:
        result = run_tool_in_subprocess(_return_value, timeout=5)

        self.assertTrue(result.ok)
        self.assertEqual(result.result, {"ok": True, "value": 3})

    def test_large_result_does_not_deadlock_pipe(self) -> None:
        result = run_tool_in_subprocess(_return_large_payload, timeout=5)

        self.assertTrue(result.ok)
        self.assertEqual(len(result.result["blob"]), 2 * 1024 * 1024)

    def test_tool_timeout_keyword_is_forwarded_to_child(self) -> None:
        result = run_tool_in_subprocess(
            _return_tool_timeout,
            isolation_timeout=5,
            tool_kwargs={"timeout": 1.5},
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.result, {"tool_timeout": 1.5})

    def test_tool_node_converts_base_exception_to_tool_message(self) -> None:
        node = build_media_aware_tool_node([_ExplodingTool()])
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "explode",
                            "args": {},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

        output = node(state)

        self.assertEqual(len(output["messages"]), 1)
        message = output["messages"][0]
        self.assertIsInstance(message, ToolMessage)
        self.assertIn("tool-node memory error", str(message.content))

    def test_registry_tool_wrapper_catches_base_exception(self) -> None:
        handler = object.__new__(SingleAgentTools)
        tool = handler._build_tool(
            _raise_memory_error,
            "memory_tool",
            "raise memory error",
            emit_events=False,
        )

        output = tool.invoke({})

        self.assertIn("Error: tool 'memory_tool' failed", output)
        self.assertIn("MemoryError", output)

    def test_registry_tool_wrapper_isolates_process_crash(self) -> None:
        handler = object.__new__(SingleAgentTools)
        tool = handler._build_tool(
            _exit_without_payload,
            "crash_tool",
            "crash process",
            emit_events=False,
            isolation="process",
            isolation_timeout=5,
        )

        output = tool.invoke({})

        self.assertIn("isolated subprocess", output)
        self.assertIn("returncode: 137", output)


if __name__ == "__main__":
    unittest.main()
