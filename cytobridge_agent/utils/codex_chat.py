from __future__ import annotations

import base64
import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from .codex_sidecar_client import codex_sidecar_call

logger = logging.getLogger("cytobridge_agent.codex_chat")


def _parse_data_url(url: str) -> Optional[Dict[str, str]]:
    if not url.startswith("data:") or "," not in url:
        return None
    header, payload = url.split(",", 1)
    if ";base64" not in header:
        return None
    mime_type = header[5:].split(";", 1)[0].strip()
    if not mime_type:
        return None
    try:
        base64.b64decode(payload, validate=True)
    except Exception:
        return None
    return {"type": "image", "data": payload, "mimeType": mime_type}


def _stringify_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        text_parts: List[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            else:
                text_parts.append(str(item))
        return "\n".join(part for part in text_parts if part)
    if value is None:
        return ""
    return str(value)


class ChatCodexOAuth(BaseChatModel):
    """LangChain adapter that delegates Codex chat + failover to the Node sidecar."""

    model_name: str = "gpt-5.3-codex"
    temperature: float = 0.0
    max_tokens: int = 10000
    preferred_profile_id: Optional[str] = None
    thinking_level: str = "low"
    session_id: Optional[str] = None
    transport: str = "auto"

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "codex-oauth-sidecar"

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "ChatCodexOAuth":
        return super().bind(tools=tools, **kwargs)

    def _extract_system_prompt(self, messages: List[BaseMessage]) -> Optional[str]:
        parts: List[str] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                text = _stringify_content(message.content).strip()
                if text:
                    parts.append(text)
        return "\n".join(parts) if parts else None

    def _format_message_content(self, message: BaseMessage) -> Any:
        if isinstance(message.content, str):
            return message.content
        if not isinstance(message.content, list):
            return _stringify_content(message.content)

        parts: List[Dict[str, Any]] = []
        for block in message.content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append({"type": "text", "text": str(block.get("text", ""))})
                continue
            if block.get("type") == "image_url":
                image_url = block.get("image_url", {}).get("url", "")
                parsed = _parse_data_url(str(image_url))
                if parsed:
                    parts.append(parsed)
        if not parts:
            return _stringify_content(message.content)
        return parts

    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                continue
            if isinstance(message, HumanMessage):
                formatted.append(
                    {"role": "user", "content": self._format_message_content(message)}
                )
                continue
            if isinstance(message, AIMessage):
                formatted.append(
                    {
                        "role": "assistant",
                        "content": _stringify_content(message.content),
                        "tool_calls": [
                            {
                                "id": call.get("id"),
                                "name": call.get("name"),
                                "args": call.get("args", {}),
                            }
                            for call in (message.tool_calls or [])
                            if isinstance(call, dict) and call.get("name")
                        ],
                    }
                )
                continue
            if isinstance(message, ToolMessage):
                formatted.append(
                    {
                        "role": "toolResult",
                        "tool_call_id": getattr(message, "tool_call_id", "") or "",
                        "tool_name": getattr(message, "name", "") or "",
                        "content": _stringify_content(message.content),
                        "is_error": False,
                    }
                )
                continue
            formatted.append({"role": "user", "content": _stringify_content(message.content)})
        if not formatted:
            # The Codex Responses transport rejects a request with only
            # `system_prompt` and no conversation input. Full context compaction
            # can legitimately collapse history into SystemMessages, so provide
            # a neutral continuation input instead of crashing the run.
            formatted.append(
                {
                    "role": "user",
                    "content": "Continue the current task using the system and compacted context above.",
                }
            )
        return formatted

    def _format_tools(self, kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        tools = kwargs.get("tools") or []
        formatted: List[Dict[str, Any]] = []
        for tool in tools:
            try:
                openai_tool = convert_to_openai_tool(tool)
            except Exception:
                logger.debug("Failed to convert tool for Codex sidecar", exc_info=True)
                continue
            function_spec = openai_tool.get("function") if isinstance(openai_tool, dict) else None
            if not isinstance(function_spec, dict):
                continue
            name = str(function_spec.get("name", "")).strip()
            if not name:
                continue
            formatted.append(
                {
                    "name": name,
                    "description": function_spec.get("description", "") or "",
                    "parameters": function_spec.get("parameters", {}) or {"type": "object"},
                }
            )
        return formatted

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop:
            logger.debug("Stop sequences are not explicitly handled by Codex sidecar: %s", stop)
        if not self.session_id:
            self.session_id = f"cytobridge-codex-{uuid.uuid4().hex}"

        result = codex_sidecar_call(
            "chat.codex.complete",
            {
                "model": self.model_name,
                "system_prompt": self._extract_system_prompt(messages),
                "messages": self._format_messages(messages),
                "tools": self._format_tools(kwargs),
                "preferred_profile_id": self.preferred_profile_id,
                "reasoning": self.thinking_level,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "session_id": self.session_id,
                "transport": self.transport,
            },
        )

        usage = result.get("usage") if isinstance(result, dict) else {}
        usage = usage if isinstance(usage, dict) else {}
        tool_calls = [
            ToolCall(
                id=str(tool_call.get("id") or f"call_{idx}_{uuid.uuid4().hex[:8]}"),
                name=str(tool_call.get("name")),
                args=tool_call.get("args", {}) if isinstance(tool_call.get("args"), dict) else {},
            )
            for idx, tool_call in enumerate(result.get("tool_calls", []) if isinstance(result, dict) else [])
            if isinstance(tool_call, dict) and tool_call.get("name")
        ]
        message = AIMessage(
            content=str(result.get("content", "") if isinstance(result, dict) else ""),
            tool_calls=tool_calls,
            response_metadata={
                "provider": result.get("provider") if isinstance(result, dict) else "openai-codex",
                "profile_id": result.get("profile_id") if isinstance(result, dict) else None,
                "finish_reason": result.get("stop_reason") if isinstance(result, dict) else None,
                "error_message": result.get("error_message") if isinstance(result, dict) else None,
                "model": result.get("model") if isinstance(result, dict) else self.model_name,
                "token_usage": usage.get("token_usage", {}),
            },
            usage_metadata={
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            additional_kwargs={
                "profile_id": result.get("profile_id") if isinstance(result, dict) else None,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
