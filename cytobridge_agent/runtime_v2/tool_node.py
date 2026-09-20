from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..tools.file_tools import ReadResult


def _stringify_tool_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, ReadResult):
        return value.tool_text
    if isinstance(value, dict) and "tool_text" in value and "media" in value:
        return str(value.get("tool_text") or "")
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            return str(value)
    return str(value)


def _extract_media_envelope(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, ReadResult):
        return {
            "tool_text": value.tool_text,
            "media": list(value.media or []),
            "metadata": dict(value.metadata or {}),
        }
    if isinstance(value, dict) and "tool_text" in value and "media" in value:
        return {
            "tool_text": str(value.get("tool_text") or ""),
            "media": list(value.get("media") or []),
            "metadata": dict(value.get("metadata") or {}),
        }
    return None


def _build_synthetic_media_message(
    tool_name: str,
    tool_call_id: str,
    envelope: Dict[str, Any],
) -> HumanMessage:
    blocks: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"[Synthetic media from tool `{tool_name}` for tool_call_id `{tool_call_id}`.] "
                "The preceding tool result contains the audit summary. Inspect the visual content below."
            ),
        }
    ]
    blocks.extend(block for block in envelope.get("media") or [] if isinstance(block, dict))
    return HumanMessage(
        content=blocks,
        additional_kwargs={
            "synthetic_tool_media": True,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "tool_metadata": dict(envelope.get("metadata") or {}),
        },
    )


def build_media_aware_tool_node(tools: Iterable[Any]):
    tool_map = {
        str(getattr(tool, "name", "")).strip(): tool
        for tool in tools
        if str(getattr(tool, "name", "")).strip()
    }

    def tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = list(state.get("messages") or [])
        if not messages:
            return {"messages": []}
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return {"messages": []}
        multimodal_enabled = bool(state.get("enable_multimodal", True))

        outputs: List[Any] = []
        for idx, tool_call in enumerate(getattr(last_message, "tool_calls", None) or []):
            if not isinstance(tool_call, dict):
                continue
            tool_name = str(tool_call.get("name") or "").strip()
            if not tool_name:
                continue
            tool_call_id = str(tool_call.get("id") or f"tool_call_{idx}")
            args = tool_call.get("args", {}) or {}
            if not isinstance(args, dict):
                args = {}

            tool = tool_map.get(tool_name)
            if tool is None:
                outputs.append(
                    ToolMessage(
                        content=f"Error: requested tool '{tool_name}' is not available.",
                        name=tool_name,
                        tool_call_id=tool_call_id,
                    )
                )
                continue

            try:
                result = tool.invoke(args)
            except BaseException as exc:  # noqa: BLE001
                outputs.append(
                    ToolMessage(
                        content=f"Error: tool '{tool_name}' failed: {exc}",
                        name=tool_name,
                        tool_call_id=tool_call_id,
                    )
                )
                continue

            envelope = _extract_media_envelope(result)
            if envelope is not None:
                tool_text = str(envelope.get("tool_text") or "")
                if envelope.get("media") and not multimodal_enabled:
                    tool_text = (
                        f"{tool_text}\n\nVisual content was not attached because multimodal is disabled "
                        "for this session."
                    ).strip()
                outputs.append(
                    ToolMessage(
                        content=tool_text,
                        name=tool_name,
                        tool_call_id=tool_call_id,
                    )
                )
                if envelope.get("media") and multimodal_enabled:
                    outputs.append(_build_synthetic_media_message(tool_name, tool_call_id, envelope))
                continue

            outputs.append(
                ToolMessage(
                    content=_stringify_tool_output(result),
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )
            )
        return {"messages": outputs}

    return tool_node
