from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langchain_openai import ChatOpenAI

from ..display import DisplayManager
from ..campaign_state import resolve_active_campaign, state_with_resolved_campaign
from ..lifecycle_state import summarize_algorithm_lifecycle
from ..tools.turn_phase import (
    PHASE_COMMENTARY,
    PHASE_FINAL_ANSWER,
    PHASE_NEEDS_INPUT,
    resolve_message_phase,
    strip_tool_call_transcript,
)
from ..utils.llm_runtime import extract_usage_stats, invoke_with_retry
from ..tools.planner_file_tools import CAMPAIGN_STAGE_ORDER, IMPLEMENTATION_REVIEW_POLICY_VERSION
from .agent_types import (
    SUBMIT_IMPLEMENTATION_REVIEW_TOOL_NAME,
    SUBMIT_PROPOSAL_REVIEW_TOOL_NAME,
    SUBMIT_RESEARCH_IDEA_REVIEW_TOOL_NAME,
    SUBMIT_SUBAGENT_RESULT_TOOL_NAME,
    resolve_subagent_type,
)
from .graph import build_runtime_graph
from .middleware import RuntimeMiddleware
from .paper_review_freshness import paper_review_manifest_status
from .prompt_builder import PromptBuilder
from .resume import (
    checkpoint_snapshot_path,
    create_runtime_checkpointer,
    export_checkpointer_snapshot,
    restore_checkpointer_snapshot,
)
from .state import (
    DEFAULT_RUNTIME_RECURSION_LIMIT,
    DEFAULT_STOP_HOOK_MAX_TRIGGERS,
    DEFAULT_STOP_HOOK_MODE,
    DEFAULT_STOP_HOOK_PROMPT,
    ensure_runtime_v2_state,
)
from .subagent_manager import is_retryable_subagent_infrastructure_error
from .tool_registry import SingleAgentTools

logger = logging.getLogger(__name__)

_DELETE_MISSING_MESSAGE_ERROR = "Attempting to delete a message with an ID that doesn't exist"
_REVIEW_DECISIONS = {"approve", "revise", "reject"}
_STOP_HOOK_DECISION_KEYS = {"decision", "result", "verdict", "ok", "pass", "needs_input"}
_SUBAGENT_SUBMIT_TOOL_NAMES = {
    SUBMIT_SUBAGENT_RESULT_TOOL_NAME,
    SUBMIT_PROPOSAL_REVIEW_TOOL_NAME,
    SUBMIT_RESEARCH_IDEA_REVIEW_TOOL_NAME,
    SUBMIT_IMPLEMENTATION_REVIEW_TOOL_NAME,
}
_STOP_HOOK_VERDICT_KEYS = {
    *_STOP_HOOK_DECISION_KEYS,
    "reason",
    "summary",
    "message",
    "question",
    "needs_input_question",
}
_PAPER_CONTEXT_TERMS = (
    "paper-authoring",
    "paper reviewer",
    "paper_reviewer",
    "paper_review",
    "outputs/paper",
    "/paper/",
    "main.tex",
    "main.pdf",
    "manuscript",
    "latex paper",
    "research paper",
    "论文",
    "文稿",
    "手稿",
)
_PAPER_COMPLETION_TERMS = (
    "complete",
    "completed",
    "finished",
    "done",
    "ready",
    "generated",
    "rewritten",
    "compiled",
    "analysis completed",
    "完成",
    "写完",
    "生成",
    "重写",
    "编译",
    "结束",
)


def _subagent_submit_tool_call_names(response: BaseMessage) -> List[str]:
    names: List[str] = []
    for call in getattr(response, "tool_calls", None) or []:
        name = str((call or {}).get("name") or "").strip()
        if name in _SUBAGENT_SUBMIT_TOOL_NAMES:
            names.append(name)
    return names


def _coerce_review_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [text]


def _parse_json_object_fragment(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    start = raw.find("{")
    if start < 0:
        return {}
    try:
        parsed, _ = json.JSONDecoder().raw_decode(raw[start:])
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _parse_review_value(value: str) -> Any:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw[:1] in {"[", "{"}:
        try:
            return json.loads(raw)
        except Exception:
            return raw
    return raw


def _extract_flat_review_fields(mapping: Dict[str, Any], review_key: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    prefixes = (
        f"{review_key}.",
        f"proposed_state_updates.{review_key}.",
    )
    for key, value in mapping.items():
        key_text = str(key or "").strip()
        for prefix in prefixes:
            if key_text.startswith(prefix):
                field = key_text[len(prefix) :].strip()
                if field:
                    payload[field] = value
                break
    return payload


def _extract_review_payload(result: Dict[str, Any], review_key: str) -> Dict[str, Any]:
    proposed = result.get("proposed_state_updates")
    if isinstance(proposed, dict):
        nested = proposed.get(review_key)
        if isinstance(nested, dict):
            return dict(nested)
        flattened = _extract_flat_review_fields(proposed, review_key)
        if flattened:
            return flattened

    direct = result.get(review_key)
    if isinstance(direct, dict):
        return dict(direct)

    flattened = _extract_flat_review_fields(result, review_key)
    if flattened:
        return flattened

    payload: Dict[str, Any] = {}
    text_items = [str(item or "") for item in (result.get("findings") or [])]
    if result.get("summary"):
        text_items.append(str(result.get("summary") or ""))
    json_markers = (
        f"proposed_state_updates.{review_key}=",
        f"{review_key}=",
    )
    field_markers = (
        f"proposed_state_updates.{review_key}.",
        f"{review_key}.",
    )
    for text in text_items:
        stripped = text.strip()
        for marker in json_markers:
            marker_index = stripped.find(marker)
            if marker_index >= 0:
                candidate = _parse_json_object_fragment(stripped[marker_index + len(marker) :])
                if candidate:
                    payload.update(candidate)
        for line in stripped.splitlines() or [stripped]:
            candidate_line = line.strip()
            for marker in field_markers:
                if not candidate_line.startswith(marker) or "=" not in candidate_line:
                    continue
                field, value = candidate_line[len(marker) :].split("=", 1)
                field = field.strip()
                if field:
                    payload[field] = _parse_review_value(value)
                break
    return payload


def _validate_structured_review_result(
    result: Dict[str, Any],
    review_key: str,
    *,
    required_text_fields: tuple[str, ...] = ("reviewer_feedback",),
    required_nonempty_list_fields: tuple[str, ...] = (),
    required_present_fields: tuple[str, ...] = (),
) -> tuple[Dict[str, Any], List[str]]:
    payload = _extract_review_payload(result, review_key)
    errors: List[str] = []
    if str(result.get("status") or "").strip().lower() != "completed":
        errors.append(f"subagent status is {result.get('status') or 'missing'}, expected completed")

    decision = str(payload.get("decision") or "").strip().lower()
    if decision not in _REVIEW_DECISIONS:
        errors.append("missing or invalid decision; expected approve|revise|reject")

    for field in required_text_fields:
        if not str(payload.get(field) or "").strip():
            errors.append(f"missing or empty `{field}`")

    for field in required_nonempty_list_fields:
        if not _coerce_review_list(payload.get(field)):
            errors.append(f"missing or empty `{field}`")

    for field in required_present_fields:
        if field not in payload:
            errors.append(f"missing `{field}`")

    return payload, errors


def _render_structured_retry_evidence(result: Dict[str, Any], review_key: str, *, max_chars: int = 12000) -> str:
    proposed_updates = result.get("proposed_state_updates") or {}
    evidence = {
        "status": result.get("status"),
        "subagent_id": result.get("subagent_id"),
        "summary": result.get("summary"),
        "findings": result.get("findings") or [],
        "artifact_refs": result.get("artifact_refs") or [],
        "previous_structured_payload": proposed_updates.get(review_key),
        "needs_input_question": result.get("needs_input_question") or "",
    }
    try:
        text = json.dumps(evidence, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = str(evidence)
    if len(text) > max_chars:
        return text[:max_chars] + "\n... [truncated previous subagent result]"
    return text


def _is_subagent_infrastructure_failure(result: Dict[str, Any]) -> bool:
    if str(result.get("status") or "").strip().lower() != "failed":
        return False
    text = "\n".join(
        [
            str(result.get("summary") or ""),
            "\n".join(str(item or "") for item in (result.get("findings") or [])),
        ]
    )
    return is_retryable_subagent_infrastructure_error(text)


def _message_text(message: Any) -> str:
    return strip_tool_call_transcript(getattr(message, "content", ""))


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    haystack = str(text or "").lower()
    return any(term.lower() in haystack for term in terms)


def _runtime_recursion_limit(
    max_turns: Optional[int] = None,
    *,
    enforce_turn_bound: bool = True,
) -> int:
    del max_turns, enforce_turn_bound
    raw = os.getenv("CYTOBRIDGE_RUNTIME_RECURSION_LIMIT", "").strip()
    if not raw:
        value = DEFAULT_RUNTIME_RECURSION_LIMIT
    else:
        try:
            value = int(raw)
        except ValueError:
            value = DEFAULT_RUNTIME_RECURSION_LIMIT
    return max(20, min(value, DEFAULT_RUNTIME_RECURSION_LIMIT))


class CytoBridgeAgent:
    def __init__(
        self,
        llm: ChatOpenAI,
        state: Dict[str, Any],
        checkpoint_callback: Optional[Callable[[], None]] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        *,
        agent_role: str = "planner",
        agent_id: str = "planner",
        parent_agent_id: str = "",
        subagent_type: str = "",
        tool_policy: Optional[Dict[str, Any]] = None,
        require_structured_result: bool = False,
        thread_id: Optional[str] = None,
        checkpoint_ns: Optional[str] = None,
    ):
        self.llm = llm
        self.state = ensure_runtime_v2_state(state)
        self.checkpoint_callback = checkpoint_callback
        self.event_callback = event_callback
        self.agent_role = str(agent_role or "planner").strip().lower() or "planner"
        default_agent_id = "planner" if self.agent_role == "planner" else str(agent_id or "subagent").strip()
        self.agent_id = str(agent_id or default_agent_id).strip() or default_agent_id
        self.parent_agent_id = str(parent_agent_id or "").strip()
        self.tool_policy = dict(tool_policy or {})
        self.subagent_type = ""
        if self.agent_role == "subagent":
            self.subagent_type = resolve_subagent_type(
                str(subagent_type or self.tool_policy.get("subagent_type") or "general").strip().lower() or "general"
            ).name
            self.tool_policy.setdefault("subagent_type", self.subagent_type)
        self.require_structured_result = bool(require_structured_result)
        self.stop_check = None
        self.display = DisplayManager()
        self.chat_history: List[BaseMessage] = []
        default_thread_id = self.state.get("session_id") or "runtime-v2-default"
        if self.agent_role == "subagent":
            default_thread_id = self.agent_id or default_thread_id
        self.thread_id = str(thread_id or default_thread_id)
        self.checkpoint_ns = str(
            checkpoint_ns
            or ("runtime_v2" if self.agent_role == "planner" else (self.agent_id or "runtime_v2_subagent"))
        )
        self.checkpointer = create_runtime_checkpointer(self.state)
        self._seed_full_history_next_run = False

        self.tools_handler = SingleAgentTools(
            llm,
            self.state,
            event_callback=self._emit_event,
            checkpoint_callback=self.checkpoint_callback,
            agent_role=self.agent_role,
            agent_id=self.agent_id,
            parent_agent_id=self.parent_agent_id,
            subagent_type=self.subagent_type,
            tool_policy=self.tool_policy,
        )
        self.tools_handler.set_planner_history_provider(lambda: self.chat_history)
        self.tools = []
        self.llm_with_tools = self.llm
        self.prompt_builder = PromptBuilder(self.state, event_sink=self._emit_event)
        self.middleware = RuntimeMiddleware(self.state, self.llm, event_sink=self._emit_event)
        self.graph = None
        self.refresh_tooling()

    def refresh_tooling(self) -> None:
        self.tools = self.tools_handler.get_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = build_runtime_graph(
            self.agent_node,
            self.tools,
            post_tool_hook_node=self.post_tool_hook_node,
            stop_hook_node=self.stop_hook_node if self._stop_hook_enabled() else None,
            checkpointer=self.checkpointer,
        )

    def _stop_hook_enabled(self) -> bool:
        if self.agent_role != "planner":
            return False
        raw = self.state.get("stop_hook_enabled", False)
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in {"", "0", "false", "no", "off"}:
            return False
        if text in {"1", "true", "yes", "on"}:
            return True
        return bool(raw)

    def _stop_hook_mode(self) -> str:
        if not self._stop_hook_enabled():
            return "builtin"
        raw = str(self.state.get("stop_hook_mode") or DEFAULT_STOP_HOOK_MODE).strip().lower()
        if raw in {"prompt", "llm"}:
            return "prompt"
        if raw in {"agent", "subagent"}:
            return "agent"
        if raw in {"builtin", "default"}:
            return "builtin"
        return DEFAULT_STOP_HOOK_MODE

    def _stop_hook_prompt(self) -> str:
        return str(self.state.get("stop_hook_prompt") or DEFAULT_STOP_HOOK_PROMPT).strip()

    def set_thread_id(self, thread_id: str) -> None:
        self.thread_id = str(thread_id or "runtime-v2-default")

    def set_checkpoint_namespace(self, checkpoint_ns: str) -> None:
        self.checkpoint_ns = str(checkpoint_ns or "runtime_v2")

    def _replace_checkpointer(self, checkpoint_path: Optional[str] = None) -> None:
        self.checkpointer = create_runtime_checkpointer(self.state, checkpoint_path=checkpoint_path)
        self.refresh_tooling()

    def _reset_checkpoint_thread(self) -> None:
        counter = int(self.state.get("langgraph_checkpoint_reset_counter", 0)) + 1
        self.state["langgraph_checkpoint_reset_counter"] = counter
        base = str(self.state.get("session_id") or self.thread_id or "runtime-v2-default")
        if self.agent_role == "subagent":
            base = str(self.agent_id or base)
        safe_base = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in base)
        self.thread_id = f"{safe_base}_checkpoint_reset_{counter}"

    def export_runtime_snapshot(self) -> Dict[str, Any]:
        return {
            "langgraph_checkpoint": export_checkpointer_snapshot(self.checkpointer),
            "thread_id": self.thread_id,
            "checkpoint_ns": self.checkpoint_ns,
            "history_revision": int(self.state.get("history_revision", 0)),
            "agent_role": self.agent_role,
            "agent_id": self.agent_id,
            "parent_agent_id": self.parent_agent_id,
            "subagent_type": self.subagent_type,
            "tool_policy": dict(self.tool_policy),
            "require_structured_result": self.require_structured_result,
        }

    def restore_runtime_snapshot(self, snapshot: Optional[Dict[str, Any]]) -> bool:
        if not snapshot:
            return False
        snapshot_revision = snapshot.get("history_revision")
        if snapshot_revision is not None:
            try:
                if int(snapshot_revision) != int(self.state.get("history_revision", 0)):
                    return False
            except Exception:
                return False
        thread_id = str(snapshot.get("thread_id") or "").strip()
        checkpoint_ns = str(snapshot.get("checkpoint_ns") or "").strip()
        if thread_id:
            self.thread_id = thread_id
        if checkpoint_ns:
            self.checkpoint_ns = checkpoint_ns
        if self.agent_role == "subagent":
            self.subagent_type = resolve_subagent_type(
                str(snapshot.get("subagent_type") or self.tool_policy.get("subagent_type") or "general").strip().lower()
                or "general"
            ).name
            self.tool_policy["subagent_type"] = self.subagent_type
            self.tools_handler.subagent_type = self.subagent_type
            self.tools_handler.tool_policy = dict(self.tool_policy)
            self.refresh_tooling()
        sqlite_checkpoint_path = checkpoint_snapshot_path(snapshot.get("langgraph_checkpoint"))
        if sqlite_checkpoint_path:
            current_path = str(getattr(self.checkpointer, "_cytobridge_checkpoint_path", "") or "")
            if current_path != sqlite_checkpoint_path:
                self._replace_checkpointer(sqlite_checkpoint_path)
        restored = restore_checkpointer_snapshot(self.checkpointer, snapshot.get("langgraph_checkpoint"))
        if restored:
            self._seed_full_history_next_run = False
        return restored

    def replace_chat_history(self, messages: List[BaseMessage], *, reset_checkpoint: bool = False) -> None:
        self.chat_history = list(messages or [])
        if reset_checkpoint:
            self._reset_checkpoint_thread()
            self._replace_checkpointer()
            self._seed_full_history_next_run = bool(self.chat_history)

    def abort_interrupted_turn(self, reason: str = "Stopped by user.") -> None:
        """Mark an interrupted turn as stopped and reseed the next graph run."""
        if getattr(self.llm, "_llm_type", "") == "codex-oauth-sidecar":
            try:
                from ..utils.codex_sidecar_client import reset_codex_sidecar_client

                reset_codex_sidecar_client()
            except Exception:
                logger.debug("Failed to reset Codex sidecar after interrupted turn", exc_info=True)
        stop_msg = AIMessage(
            content=(
                f"{reason} The interrupted turn was stopped by the user."
            ),
            additional_kwargs={"phase": PHASE_FINAL_ANSWER},
        )
        self.replace_chat_history(add_messages(list(self.chat_history or []), [stop_msg]), reset_checkpoint=True)
        self.state["messages"] = [self._serialize_message(m) for m in self.chat_history]

    def checkpoint_messages(self) -> List[BaseMessage]:
        candidates = [
            {"configurable": {"thread_id": self.thread_id, "checkpoint_ns": self.checkpoint_ns}},
        ]
        candidates.extend(
            [
                {"configurable": {"thread_id": self.thread_id, "checkpoint_ns": ""}},
                {"configurable": {"thread_id": self.thread_id}},
            ]
        )
        for config in candidates:
            try:
                snapshot = self.graph.get_state(config)
            except Exception:
                continue
            values = getattr(snapshot, "values", None) or {}
            messages = values.get("messages")
            if messages:
                return list(messages)
        return []

    def _apply_message_updates(
        self,
        new_messages: List[BaseMessage],
        fallback_base_messages: Optional[List[BaseMessage]] = None,
    ) -> List[BaseMessage]:
        try:
            self.chat_history = add_messages(self.chat_history, new_messages)
            return list(self.chat_history)
        except ValueError as exc:
            text = str(exc)
            if _DELETE_MISSING_MESSAGE_ERROR not in text:
                raise

        fallback_messages = list(fallback_base_messages or [])
        if not fallback_messages:
            fallback_messages = self.checkpoint_messages()
        if fallback_messages:
            self.chat_history = add_messages(list(fallback_messages), new_messages)
            self._emit_event(
                "status",
                {
                    "agent": self.agent_id,
                    "message": "Recovered local chat history from checkpoint after compaction/checkpoint divergence.",
                },
            )
            return list(self.chat_history)
        raise

    @staticmethod
    def _is_delete_missing_message_error(exc: Exception) -> bool:
        return _DELETE_MISSING_MESSAGE_ERROR in str(exc)

    def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        enriched = dict(payload or {})
        enriched.setdefault("agent_id", self.agent_id)
        enriched.setdefault("agent_role", self.agent_role)
        if self.parent_agent_id:
            enriched.setdefault("parent_agent_id", self.parent_agent_id)
        if not (self.agent_role == "subagent" and self.event_callback is not None):
            try:
                self.display._emit(event_type, enriched)
            except Exception:
                logger.debug("runtime_v2 display event emit failed", exc_info=True)
        if self.event_callback:
            try:
                self.event_callback(event_type, enriched)
            except Exception:
                logger.debug("runtime_v2 external event callback failed", exc_info=True)

    def _get_system_message(self, prompt_text: Optional[str] = None) -> SystemMessage:
        return SystemMessage(
            content=prompt_text
            if prompt_text is not None
            else self.prompt_builder.build(
                agent_role=self.agent_role,
                agent_id=self.agent_id,
                parent_agent_id=self.parent_agent_id,
                subagent_type=self.subagent_type,
                tool_policy=self.tool_policy,
            )
        )

    @staticmethod
    def _with_phase(response: BaseMessage, phase: str) -> AIMessage:
        content = strip_tool_call_transcript(getattr(response, "content", ""))
        kwargs = dict(getattr(response, "additional_kwargs", {}) or {})
        kwargs["phase"] = phase
        return AIMessage(content=content, additional_kwargs=kwargs, tool_calls=getattr(response, "tool_calls", None) or [])

    def _emit_tool_call_commentary(self, response: BaseMessage, phase: str) -> None:
        """Surface assistant text that accompanies tool calls without changing history semantics."""
        if phase != PHASE_COMMENTARY:
            return
        if not (getattr(response, "tool_calls", None) or []):
            return
        content = strip_tool_call_transcript(getattr(response, "content", ""))
        if not content:
            return
        self._emit_event(
            "agent_thought",
            {
                "agent": "cytobridge" if self.agent_role == "planner" else self.agent_id,
                "content": content,
                "phase": phase,
                "source": "tool_call_response_text",
            },
        )

    @staticmethod
    def _build_user_message_content(text: str, attachments: Optional[List[Dict[str, Any]]] = None) -> Any:
        normalized_attachments = list(attachments or [])
        if not normalized_attachments:
            return text

        blocks: List[Dict[str, Any]] = []
        if text.strip():
            blocks.append({"type": "text", "text": text})
        else:
            blocks.append({"type": "text", "text": "[User uploaded image attachment(s)]"})

        for idx, item in enumerate(normalized_attachments):
            mime_type = str(item.get("mime_type") or item.get("mimeType") or "").strip()
            content = str(item.get("content") or "").strip()
            if not mime_type or not content:
                continue
            file_name = str(item.get("file_name") or item.get("fileName") or f"image_{idx + 1}").strip() or f"image_{idx + 1}"
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{content}",
                        "detail": "auto",
                    },
                    "metadata": {
                        "file_name": file_name,
                        "mime_type": mime_type,
                    },
                }
            )
        return blocks if blocks else text

    @staticmethod
    def _render_stop_hook_content(value: Any) -> str:
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = str(item.get("text") or "").strip()
                    if text:
                        parts.append(text)
                else:
                    text = str(item).strip()
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()
        return str(value or "").strip()

    @staticmethod
    def _render_stop_hook_message_role(message: BaseMessage) -> str:
        if isinstance(message, HumanMessage):
            return "user"
        if isinstance(message, AIMessage):
            return "assistant"
        if isinstance(message, SystemMessage):
            return "system"
        return message.__class__.__name__.replace("Message", "").lower() or "message"

    def _render_stop_hook_transcript(self, messages: List[BaseMessage], *, limit: int = 6) -> str:
        tail = list(messages or [])[-max(1, limit) :]
        lines: List[str] = []
        for idx, message in enumerate(tail, start=1):
            role = self._render_stop_hook_message_role(message)
            content = strip_tool_call_transcript(self._render_stop_hook_content(getattr(message, "content", "")))
            if not content:
                tool_calls = getattr(message, "tool_calls", None) or []
                content = f"[tool calls: {len(tool_calls)}]" if tool_calls else "[empty]"
            lines.append(f"{idx}. {role}: {content}")
        return "\n".join(lines).strip()

    @staticmethod
    def _truncate_stop_hook_json(value: Any, *, max_chars: int = 12000) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception:
            text = str(value)
        if len(text) > max_chars:
            return text[:max_chars] + "\n... [truncated stop-hook context]"
        return text

    @staticmethod
    def _summarize_stop_hook_stage(stage_name: str, stage_state: Dict[str, Any], status_view: Dict[str, Any]) -> Dict[str, Any]:
        last_gate = dict(stage_state.get("last_gate_check") or {})
        stage_panel = stage_state.get("stage_panel") if isinstance(stage_state.get("stage_panel"), dict) else {}
        target_dataset_ids = [
            str(item or "").strip()
            for item in list(stage_panel.get("target_dataset_ids") or status_view.get("target_dataset_ids") or [])
            if str(item or "").strip()
        ]
        return {
            "stage": stage_name,
            "stage_status": str(status_view.get("stage_status") or stage_state.get("status") or ""),
            "stage_internal_status": str(status_view.get("stage_internal_status") or ""),
            "stage_gate_status": str(status_view.get("stage_gate_status") or ""),
            "gate_ready": bool(status_view.get("gate_ready") or stage_state.get("gate_ready", False)),
            "gate_passed_at": str(stage_state.get("gate_passed_at") or ""),
            "gate_passed_trial_id": str(stage_state.get("gate_passed_trial_id") or ""),
            "active_best_trial_id": str(status_view.get("active_best_trial_id") or stage_state.get("active_best_trial_id") or ""),
            "current_trial_id": str(status_view.get("current_trial_id") or ""),
            "target_dataset_ids": target_dataset_ids,
            "trial_count": int(stage_state.get("trial_count") or 0),
            "promote_count": int(stage_state.get("promote_count") or 0),
            "reject_count": int(stage_state.get("reject_count") or 0),
            "last_gate_ok": bool(last_gate.get("ok", False)),
            "last_gate_blockers": [str(item) for item in list(last_gate.get("blockers") or []) if str(item).strip()],
            "user_facing_status": str(status_view.get("user_facing_status") or ""),
            "next_required_action": str(status_view.get("next_required_action") or ""),
        }

    def _stop_hook_campaign_snapshot(self) -> Dict[str, Any]:
        active_context = dict(self.state.get("active_algorithm_context") or {})
        active_algorithm_id = str(active_context.get("algorithm_id") or "").strip()
        active_lifecycle_status = (
            str(active_context.get("algorithm_lifecycle_status") or "developing")
            if active_algorithm_id
            else ""
        )
        snapshot: Dict[str, Any] = {
            "active_algorithm_context": {
                "algorithm_id": active_algorithm_id,
                "algorithm_lifecycle_status": active_lifecycle_status,
                "algorithm_lifecycle_status_reason": str(active_context.get("algorithm_lifecycle_status_reason") or ""),
                "completed_campaign_id": str(active_context.get("completed_campaign_id") or ""),
                "proposal_id": str(active_context.get("proposal_id") or ""),
                "proposal_status": str(active_context.get("proposal_status") or ""),
                "workspace_path": str(active_context.get("workspace_path") or ""),
                "active_snapshot_id": str(active_context.get("active_snapshot_id") or ""),
                "dirty_since_snapshot": bool(active_context.get("dirty_since_snapshot", False)),
                "dirty_paths": list(active_context.get("dirty_paths") or []),
            },
            "active_campaign": {},
        }

        file_tools = getattr(getattr(self, "tools_handler", None), "planner_file_tools", None)
        campaign_resolution = resolve_active_campaign(self.state, file_tools=file_tools)
        campaign_id = str(campaign_resolution.get("campaign_id") or "").strip()
        campaign: Dict[str, Any] = dict(campaign_resolution.get("campaign") or {})
        if not campaign:
            snapshot["campaign_resolution"] = {key: value for key, value in campaign_resolution.items() if key != "campaign"}
            return snapshot

        stages = campaign.get("stages") if isinstance(campaign.get("stages"), dict) else {}
        status_views = campaign.get("stage_statuses") if isinstance(campaign.get("stage_statuses"), dict) else {}
        stage_order = [
            str(item or "").strip()
            for item in list(campaign.get("stage_order") or CAMPAIGN_STAGE_ORDER)
            if str(item or "").strip()
        ]
        stage_summary: Dict[str, Any] = {}
        for stage_name in stage_order:
            stage_summary[stage_name] = self._summarize_stop_hook_stage(
                stage_name,
                dict(stages.get(stage_name) or {}),
                dict(status_views.get(stage_name) or {}),
            )

        snapshot["active_campaign"] = {
            "campaign_id": str(campaign.get("campaign_id") or campaign_id),
            "algorithm_id": str(campaign.get("algorithm_id") or ""),
            "status": str(campaign.get("status") or ""),
            "algorithm_lifecycle_status": str(campaign.get("algorithm_lifecycle_status") or active_context.get("algorithm_lifecycle_status") or "developing"),
            "algorithm_lifecycle_status_reason": str(campaign.get("algorithm_lifecycle_status_reason") or active_context.get("algorithm_lifecycle_status_reason") or ""),
            "completed_campaign_id": str(campaign.get("completed_campaign_id") or active_context.get("completed_campaign_id") or ""),
            "current_stage": str(campaign.get("current_stage") or ""),
            "current_trial_id": str(campaign.get("current_trial_id") or ""),
            "algorithm_validated": bool(campaign.get("algorithm_validated", False)),
            "locked_release": dict(campaign.get("locked_release") or {}),
            "stage_order": stage_order,
            "current_stage_status": dict(campaign.get("current_stage_status") or {}),
            "stage_statuses": stage_summary,
            "completion_criteria": {
                "algorithm_lifecycle_status": "must be complete for algorithm-lifecycle tasks; developing or failed is not completion",
                "stage1_feasibility": "formal gate passed and advanced/persisted",
                "stage2_claim_validation": "formal gate passed; trial promotion alone is insufficient",
                "stage3_tuning": "completed or intentionally advanced according to policy",
                "final_regression": "final result exists and campaign is locked/released",
            },
        }
        snapshot["campaign_resolution"] = {key: value for key, value in campaign_resolution.items() if key != "campaign"}
        snapshot["lifecycle_state"] = summarize_algorithm_lifecycle(
            state_with_resolved_campaign(
                {
                    **dict(self.state or {}),
                    "active_algorithm_context": active_context,
                },
                campaign_resolution,
            )
        )
        return snapshot

    def _render_stop_hook_context_snapshot(self) -> str:
        return self._truncate_stop_hook_json(self._stop_hook_campaign_snapshot())

    @staticmethod
    def _extract_json_payload(text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if not raw:
            raise ValueError("empty stop-hook response")
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            decoder = json.JSONDecoder()
            candidates: List[Dict[str, Any]] = []
            for index, char in enumerate(raw):
                if char != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(raw[index:])
                except Exception:
                    continue
                if isinstance(candidate, dict):
                    candidates.append(candidate)
            if not candidates:
                raise
            for candidate in candidates:
                if any(key in candidate for key in _STOP_HOOK_DECISION_KEYS):
                    parsed = candidate
                    break
            else:
                parsed = candidates[0]
        if not isinstance(parsed, dict):
            raise ValueError("stop-hook response must be a JSON object")
        return parsed

    @staticmethod
    def _has_stop_hook_decision_payload(payload: Dict[str, Any]) -> bool:
        return any(key in payload for key in _STOP_HOOK_DECISION_KEYS)

    @classmethod
    def _extract_stop_hook_decision_payload(cls, text: str) -> Dict[str, Any]:
        payload = cls._extract_json_payload(text)
        if not cls._has_stop_hook_decision_payload(payload):
            raise ValueError("stop-hook response JSON is missing decision/result/verdict/ok/pass/needs_input")
        return payload

    def _looks_like_stop_hook_verdict_response(self, content: Any) -> bool:
        if not self._stop_hook_enabled():
            return False
        text = strip_tool_call_transcript(content)
        if not text:
            return False
        try:
            payload = self._extract_json_payload(text)
        except Exception:
            return False
        normalized = self._normalize_stop_hook_decision(payload)
        if normalized["decision"] not in {"pass", "block", "needs_input"}:
            return False
        keys = {str(key) for key in payload.keys()}
        return bool(keys) and keys.issubset(_STOP_HOOK_VERDICT_KEYS)

    def _latest_paper_review_payload(self) -> Dict[str, Any]:
        registry = self.state.get("subagent_registry") or {}
        if not isinstance(registry, dict):
            return {}

        candidates: List[Dict[str, Any]] = []
        for entry in registry.values():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("subagent_type") or "").strip().lower() != "paper_reviewer":
                continue
            if str(entry.get("status") or "").strip().lower() != "completed":
                continue
            result = entry.get("last_result")
            if not isinstance(result, dict):
                continue
            review = _extract_review_payload(result, "paper_review")
            if review:
                candidates.append(
                    {
                        "finished_at": str(entry.get("finished_at") or entry.get("updated_at") or ""),
                        "review": review,
                    }
                )

        if not candidates:
            return {}
        candidates.sort(key=lambda item: item.get("finished_at") or "")
        return dict(candidates[-1].get("review") or {})

    def _has_approved_paper_review(self) -> bool:
        review = self._latest_paper_review_payload()
        decision = str(review.get("decision") or "").strip().lower()
        if decision != "approve":
            return False
        if _coerce_review_list(review.get("blocking_issues")):
            return False
        return bool(paper_review_manifest_status(review.get("reviewed_file_hashes")).get("fresh"))

    def _paper_review_guard_reason(self) -> str:
        review = self._latest_paper_review_payload()
        if not review:
            return "the current paper has no completed `paper_reviewer` result in session state"
        decision = str(review.get("decision") or "").strip().lower()
        if decision != "approve":
            return f"the latest `paper_reviewer` decision is `{decision or 'missing'}`, not `approve`"
        blocking = _coerce_review_list(review.get("blocking_issues"))
        if blocking:
            return "the latest `paper_reviewer` approval has blocking issues"
        freshness = paper_review_manifest_status(review.get("reviewed_file_hashes"))
        if not bool(freshness.get("fresh")):
            return str(freshness.get("reason") or "the latest reviewer approval is stale")
        return "the latest reviewer approval is fresh"

    def _should_guard_paper_completion_without_review(
        self,
        *,
        messages: List[BaseMessage],
        response: BaseMessage,
    ) -> bool:
        if self.agent_role != "planner":
            return False
        if getattr(response, "tool_calls", None):
            return False
        if resolve_message_phase(response) != PHASE_FINAL_ANSWER:
            return False
        if self._has_approved_paper_review():
            return False

        response_text = _message_text(response)
        recent_text = "\n".join(_message_text(message) for message in list(messages or [])[-8:])
        context_text = f"{recent_text}\n{response_text}"
        if not _contains_any(context_text, _PAPER_CONTEXT_TERMS):
            return False
        return _contains_any(response_text, _PAPER_COMPLETION_TERMS)

    def _paper_reviewer_guard_message(self) -> AIMessage:
        reason = self._paper_review_guard_reason()
        return AIMessage(
            content=(
                "Paper-authoring completion is blocked because "
                f"{reason}. Do not report the paper as complete yet. "
                "Spawn a read-only `paper_reviewer` subagent for the current `outputs/paper` files, inspect "
                "its `proposed_state_updates.paper_review.decision`, revise if needed, and only finish after "
                "the reviewer returns `approve` with no blocking issues and a fresh file-hash manifest."
            ),
            additional_kwargs={
                "phase": PHASE_COMMENTARY,
                "hook_stage": "agent_node",
                "hook_name": "paper_reviewer_completion_guard",
            },
        )

    @staticmethod
    def _normalize_stop_hook_decision(data: Any) -> Dict[str, str]:
        payload = dict(data or {}) if isinstance(data, dict) else {}
        decision_raw = str(
            payload.get("decision")
            or payload.get("result")
            or payload.get("verdict")
            or ""
        ).strip().lower()
        if not decision_raw:
            if bool(payload.get("needs_input")):
                decision_raw = "needs_input"
            elif payload.get("ok") is True or payload.get("pass") is True:
                decision_raw = "pass"
            elif payload.get("ok") is False:
                decision_raw = "block"

        if decision_raw in {"pass", "allow", "accept", "approved"}:
            decision = "pass"
        elif decision_raw in {"needs_input", "ask_user", "request_input", "question"}:
            decision = "needs_input"
        else:
            decision = "block"

        reason = str(
            payload.get("reason")
            or payload.get("summary")
            or payload.get("message")
            or ""
        ).strip()
        question = str(
            payload.get("question")
            or payload.get("needs_input_question")
            or ""
        ).strip()
        return {
            "decision": decision,
            "reason": reason,
            "question": question,
        }

    def _validate_stop_hook_subagent_result(self, result: Dict[str, Any]) -> tuple[Dict[str, str], List[str]]:
        proposed_raw = (result.get("proposed_state_updates") or {}).get("stop_hook") if isinstance(result, dict) else None
        proposed = dict(proposed_raw or {}) if isinstance(proposed_raw, dict) else {}
        errors: List[str] = []
        if str(result.get("status") or "").strip().lower() != "completed":
            errors.append(f"subagent status is {result.get('status') or 'missing'}, expected completed")
        if not proposed:
            errors.append("missing `proposed_state_updates.stop_hook`")
        if "decision" not in proposed:
            errors.append("missing `proposed_state_updates.stop_hook.decision`")

        normalized = self._normalize_stop_hook_decision(proposed)
        if normalized["decision"] not in {"pass", "block", "needs_input"}:
            errors.append("invalid stop-hook decision")
        if not normalized["reason"] and not normalized["question"]:
            errors.append("missing stop-hook reason/question")
        return normalized, errors

    def _build_stop_hook_commentary_payload(
        self,
        *,
        reason: str,
        hook_name: str,
        hook_mode: str,
    ) -> Dict[str, Any]:
        reason_text = str(reason or "").strip()
        continuation_lines = [
            "Runtime stop-hook feedback:",
            reason_text or "The attempted terminal response does not satisfy the stop-hook completion criteria.",
            "",
            "Do not ask the user for confirmation or clarification in this autonomous stop-hook path.",
            "Do not restate this blocker as a final answer.",
            "Continue by taking the next concrete tool/action that can move the original user goal forward.",
            "If repeated trials, reviewer feedback, or baseline gaps suggest the current direction is wrong, do not blindly run another trial.",
            "Pause the trial loop long enough to inspect risk.md, compare against the relevant baseline, audit per-dataset/per-timepoint failures, and write a diagnosis artifact before changing the algorithm again.",
            "Diagnosis can isolate one component at a time using existing artifacts: coupling, conditional path, optimization, inference rollout/mass dynamics, metric/evaluator, data contract, or proposal theory.",
            "If the suspected failure is theoretical or proposal-level, do targeted theory/literature reading for that exact assumption, patch the proposal, and wait for review instead of continuing down the wrong path.",
            "If a reviewer/gate blocked progress, revise the proposal, implementation, config, or evidence using the existing feedback and rerun the relevant review/gate.",
            "If no feasible path remains after concrete attempts, write a failure lifecycle report with evidence instead of asking the user to choose.",
        ]
        message = (
            "\n".join(continuation_lines)
        )
        self.state["planner_phase"] = "working"
        self.state["planner_need"] = {}
        self._emit_event(
            "stop_hook_blocked",
            {
                "hook": hook_name,
                "hook_mode": hook_mode,
                "reason": reason_text,
            },
        )
        return {
            "messages": [
                HumanMessage(
                    content=message,
                    additional_kwargs={
                        "phase": PHASE_COMMENTARY,
                        "is_meta": True,
                        "internal": True,
                        "hook_stage": "stop_hook",
                        "hook_name": hook_name,
                        "hook_mode": hook_mode,
                        "reason": reason_text,
                    },
                )
            ],
            "commentary_retries": 0,
        }

    def _evaluate_prompt_stop_hook(
        self,
        *,
        messages: List[BaseMessage],
        last_message: BaseMessage,
        custom_prompt: str,
    ) -> Dict[str, Any]:
        planner_need = dict(self.state.get("planner_need") or {})
        phase = str(self.state.get("planner_phase") or "working").strip() or "working"
        recent_transcript = self._render_stop_hook_transcript(messages)
        latest_answer = strip_tool_call_transcript(self._render_stop_hook_content(getattr(last_message, "content", "")))
        context_snapshot = self._render_stop_hook_context_snapshot()
        review_messages: List[BaseMessage] = [
            SystemMessage(
                content=(
                    "You are a CytoBridge runtime stop-hook evaluator.\n"
                    "Decide whether the planner's latest terminal response can be accepted right now.\n"
                    "Use the structured campaign/workflow snapshot as evidence; do not rely only on the planner's prose.\n"
                    "For an automated custom algorithm lifecycle, Stage 1 alone is never completion. "
                    "A pass requires the structured algorithm_lifecycle_status to be complete, which is set only after "
                    "final_regression evidence / locked release. If algorithm_lifecycle_status is developing or failed, return block "
                    "so the planner continues the lifecycle, fixes/revises the algorithm, or designs a replacement, "
                    "unless the session-specific instructions explicitly define a narrower non-lifecycle task.\n"
                    "This hook is an autonomy guard: do not ask the user for confirmation or clarification. "
                    "If the planner is trying to ask the user before completing the requested goal, return block.\n"
                    "Return exactly one JSON object with keys:\n"
                    '`decision` ("pass" | "block") and `reason` (string).\n'
                    "Do not call tools. Do not return markdown. Do not add any text outside the JSON object."
                )
            ),
            HumanMessage(
                content=(
                    "Session-specific stop-hook instructions:\n"
                    f"{custom_prompt}\n\n"
                    f"Current planner_phase: {phase}\n"
                    f"Current workflow_phase: {self.state.get('workflow_phase') or 'intake'}\n"
                    f"Current planner_need: {json.dumps(planner_need, ensure_ascii=False)}\n\n"
                    "Structured workflow/campaign snapshot:\n"
                    f"{context_snapshot}\n\n"
                    "Recent transcript tail:\n"
                    f"{recent_transcript or '(empty)'}\n\n"
                    "Planner terminal response under review:\n"
                    f"{latest_answer or '(empty)'}"
                )
            ),
        ]
        response = invoke_with_retry(self.llm, review_messages, logger, "runtime_v2_stop_hook_prompt")
        response_text = self._render_stop_hook_content(getattr(response, "content", ""))
        try:
            parsed = self._extract_stop_hook_decision_payload(response_text)
        except Exception:
            repair_messages: List[BaseMessage] = [
                SystemMessage(
                    content=(
                        "You are repairing a CytoBridge stop-hook evaluator response.\n"
                        "The previous response did not follow the required contract. "
                        "Do not answer the user. Do not continue the planner task. "
                        "Return exactly one JSON object with keys `decision` and `reason`.\n"
                        "`decision` must be either \"pass\" or \"block\". "
                        "Use pass only if the planner final answer under review is acceptable under the original rubric."
                    )
                ),
                HumanMessage(
                    content=(
                        "Original stop-hook review request:\n"
                        f"{review_messages[-1].content}\n\n"
                        "Invalid previous stop-hook response:\n"
                        f"{response_text[:8000]}\n\n"
                        "Return JSON only."
                    )
                ),
            ]
            repaired = invoke_with_retry(self.llm, repair_messages, logger, "runtime_v2_stop_hook_prompt_repair")
            parsed = self._extract_stop_hook_decision_payload(
                self._render_stop_hook_content(getattr(repaired, "content", ""))
            )
        return self._normalize_stop_hook_decision(parsed)

    def _evaluate_agent_stop_hook(
        self,
        *,
        messages: List[BaseMessage],
        last_message: BaseMessage,
        custom_prompt: str,
    ) -> Dict[str, Any]:
        manager = getattr(self.tools_handler, "subagent_manager", None)
        if manager is None:
            raise RuntimeError("subagent manager unavailable for agent-mode stop hook")
        result = manager.run_sync(
            subagent_type="general",
            task=(
                "Evaluate whether the planner's latest terminal response should be accepted now. "
                "Use the session-specific stop-hook instructions below as the review rubric. "
                "Use the structured campaign/workflow snapshot as evidence; do not rely only on planner prose. "
                "For an automated custom algorithm lifecycle, Stage 1 alone is not completion; pass only when the "
                "structured algorithm_lifecycle_status is complete, which is set only after final_regression locked-release evidence. "
                "If the status is developing or failed, return block so the planner continues, fixes/revises the algorithm, "
                "or designs a replacement that satisfies the user goal. "
                "Before finishing, call submit_subagent_result(...) and place a dict at "
                "`proposed_state_updates.stop_hook` with keys `decision` and `reason`. "
                "Use decision=pass when the final answer can be accepted, decision=block when the planner must continue, "
                "and do not request user input."
            ),
            success_criteria=[
                "Assess the latest planner terminal response against the provided stop-hook instructions.",
                "Return a structured stop-hook verdict in proposed_state_updates.stop_hook.",
                "Use only pass or block; if the planner wants user input before the goal is complete, return block.",
                "Do not modify shared workflow state directly.",
            ],
            context_notes=(
                "Session-specific stop-hook instructions:\n"
                f"{custom_prompt}\n\n"
                f"Current workflow_phase: {self.state.get('workflow_phase') or 'intake'}\n"
                f"Current planner_phase: {self.state.get('planner_phase') or 'working'}\n\n"
                "Structured workflow/campaign snapshot:\n"
                f"{self._render_stop_hook_context_snapshot()}\n\n"
                "Recent transcript tail:\n"
                f"{self._render_stop_hook_transcript(messages) or '(empty)'}\n\n"
                "Planner terminal response under review:\n"
                f"{strip_tool_call_transcript(self._render_stop_hook_content(getattr(last_message, 'content', ''))) or '(empty)'}"
            ),
            relevant_paths=[],
        )
        normalized, errors = self._validate_stop_hook_subagent_result(result)
        if not errors:
            return normalized

        self._emit_event(
            "structured_subagent_result_retry",
            {
                "subagent_type": "general",
                "review_key": "stop_hook",
                "subagent_id": str(result.get("subagent_id") or ""),
                "errors": list(errors),
            },
        )
        retry_result = manager.run_sync(
            subagent_type="general",
            task=(
                "Repair the previous stop-hook evaluator output using the prior result context below. "
                "Do not answer the user. Return only through submit_subagent_result(...) with "
                "proposed_state_updates.stop_hook containing decision and reason."
            ),
            success_criteria=[
                "Return proposed_state_updates.stop_hook.decision as pass or block.",
                "Include a concise proposed_state_updates.stop_hook.reason.",
                "Do not request user input.",
            ],
            context_notes=(
                "STRUCTURED OUTPUT CONTRACT VIOLATION:\n"
                + "\n".join(f"- {error}" for error in errors)
                + "\n\n"
                "Original stop-hook instructions:\n"
                f"{custom_prompt}\n\n"
                "Previous invalid subagent result to reuse:\n"
                f"{_render_structured_retry_evidence(result, 'stop_hook', max_chars=8000)}"
            ),
            relevant_paths=[],
        )
        retry_normalized, retry_errors = self._validate_stop_hook_subagent_result(retry_result)
        if not retry_errors:
            return retry_normalized
        self._emit_event(
            "structured_subagent_result_invalid",
            {
                "subagent_type": "general",
                "review_key": "stop_hook",
                "subagent_id": str(retry_result.get("subagent_id") or ""),
                "errors": list(retry_errors),
            },
        )
        fallback = {
            "decision": "block",
            "reason": str(retry_result.get("summary") or result.get("summary") or "").strip()
            or "stop-hook evaluator did not return structured output",
            "question": str(retry_result.get("needs_input_question") or result.get("needs_input_question") or "").strip(),
        }
        return self._normalize_stop_hook_decision(fallback)

    def _run_structured_review_subagent(
        self,
        manager: Any,
        *,
        subagent_type: str,
        task: str,
        success_criteria: List[str],
        context_notes: str,
        relevant_paths: List[str],
        review_key: str,
        required_text_fields: tuple[str, ...] = ("reviewer_feedback",),
        required_nonempty_list_fields: tuple[str, ...] = (),
        required_present_fields: tuple[str, ...] = (),
        repair_instruction: str = "",
    ) -> Dict[str, Any]:
        result = manager.run_sync(
            subagent_type=subagent_type,
            task=task,
            success_criteria=list(success_criteria),
            context_notes=context_notes,
            relevant_paths=list(relevant_paths),
        )
        if _is_subagent_infrastructure_failure(result):
            self._emit_event(
                "structured_subagent_infrastructure_failed",
                {
                    "subagent_type": subagent_type,
                    "review_key": review_key,
                    "subagent_id": str(result.get("subagent_id") or ""),
                    "summary": str(result.get("summary") or "").strip(),
                },
            )
            return result
        _, errors = _validate_structured_review_result(
            result,
            review_key,
            required_text_fields=required_text_fields,
            required_nonempty_list_fields=required_nonempty_list_fields,
            required_present_fields=required_present_fields,
        )
        if not errors:
            return result

        self._emit_event(
            "structured_subagent_result_retry",
            {
                "subagent_type": subagent_type,
                "review_key": review_key,
                "subagent_id": str(result.get("subagent_id") or ""),
                "errors": list(errors),
            },
        )
        retry_context = (
            f"{context_notes}\n\n"
            "STRUCTURED OUTPUT CONTRACT VIOLATION:\n"
            + "\n".join(f"- {error}" for error in errors)
            + "\n\n"
            "Repair the review output now. Reuse the prior subagent analysis below; do not redo the full review "
            "unless the prior analysis is insufficient for the missing fields. Do not summarize free-form only. "
            "You must finish by calling the required structured submit tool and include every required field. "
            f"{repair_instruction}".strip()
            + "\n\n"
            "Previous invalid subagent result to reuse:\n"
            f"{_render_structured_retry_evidence(result, review_key)}"
        )
        retry_result = manager.run_sync(
            subagent_type=subagent_type,
            task=(
                "Repair the previous structured output for this already-run review. "
                "Use the prior result context; only inspect files again if needed to fill missing required fields.\n\n"
                f"Original task:\n{task}"
            ),
            success_criteria=list(success_criteria)
            + [
                "Repair the previous structured-output contract violation before returning.",
                "Reuse the prior review analysis instead of restarting from scratch when it is sufficient.",
            ],
            context_notes=retry_context,
            relevant_paths=list(relevant_paths),
        )
        if _is_subagent_infrastructure_failure(retry_result):
            self._emit_event(
                "structured_subagent_infrastructure_failed",
                {
                    "subagent_type": subagent_type,
                    "review_key": review_key,
                    "subagent_id": str(retry_result.get("subagent_id") or ""),
                    "summary": str(retry_result.get("summary") or "").strip(),
                    "during_structured_retry": True,
                },
            )
            return retry_result
        _, retry_errors = _validate_structured_review_result(
            retry_result,
            review_key,
            required_text_fields=required_text_fields,
            required_nonempty_list_fields=required_nonempty_list_fields,
            required_present_fields=required_present_fields,
        )
        if retry_errors:
            self._emit_event(
                "structured_subagent_result_invalid",
                {
                    "subagent_type": subagent_type,
                    "review_key": review_key,
                    "subagent_id": str(retry_result.get("subagent_id") or ""),
                    "errors": list(retry_errors),
                },
            )
        return retry_result

    def _run_custom_stop_hook(
        self,
        *,
        messages: List[BaseMessage],
        last_message: BaseMessage,
        trigger_count: int,
    ) -> Dict[str, Any]:
        mode = self._stop_hook_mode()
        if mode == "builtin":
            return {}
        custom_prompt = self._stop_hook_prompt()
        if not custom_prompt:
            return {}
        hook_name = f"custom_stop_hook_{mode}"
        try:
            decision = (
                self._evaluate_prompt_stop_hook(
                    messages=messages,
                    last_message=last_message,
                    custom_prompt=custom_prompt,
                )
                if mode == "prompt"
                else self._evaluate_agent_stop_hook(
                    messages=messages,
                    last_message=last_message,
                    custom_prompt=custom_prompt,
                )
            )
        except Exception as exc:
            logger.exception("Custom stop hook (%s) failed", mode)
            decision = {
                "decision": "block",
                "reason": f"Custom stop hook ({mode}) failed: {exc}",
                "question": "",
            }

        normalized = self._normalize_stop_hook_decision(decision)
        if normalized["decision"] == "needs_input":
            reason = normalized["reason"] or "stop_hook_requested_user_input"
            question = normalized["question"]
            if question:
                reason = f"{reason} Requested user input is not allowed for autonomous stop-hook mode: {question}"
            normalized = {
                "decision": "block",
                "reason": reason,
                "question": "",
            }
        self._emit_event(
            "stop_hook_custom_evaluated",
            {
                "hook": hook_name,
                "hook_mode": mode,
                "decision": normalized["decision"],
                "reason": normalized["reason"],
                "trigger_count": trigger_count,
            },
        )
        if normalized["decision"] == "pass":
            return {}
        if normalized["decision"] == "needs_input":
            question = normalized["question"] or normalized["reason"] or (
                "The stop hook requires additional user input before this run can finish."
            )
            reason = normalized["reason"] or "custom_stop_hook_requires_input"
            self.state["planner_phase"] = "needs_input"
            self.state["planner_need"] = {
                "source": "stop_hook",
                "question": question,
                "reason": reason,
                "hook_mode": mode,
                "hook_name": hook_name,
            }
            return self._build_needs_input_interrupt_payload(
                question=question,
                source="stop_hook",
                reason=reason,
                hook_stage="stop_hook",
                hook_name=hook_name,
            )
        return self._build_stop_hook_commentary_payload(
            reason=normalized["reason"],
            hook_name=hook_name,
            hook_mode=mode,
        )

    def agent_node(self, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        if self.stop_check and self.stop_check():
            return {
                "messages": [AIMessage(content="Stopped by user.", additional_kwargs={"phase": PHASE_FINAL_ANSWER})],
                "commentary_retries": 0,
            }

        turn_system_prompt = str(graph_state.get("turn_system_prompt") or "")
        prompt_messages: List[BaseMessage] = [self._get_system_message(turn_system_prompt)] + list(graph_state["messages"])
        prepared = self.middleware.prepare_messages(prompt_messages)
        response = invoke_with_retry(self.llm_with_tools, prepared.prompt_messages, logger, "runtime_v2")
        self._record_llm_context_usage(response)
        blocked_submit_tools = _subagent_submit_tool_call_names(response) if self.agent_role == "planner" else []
        if blocked_submit_tools:
            self._emit_event(
                "internal_subagent_tool_call_guarded",
                {
                    "agent_id": self.agent_id,
                    "agent_role": self.agent_role,
                    "tool_names": blocked_submit_tools,
                    "reason": "subagent_submit_tool_call_from_planner",
                },
            )
            response = AIMessage(
                content=(
                    "A reviewer-only subagent submit tool was produced in the planner response. "
                    "Ignore that stale/internal tool call and answer the current user request directly."
                ),
                additional_kwargs={"phase": PHASE_COMMENTARY},
            )
        if (
            not (getattr(response, "tool_calls", None) or [])
            and self._looks_like_stop_hook_verdict_response(getattr(response, "content", ""))
        ):
            self._emit_event(
                "internal_stop_hook_verdict_guarded",
                {
                    "agent_id": self.agent_id,
                    "agent_role": self.agent_role,
                    "reason": "assistant_output_looked_like_internal_stop_hook_verdict",
                    "content": str(getattr(response, "content", "") or "")[:500],
                },
            )
            response = AIMessage(
                content=(
                    "An internal stop-hook verdict was emitted as assistant text. "
                    "Do not expose stop-hook JSON to the user; continue with a normal user-facing answer."
                ),
                additional_kwargs={"phase": PHASE_COMMENTARY},
            )
        paper_reviewer_guarded = False
        if self._should_guard_paper_completion_without_review(messages=list(graph_state["messages"]), response=response):
            self._emit_event(
                "paper_reviewer_completion_guarded",
                {
                    "agent_id": self.agent_id,
                    "agent_role": self.agent_role,
                    "reason": "paper_authoring_final_answer_without_reviewer_approval",
                    "content": str(getattr(response, "content", "") or "")[:500],
                },
            )
            response = self._paper_reviewer_guard_message()
            paper_reviewer_guarded = True
        phase = resolve_message_phase(response) or (
            PHASE_COMMENTARY if (getattr(response, "tool_calls", None) or []) else PHASE_FINAL_ANSWER
        )
        self._emit_tool_call_commentary(response, phase)
        wrapped = self._with_phase(response, phase)
        if prepared.persisted_history is not None:
            message_updates = [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + list(prepared.persisted_history) + [wrapped]
        else:
            message_updates = list(prepared.history_updates or []) + [wrapped]
        retries = int(graph_state.get("commentary_retries", 0))
        if phase == PHASE_COMMENTARY:
            retries += 1
        else:
            retries = 0
        if paper_reviewer_guarded:
            retries = 0
        return {"messages": message_updates, "commentary_retries": retries}

    def _record_llm_context_usage(self, response: Any) -> None:
        stats = extract_usage_stats(response)

        def _to_int(value: Any) -> Optional[int]:
            try:
                number = int(float(value))
            except Exception:
                return None
            return number if number > 0 else None

        total_tokens = _to_int(stats.get("total_tokens"))
        prompt_tokens = _to_int(stats.get("prompt_tokens"))
        completion_tokens = _to_int(stats.get("completion_tokens"))
        cached_prompt_tokens = _to_int(stats.get("cached_prompt_tokens"))
        budget_tokens = total_tokens or prompt_tokens
        if budget_tokens is None:
            return
        self.state["llm_context_usage"] = {
            "budget_tokens": int(budget_tokens),
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "source": "usage_total_tokens" if total_tokens is not None else "usage_prompt_tokens",
        }

    @staticmethod
    def _last_tool_result_is_subagent_submit(messages: List[BaseMessage]) -> bool:
        if not messages or not isinstance(messages[-1], ToolMessage):
            return False
        tool_call_id = str(getattr(messages[-1], "tool_call_id", "") or "")
        if not tool_call_id:
            return False
        for message in reversed(messages[:-1]):
            if not isinstance(message, AIMessage):
                continue
            for call in getattr(message, "tool_calls", None) or []:
                if not isinstance(call, dict):
                    continue
                if str(call.get("id") or "") != tool_call_id:
                    continue
                name = str(call.get("name") or "").strip()
                return name in _SUBAGENT_SUBMIT_TOOL_NAMES
        return False

    def post_tool_hook_node(self, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        if self.agent_role != "planner":
            messages = list(graph_state.get("messages") or [])
            if self._last_tool_result_is_subagent_submit(messages):
                submitted = self.tools_handler.peek_subagent_result(copy_result=True) or {}
                status = str(submitted.get("status") or "completed").strip() or "completed"
                summary = str(submitted.get("summary") or "Structured subagent result submitted.").strip()
                return {
                    "messages": [
                        AIMessage(
                            content=f"Structured subagent result submitted. status={status}\n{summary}",
                            additional_kwargs={"phase": PHASE_FINAL_ANSWER},
                        )
                    ],
                    "commentary_retries": 0,
                }
            return {}
        runtime_action = self._consume_runtime_action()
        if runtime_action:
            return self._run_runtime_action_chain(runtime_action, hook_source="post_tool_hook")
        if str(self.state.get("planner_phase") or "working") != "needs_input":
            return {}

        need = dict(self.state.get("planner_need") or {})
        question = str(need.get("question") or "Please provide additional information.").strip()
        source = str(need.get("source") or "cytobridge").strip() or "cytobridge"
        reason = str(need.get("reason") or "post_tool_hook_needs_input").strip() or "post_tool_hook_needs_input"
        return self._build_needs_input_interrupt_payload(
            question=question,
            source=source,
            reason=reason,
            hook_stage="post_tool_hook",
            hook_name="needs_input_interrupt",
        )

    def stop_hook_node(self, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        if not self._stop_hook_enabled():
            return {}
        messages = list(graph_state.get("messages") or [])
        if not messages:
            return {}
        current_trigger_count = int(graph_state.get("stop_hook_triggers", 0) or 0)
        limit = self._stop_hook_max_triggers()
        if current_trigger_count >= limit:
            return self._build_stop_hook_limit_payload(trigger_count=current_trigger_count, limit=limit)
        trigger_count = current_trigger_count + 1
        last_message = messages[-1]
        phase = resolve_message_phase(last_message) or (
            PHASE_COMMENTARY if (getattr(last_message, "tool_calls", None) or []) else PHASE_FINAL_ANSWER
        )
        terminal_review_phases = {PHASE_FINAL_ANSWER, PHASE_NEEDS_INPUT, PHASE_COMMENTARY}
        if phase not in terminal_review_phases:
            return {"stop_hook_triggers": trigger_count}

        runtime_action = self._consume_runtime_action()
        if runtime_action:
            payload = self._run_runtime_action_chain(runtime_action, hook_source="stop_hook")
            payload["stop_hook_triggers"] = trigger_count
            return payload

        if phase == PHASE_FINAL_ANSWER and str(self.state.get("planner_phase") or "working") == "needs_input":
            need = dict(self.state.get("planner_need") or {})
            question = str(need.get("question") or "Please provide additional information.").strip()
            source = str(need.get("source") or "cytobridge").strip() or "cytobridge"
            reason = str(need.get("reason") or "stop_hook_needs_input").strip() or "stop_hook_needs_input"
            mode = self._stop_hook_mode()
            if mode != "builtin":
                payload = self._build_stop_hook_commentary_payload(
                    reason=(
                        f"Planner still has unresolved needs-input state from `{source}` ({reason}): {question}. "
                        "Autonomous stop-hook mode requires resolving or working around this without asking the user."
                    ),
                    hook_name="planner_needs_input_guard",
                    hook_mode=mode,
                )
                payload["stop_hook_triggers"] = trigger_count
                return payload
            payload = self._build_needs_input_interrupt_payload(
                question=question,
                source=source,
                reason=reason,
                hook_stage="stop_hook",
                hook_name="planner_needs_input_guard",
            )
            payload["stop_hook_triggers"] = trigger_count
            return payload

        if phase in terminal_review_phases:
            payload = self._run_custom_stop_hook(
                messages=messages,
                last_message=last_message,
                trigger_count=trigger_count,
            )
            if payload:
                payload["stop_hook_triggers"] = trigger_count
                return payload

        return {"stop_hook_triggers": trigger_count}

    def _stop_hook_max_triggers(self) -> int:
        raw = self.state.get("stop_hook_max_triggers", DEFAULT_STOP_HOOK_MAX_TRIGGERS)
        try:
            value = int(raw)
        except Exception:
            value = DEFAULT_STOP_HOOK_MAX_TRIGGERS
        return max(1, min(value, 50))

    def _build_stop_hook_limit_payload(self, *, trigger_count: int, limit: int) -> Dict[str, Any]:
        need = dict(self.state.get("planner_need") or {})
        question = str(need.get("question") or "").strip()
        if not question:
            question = (
                f"Runtime stop-hook trigger limit reached ({limit}) before the planner could finalize cleanly. "
                "Review the latest planner output and unresolved runtime state before continuing."
            )
        source = str(need.get("source") or "cytobridge").strip() or "cytobridge"
        reason = str(need.get("reason") or "stop_hook_limit_reached").strip() or "stop_hook_limit_reached"
        self.state["planner_phase"] = "needs_input"
        self.state["planner_need"] = {
            **need,
            "source": source,
            "question": question,
            "reason": reason,
            "stop_hook_trigger_count": trigger_count,
            "stop_hook_trigger_limit": limit,
        }
        self._emit_event(
            "stop_hook_limit_reached",
            {
                "trigger_count": trigger_count,
                "trigger_limit": limit,
                "question": question,
            },
        )
        payload = self._build_needs_input_interrupt_payload(
            question=question,
            source=source,
            reason=reason,
            hook_stage="stop_hook",
            hook_name="stop_hook_trigger_limit",
        )
        payload["stop_hook_triggers"] = trigger_count
        return payload

    def _build_needs_input_interrupt_payload(
        self,
        *,
        question: str,
        source: str,
        reason: str,
        hook_stage: str,
        hook_name: str,
    ) -> Dict[str, Any]:
        self._emit_event(
            f"{hook_stage}_triggered",
            {
                "hook": hook_name,
                "source": source,
                "reason": reason,
                "question": question,
            },
        )
        return {
            "messages": [
                AIMessage(
                    content=question,
                    additional_kwargs={
                        "phase": PHASE_NEEDS_INPUT,
                        "needs_input_source": source,
                        "needs_input_reason": reason,
                        "hook_stage": hook_stage,
                        "hook_name": hook_name,
                    },
                )
            ],
            "commentary_retries": 0,
        }

    def _run_runtime_action(self, runtime_action: Dict[str, Any], *, hook_source: str) -> Dict[str, Any]:
        kind = str(runtime_action.get("kind") or "").strip().lower()
        if kind == "proposal_agent_review":
            return self._run_proposal_agent_review(runtime_action)
        if kind == "research_idea_agent_review":
            return self._run_research_idea_agent_review(runtime_action)
        if kind == "inference_agent_review":
            return self._run_inference_agent_review(runtime_action)
        if kind == "implementation_agent_review":
            return self._run_implementation_agent_review(runtime_action)
        if not kind:
            return {}
        self._emit_event(
            f"{hook_source}_triggered",
            {
                "hook": "runtime_action_guard",
                "runtime_action_kind": kind,
            },
        )
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"A pending runtime action (`{kind}`) must be handled before this turn can finish. "
                        "Do not finalize yet."
                    ),
                    additional_kwargs={
                        "phase": PHASE_COMMENTARY,
                        "hook_stage": hook_source,
                        "hook_name": "runtime_action_guard",
                        "runtime_action_kind": kind,
                    },
                )
            ],
            "commentary_retries": 0,
        }

    def _run_runtime_action_chain(
        self,
        runtime_action: Dict[str, Any],
        *,
        hook_source: str,
        max_actions: int = 4,
    ) -> Dict[str, Any]:
        """Run queued runtime actions before returning control to the planner.

        Some runtime actions resume a tool call that may enqueue a follow-up
        action, e.g. implementation review approval can resume a campaign trial
        which then requests inference review. Returning to the planner between
        those internal actions lets the model guess at structured results that
        only the runtime should produce.
        """
        messages: List[BaseMessage] = []
        payload: Dict[str, Any] = {}
        action = dict(runtime_action or {})
        actions_run = 0
        while action and actions_run < max_actions:
            actions_run += 1
            payload = self._run_runtime_action(action, hook_source=hook_source)
            messages.extend(list(payload.get("messages") or []))
            action = self._consume_runtime_action()

        if action:
            self.state["runtime_action"] = action
            kind = str(action.get("kind") or "").strip().lower()
            self._emit_event(
                "runtime_action_chain_limit_reached",
                {
                    "hook_source": hook_source,
                    "actions_run": actions_run,
                    "pending_runtime_action_kind": kind,
                },
            )
            messages.append(
                AIMessage(
                    content=(
                        f"Runtime action chain limit reached after {actions_run} action(s). "
                        f"Pending action `{kind or 'unknown'}` remains queued; do not finalize until it is handled."
                    ),
                    additional_kwargs={
                        "phase": PHASE_COMMENTARY,
                        "hook_stage": hook_source,
                        "hook_name": "runtime_action_chain_limit",
                        "runtime_action_kind": kind,
                    },
                )
            )

        result = dict(payload or {})
        if messages:
            result["messages"] = messages
        result["commentary_retries"] = 0
        return result

    def _consume_runtime_action(self) -> Dict[str, Any]:
        action = dict(self.state.get("runtime_action") or {})
        self.state["runtime_action"] = {}
        return action

    def _build_proposal_evaluator_context(self, action: Dict[str, Any]) -> Dict[str, Any]:
        algorithm_id = str(action.get("algorithm_id") or "").strip().lower()
        proposal_id = str(action.get("proposal_id") or "").strip()
        proposals = self.state.get("algorithm_proposals") or {}
        record = dict(proposals.get(algorithm_id) or {})
        proposal_path = str(
            action.get("review_proposal_path")
            or action.get("proposal_markdown_registry_path")
            or record.get("proposal_markdown_registry_path")
            or action.get("proposal_path")
            or record.get("proposal_path")
            or ""
        ).strip()
        editable_proposal_path = str(
            action.get("editable_proposal_path")
            or record.get("editable_proposal_path")
            or record.get("proposal_path")
            or ""
        ).strip()
        proposal_json_path = str(
            action.get("proposal_json_path")
            or record.get("registry_path")
            or record.get("proposal_json_path")
            or ""
        ).strip()
        relevant_paths = []
        for path in [proposal_path, editable_proposal_path, proposal_json_path]:
            if path and path not in relevant_paths:
                relevant_paths.append(path)
        previous_proposal_id = str(action.get("previous_proposal_id") or "").strip()
        revision_note = str(action.get("revision_note") or "").strip()
        revision_diff = str(action.get("proposal_revision_diff") or "").strip()
        revision_patch = str(action.get("proposal_revision_patch") or "").strip()
        revision_block = ""
        if previous_proposal_id or revision_diff or revision_patch:
            diff_or_patch = revision_diff or revision_patch
            max_chars = 12000
            truncated = ""
            if len(diff_or_patch) > max_chars:
                diff_or_patch = diff_or_patch[:max_chars]
                truncated = "\n[revision diff/patch truncated for prompt length]"
            revision_block = (
                "\n\nRevision context:\n"
                f"- Previous proposal_id: {previous_proposal_id or '(unknown)'}\n"
                f"- Current proposal_id: {proposal_id or '(unknown)'}\n"
                f"- Revision note: {revision_note or '(none)'}\n"
                "- This proposal was created by patching PROPOSAL.md. Compare the patch against the final proposal; "
                "check whether the revision genuinely fixes prior issues or instead lowers ambition, hides gaps, turns the algorithm into a weaker config/diagnostic/proxy variant, changes the primary claim into an easier proxy, or creates inconsistencies.\n"
                "```diff\n"
                f"{diff_or_patch}{truncated}\n"
                "```"
            )
        task = (
            f"Evaluate the mathematical correctness of custom algorithm proposal `{algorithm_id}` "
            f"(proposal_id={proposal_id or 'unknown'}). Determine whether the proposal, if trained to convergence "
            "under its stated assumptions, can fit the observed per-time weighted particle distributions and the "
            "observed mass / cell-count changes when mass modeling is actually claimed by the proposal. Also determine "
            "whether the proposed algorithm can solve the problem and claimed capability stated in the proposal."
        )
        context_notes = (
            "Review scope:\n"
            "- Focus primarily on the proposal artifact itself.\n"
            "- Read the proposal's `Abstract`, `Literature and Package Grounding`, `Problem Statement`, `Claimed Capability`, `Expected Evaluation Outcome`, `Inductive Generalization Argument`, and optional `Problem Mathematical Form` before judging the method.\n"
            "- For a genuinely new algorithm, require enough package/literature grounding to show the proposer read beyond RAG snippets: at least 10 directly relevant algorithm-paper citations or literature notes with relevance notes, unless this is only a minor config change or known builtin variant.\n"
            "- If the proposal uses nontrivial OT/SB/WFR/UOT/continuity-equation or variational-flow theory, require theory-book grounding too: book/page or chapter/section/page ranges that show the proposer read the relevant mathematical source rather than only paper snippets.\n"
            "- A missing global mathematical form is acceptable for methods that do not have one, but the problem and solution route must still be explicit.\n"
            "- Read the proposal's machine-readable `Mass Modeling Scope` and `Unbalanced Decision` sections before deciding whether mass / cell-count fit is part of the claimed contract.\n"
            "- Treat CytoBridge's target semantics as weighted distribution fit across time, plus mass-fit / observed cell-count change fit only if the proposal explicitly claims to model unbalanced mass.\n"
            "- If the proposal claims `models_unbalanced_mass`, require a meaningful mass/growth mechanism: a learned growth field/head integrated as `d log w / dt`, a WFR/UOT/SB birth-death/proliferation mechanism, a justified condition-driven growth prior, or another proposal-defined biological/dynamical mechanism. Do not approve target-count lookup, post-hoc weight rescaling, time-only count-ratio clocks, renormalization layers, or corrections whose only role is to force TMV after trajectory generation.\n"
            "- If the proposal claims to solve a mathematical problem, check whether it formulates the dynamic problem clearly enough, for example dynamic OT, Schrödinger bridge, WFR/UOT, mean-field dynamics, or a new well-defined dynamic objective; if not, require precise local dynamics, target marginals, and validation contract.\n"
            "- If the proposal claims a concrete mathematical problem or algorithmic novelty, require a self-contained `Mathematical Derivation to Algorithm Design`: problem variables -> computable representation or justified relaxation -> supervision targets/losses -> inference rule/evaluation trajectory -> distribution recovery and mass recovery when applicable. Use `CytoBridge-main/docs/theory/wfrfm-derivation-example.md` as the explicitness bar; do not approve derivations that skip the bridge from objective to trainable targets.\n"
            "- Check whether the proposal defines an inductive runtime dynamics rule for new valid t=0 cells/particles under the stated data contract. Do not approve proposals whose inference principle depends on training-cell ids, row ids, memorized OT rows, target-specific nearest-neighbor tables, barcode-specific hardcoding unavailable for new cells, future observed snapshots, or target-specific correction.\n"
            "- If extra modalities, barcodes, lineage labels, batch metadata, or exogenous conditions are used, require the proposal to distinguish training-only supervision from inference-time inputs and to state the data contract needed for new cells.\n"
            "- For applied or biology-motivation-first proposals, check that components, conditioning variables, mass/growth mechanism, inference rule, and expected evidence correspond to a real biological question/process/data regime rather than an arbitrary correction layer or metric hack.\n"
            "- Treat claim-metric alignment as a proposal-level hard gate. Do not approve a proposal or revision that replaces the original user/data-driven gap with an easier proxy that can pass while the biological or dynamical problem remains unsolved. A proxy is acceptable only if the proposal proves why it is necessary for the original claim and defines random/collapsed/shuffled/claim-absent controls that would fail it.\n"
            "- Treat automatic algorithm downgrading as a proposal-level hard gate. If a custom algorithm was started to solve a real biological, dynamical, or theoretical gap, do not approve a revision that turns it into a diagnostic report, baseline config tweak, metric-only wrapper, post-hoc correction, or low-novelty builtin-adjacent variant unless the user explicitly requested that narrower mode. Repeated campaign failure should lead to a stronger revised algorithm or rejection with better routes, not silent downgrading.\n"
            "- For theory-first proposals, check genuine theoretical contribution and construction quality: a meaningful mathematical/dynamical problem, nontrivial equivalence or estimator, or a clean new modeling axis rather than stitched surrogate losses or renamed builtin variants.\n"
            "- Do not require a deterministic ODE path or individual weighted particle to literally split unless branching, lineage bifurcation, multimodal fate uncertainty for one initial cell, or branch coverage is central to the proposal's claimed scientific problem. In the infinite-particle / weighted-empirical-measure limit, non-branching characteristics and branching particle stories can be different microscopic realizations of the same evolving marginal distribution; judge non-branching proposals by whether their induced weighted distribution can match the target marginals over time.\n"
            "- Check whether the proposal's exact-fit / convergence-limit reasoning is mathematically coherent.\n"
            "- Check whether the theoretical core, evaluation plan, and expected evaluation outcome actually validate the claimed problem/capability, not merely generic distribution fit.\n"
            "- The expected evaluation outcome should state a plausible working scenario, expected result pattern beyond builtin W1/TMV, and why those results would validate the claimed capability.\n"
            "- Check whether the proposal really justifies fitting the observed next-time distributions, not merely a surrogate objective.\n"
            "- If the proposal's `Mass Modeling Scope` is `balanced_only`, do not penalize it for not matching observed total-mass changes.\n"
            "- If the proposal's `Mass Modeling Scope` is `models_unbalanced_mass`, check whether mass / growth handling is sufficient to recover observed total-mass changes.\n"
            "- approve/revise/reject must be determined by theoretical correctness/sufficiency within the proposal's claimed modeling scope and, for genuinely new algorithms, conceptual quality/novelty relative to relevant builtin and literature baselines.\n"
            "- Always report concrete engineering implementation risk points separately as advisory notes, even when the theory is approved.\n"
            "- Implementation risks should focus on practical execution concerns: numerical stability, optimization sensitivity, finite-sample estimator variance, rollout drift, supervision mismatch under finite training, memory/compute scaling, or likely failure modes in actual training/inference.\n"
            "- Do not use the implementation-risk section to merely restate theorem assumptions or theoretical caveats unless they directly create an engineering failure mode.\n"
            "- Implementation risks are non-blocking unless they reveal a genuine theoretical flaw.\n"
            "- Keep the bar high. If key derivation steps are missing or too vague, prefer `revise` over `approve`.\n"
            "- Use `reject` when the core idea is low-quality rather than merely incomplete; include 2-4 concrete improvement routes in the feedback, such as deriving a real growth field, formulating a principled WFR/SB/UOT dynamic problem, narrowing honestly to balanced-only, or replacing correction layers with a single derived objective.\n"
            "- Finish by calling submit_proposal_review(decision, summary, reviewer_feedback, findings, implementation_risks, risk_assessment, confidence).\n"
            "- `reviewer_feedback` should give actionable mathematical review comments for the parent planner.\n"
            "- `implementation_risks` should list likely engineering failure points in implementation, optimization, or numerical behavior without changing the theoretical verdict by themselves. Sort them by expected severity, highest first.\n"
            "- `risk_assessment` should be severity-ranked markdown with one section or table per risk, covering: risk, severity rationale, likely manifestation in CytoBridge training/campaign/evaluation, and practical diagnostics."
            f"{revision_block}"
        )
        success_criteria = [
            "Read the proposal artifact before judging it.",
            "Read the proposal's literature/package grounding, problem statement, claimed capability, expected evaluation outcome, Mass Modeling Scope, and Unbalanced Decision.",
            "Derive the key mathematical bridge from the proposal's own claims.",
            "If a revision patch is provided, compare the patch to the final proposal and reject/revise regressions or inconsistent edits.",
            "Reject or request revision when the primary claim metric no longer tests the original motivating gap.",
            "Judge whether convergence would imply fitting the claimed target: always the weighted distributions, and mass changes only when unbalanced mass is explicitly modeled.",
            "Judge whether the proposal can solve its own claimed problem/capability under its stated assumptions.",
            "Judge whether the expected evaluation outcome is specific enough to guide later benchmark/simulation selection and validate the claim.",
            "Return approve/revise/reject with high-standard review feedback.",
        ]
        return {
            "algorithm_id": algorithm_id,
            "proposal_id": proposal_id,
            "relevant_paths": relevant_paths,
            "task": task,
            "context_notes": context_notes,
            "success_criteria": success_criteria,
            "previous_proposal_id": previous_proposal_id,
            "proposal_revision_diff": revision_diff,
        }

    def _run_proposal_agent_review(self, action: Dict[str, Any]) -> Dict[str, Any]:
        manager = getattr(self.tools_handler, "subagent_manager", None)
        if manager is None:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Proposal evaluator could not run because the planner runtime has no subagent manager. "
                            "Do not proceed to authoring until the proposal is reviewed."
                        ),
                        additional_kwargs={"phase": PHASE_COMMENTARY},
                    )
                ],
                "commentary_retries": 0,
            }

        ctx = self._build_proposal_evaluator_context(action)
        self._emit_event(
            "algorithm_proposal_agent_review_started",
            {
                "algorithm_id": ctx["algorithm_id"],
                "proposal_id": ctx["proposal_id"],
                "proposal_paths": list(ctx["relevant_paths"]),
            },
        )
        result = self._run_structured_review_subagent(
            manager,
            subagent_type="proposal_evaluator",
            task=ctx["task"],
            success_criteria=list(ctx["success_criteria"]),
            context_notes=ctx["context_notes"],
            relevant_paths=list(ctx["relevant_paths"]),
            review_key="proposal_review",
            required_text_fields=("reviewer_feedback", "risk_assessment"),
            required_nonempty_list_fields=("implementation_risks",),
            repair_instruction=(
                "For proposal reviews, call submit_proposal_review(...) and include decision, summary, "
                "reviewer_feedback, implementation_risks sorted by severity, risk_assessment, and confidence."
            ),
        )
        review_payload = _extract_review_payload(result, "proposal_review")
        _, contract_errors = _validate_structured_review_result(
            result,
            "proposal_review",
            required_text_fields=("reviewer_feedback", "risk_assessment"),
            required_nonempty_list_fields=("implementation_risks",),
        )
        decision = str(review_payload.get("decision") or "").strip().lower()
        feedback = str(review_payload.get("reviewer_feedback") or "").strip()
        implementation_risks = [
            str(item).strip()
            for item in (review_payload.get("implementation_risks") or [])
            if str(item).strip()
        ]
        risk_assessment = str(review_payload.get("risk_assessment") or "").strip()
        confidence = review_payload.get("confidence")
        if contract_errors or result.get("status") != "completed" or decision not in {"approve", "revise", "reject"} or not feedback:
            summary = str(result.get("summary") or "Proposal evaluator did not produce a usable verdict.").strip()
            findings = [str(item).strip() for item in (result.get("findings") or []) if str(item).strip()]
            details = "\n".join(f"- {item}" for item in findings[:6])
            contract_details = "\n".join(f"- {item}" for item in contract_errors)
            content = (
                f"Proposal evaluator could not complete a usable review for `{ctx['algorithm_id']}`.\n"
                f"Summary: {summary}\n"
                "Do not proceed to authoring yet. Diagnose the proposal or retry the evaluator."
            )
            if contract_details:
                content = content.replace(
                    "Do not proceed to authoring yet.",
                    f"Structured contract errors:\n{contract_details}\nDo not proceed to authoring yet.",
                )
            if details:
                content += f"\nFindings:\n{details}"
            return {
                "messages": [AIMessage(content=content, additional_kwargs={"phase": PHASE_COMMENTARY})],
                "commentary_retries": 0,
            }

        apply_result = self.tools_handler.review_algorithm_proposal(
            algorithm_id=ctx["algorithm_id"],
            decision=decision,
            reviewer_feedback=feedback,
            proposal_id=ctx["proposal_id"],
            implementation_risks=implementation_risks,
            risk_assessment=risk_assessment,
        )
        if isinstance(apply_result, str) and apply_result.startswith("Error:"):
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"Proposal evaluator produced a verdict for `{ctx['algorithm_id']}`, but applying the review failed:\n"
                            f"{apply_result}\n"
                            "Do not proceed to authoring until this is resolved."
                        ),
                        additional_kwargs={"phase": PHASE_COMMENTARY},
                    )
                ],
                "commentary_retries": 0,
            }

        self._emit_event(
            "algorithm_proposal_agent_review_completed",
            {
                "algorithm_id": ctx["algorithm_id"],
                "proposal_id": ctx["proposal_id"],
                "decision": decision,
                "confidence": confidence,
                "summary": str(result.get("summary") or "").strip(),
                "implementation_risks": implementation_risks,
                "risk_assessment": risk_assessment,
            },
        )
        next_step = {
            "approve": "The proposal is approved. Continue to workspace initialization if custom authoring is still required.",
            "revise": "The proposal is marked revision_requested. Revise the proposal before any authoring or training.",
            "reject": "The proposal is rejected. Rework the mathematical formulation before proceeding.",
        }[decision]
        findings = [str(item).strip() for item in (result.get("findings") or []) if str(item).strip()]
        findings_block = "\n".join(f"- {item}" for item in findings[:8])
        content_lines = [
            f"Proposal evaluator review completed for `{ctx['algorithm_id']}`.",
            f"Decision: {decision}",
            f"Summary: {str(result.get('summary') or '').strip() or '(no summary)'}",
            "Reviewer feedback:",
            feedback,
        ]
        if findings_block:
            content_lines.extend(["Findings:", findings_block])
        if implementation_risks:
            content_lines.extend(
                [
                    "Implementation risks:",
                    "\n".join(f"- {item}" for item in implementation_risks[:8]),
                ]
            )
        if risk_assessment:
            content_lines.extend(["Risk assessment saved to risk.md:", risk_assessment[:1200]])
        content_lines.append(next_step)
        return {
            "messages": [
                AIMessage(
                    content="\n".join(content_lines),
                    additional_kwargs={"phase": PHASE_COMMENTARY, "proposal_review_decision": decision},
                )
            ],
            "commentary_retries": 0,
        }

    def _build_inference_evaluator_context(self, action: Dict[str, Any]) -> Dict[str, Any]:
        algorithm_id = str(action.get("algorithm_id") or "").strip().lower()
        proposal_id = str(action.get("proposal_id") or "").strip()
        purpose = str(action.get("purpose") or "").strip()
        review_hash = str(action.get("review_hash") or "").strip()
        relevant_paths = [
            str(path).strip()
            for path in (action.get("relevant_paths") or [])
            if str(path).strip()
        ]
        fingerprint_payload = dict(action.get("fingerprint_payload") or {})
        task = (
            f"Review custom algorithm `{algorithm_id}` inference and claim-metric code before trusted training "
            f"(purpose={purpose or 'training'}, proposal_id={proposal_id or 'unknown'}, review_hash={review_hash}). "
            "Decide whether the code generates evaluation predictions from allowed t=0/exogenous/model-state inputs "
            "as a full t0-to-final trajectory, and whether custom claim metrics faithfully measure the approved "
            "proposal target rather than a mismatched proxy."
        )
        context_notes = (
            "Review scope:\n"
            "- Read the algorithm implementation, approved proposal, IMPLEMENTATION_MAP.md, relevant proposal/contract files, and read-only dataset metadata when a `.h5ad` path or benchmark dataset id is available.\n"
            "- Use the package default inference path as reference: TrainingPipeline._predict_evaluation_trajectory(...) "
            "calls simulate_trajectory(...) from t=0 particles over SimulationContext.trajectory_time_points; "
            "then package code slices that trajectory for builtin W1/TMV.\n"
            "- Approved custom inference may use t=0 cells, t=0 aligned modalities, known exogenous conditions, constants, "
            "time grid, config, and trained model state.\n"
            "- Custom inference must not read future observed cells, future cell counts, future total mass, future modalities, "
            "or future distribution statistics to construct predictions.\n"
            "- Reject or request revision if prediction weights/positions are post-hoc corrected to match TMV/W1/claim metrics.\n"
            "- Reject or request revision if simulation_hook returns only metric-specific future predictions instead of the full trajectory requested by the runtime.\n"
            "- Reject or request revision if a head directly predicts TMV/W1/total mass/claim metric and treats that as the dynamics output.\n"
            "- For additive claim metrics and baseline adapters, audit semantic fidelity to the approved proposal. "
            "Name the metric's prediction_source, truth_source, grouping_key, timepoints, label_source, leakage_controls, proposal_mapping, and gap_mapping. "
            "Reject or request revision if IMPLEMENTATION_MAP.md lacks an explicit claim-metric contract, if the metric compares generated outputs to the wrong observed/truth target, wrong time point, wrong group, or an unauditable proxy, or if the metric has been downgraded to a weaker proxy that can pass while the original user/data-driven gap remains unsolved, even when it does not mutate prediction artifacts. "
            "Do not assume initial-time labels are valid endpoint/descendant/held-out truth unless the approved proposal, implementation map, dataset registration contract, or preprocessing contract explicitly says so; advisory risk files and previous reviewer notes are not authoritative truth-source contracts.\n"
            "- Use inspect_h5ad_contract(...) for relevant .h5ad files to verify actual obs columns, time keys, label/fate/lineage columns, and value distributions. This is read-only metadata inspection only: do not run training, execute generic Python, mutate AnnData, materialize/download data, or change dataset files.\n"
            "- Do not infer biological truth semantics from column names or category counts alone. Column availability proves only that the field exists; it does not prove that t0 labels are terminal fate truth, lineage strings define descendant endpoints, current-state labels are held-out truth, or prefix/clone relations are the intended ground-truth mapping. If a custom metric relies on those semantics and the approved proposal/map/dataset contract does not state them explicitly, request revision.\n"
            "- If an inference_context_builder exists, check that it prepares a flexible payload with credible provenance from t=0/exogenous/model-state inputs.\n"
            "- Do not require fixed payload field names; judge the data boundary and dynamics-rollout semantics.\n"
            "- Finish by calling submit_subagent_result(...) with proposed_state_updates.inference_review containing keys: "
            "decision (approve|revise|reject), reviewer_feedback, blocking_issues, advisory_risks.\n"
            f"- Fingerprint payload for traceability:\n{json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True)[:8000]}"
        )
        success_criteria = [
            "Identify the effective simulation_hook and inference_context_builder when present.",
            "Verify whether evaluation prediction uses only t=0/exogenous/model-state inputs.",
            "Verify whether the generated trajectory comes from the learned dynamics rather than a metric-specific prediction or post-hoc correction.",
            "For each custom claim metric, verify semantic alignment between proposal, implementation map, code, dataset contract, and read-only dataset metadata; require an explicit Claim Metric Contract in IMPLEMENTATION_MAP.md, including why the metric still tests the original motivating gap.",
            "Return approve/revise/reject with concrete blocking issues and advisory risks.",
        ]
        return {
            "algorithm_id": algorithm_id,
            "proposal_id": proposal_id,
            "purpose": purpose,
            "review_hash": review_hash,
            "relevant_paths": relevant_paths,
            "task": task,
            "context_notes": context_notes,
            "success_criteria": success_criteria,
        }

    def _run_inference_agent_review(self, action: Dict[str, Any]) -> Dict[str, Any]:
        manager = getattr(self.tools_handler, "subagent_manager", None)
        ctx = self._build_inference_evaluator_context(action)
        if manager is None:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Inference evaluator could not run because the planner runtime has no subagent manager. "
                            "Do not proceed to trusted training/campaign evaluation until inference code is reviewed."
                        ),
                        additional_kwargs={"phase": PHASE_COMMENTARY},
                    )
                ],
                "commentary_retries": 0,
            }
        self._emit_event(
            "algorithm_inference_agent_review_started",
            {
                "algorithm_id": ctx["algorithm_id"],
                "proposal_id": ctx["proposal_id"],
                "purpose": ctx["purpose"],
                "review_hash": ctx["review_hash"],
                "relevant_paths": list(ctx["relevant_paths"]),
            },
        )
        result = self._run_structured_review_subagent(
            manager,
            subagent_type="inference_evaluator",
            task=ctx["task"],
            success_criteria=list(ctx["success_criteria"]),
            context_notes=ctx["context_notes"],
            relevant_paths=list(ctx["relevant_paths"]),
            review_key="inference_review",
            required_text_fields=("reviewer_feedback",),
            required_present_fields=("blocking_issues", "advisory_risks"),
            repair_instruction=(
                "For inference reviews, call submit_subagent_result(...) with "
                "proposed_state_updates.inference_review containing decision, reviewer_feedback, "
                "blocking_issues, and advisory_risks."
            ),
        )
        review_payload = _extract_review_payload(result, "inference_review")
        _, contract_errors = _validate_structured_review_result(
            result,
            "inference_review",
            required_text_fields=("reviewer_feedback",),
            required_present_fields=("blocking_issues", "advisory_risks"),
        )
        decision = str(review_payload.get("decision") or "").strip().lower()
        feedback = str(review_payload.get("reviewer_feedback") or "").strip()
        blocking_issues = _coerce_review_list(review_payload.get("blocking_issues"))
        advisory_risks = _coerce_review_list(review_payload.get("advisory_risks"))
        findings = [str(item).strip() for item in (result.get("findings") or []) if str(item).strip()]
        if contract_errors or result.get("status") != "completed" or decision not in _REVIEW_DECISIONS or not feedback:
            summary = str(result.get("summary") or "Inference evaluator did not produce a usable verdict.").strip()
            contract_details = "\n".join(f"- {item}" for item in contract_errors)
            content = (
                f"Inference evaluator could not complete a usable review for `{ctx['algorithm_id']}`.\n"
                f"Summary: {summary}\n"
            )
            if contract_details:
                content += f"Structured contract errors:\n{contract_details}\n"
            content += "Do not proceed to trusted training or campaign evaluation until this is resolved."
            return {
                "messages": [
                    AIMessage(
                        content=content,
                        additional_kwargs={"phase": PHASE_COMMENTARY},
                    )
                ],
                "commentary_retries": 0,
            }
        record = self.tools_handler.planner_file_tools.record_inference_review(
            ctx["algorithm_id"],
            review_hash=ctx["review_hash"],
            decision=decision,
            reviewer_feedback=feedback,
            findings=findings,
            blocking_issues=blocking_issues,
            advisory_risks=advisory_risks,
            subagent_id=str(result.get("subagent_id") or ""),
            reviewed_paths=list(ctx["relevant_paths"]),
            review_summary=str(result.get("summary") or "").strip(),
        )
        self._emit_event(
            "algorithm_inference_agent_review_completed",
            {
                "algorithm_id": ctx["algorithm_id"],
                "proposal_id": ctx["proposal_id"],
                "purpose": ctx["purpose"],
                "decision": decision,
                "status": record.get("status"),
                "review_hash": ctx["review_hash"],
                "blocking_issues": blocking_issues,
                "advisory_risks": advisory_risks,
            },
        )
        content_lines = [
            f"Inference evaluator review completed for `{ctx['algorithm_id']}`.",
            f"Decision: {decision}",
            f"Summary: {str(result.get('summary') or '').strip() or '(no summary)'}",
            "Reviewer feedback:",
            feedback,
        ]
        if findings:
            content_lines.extend(["Findings:", "\n".join(f"- {item}" for item in findings[:8])])
        if blocking_issues:
            content_lines.extend(["Blocking issues:", "\n".join(f"- {item}" for item in blocking_issues[:8])])
        if advisory_risks:
            content_lines.extend(["Advisory risks:", "\n".join(f"- {item}" for item in advisory_risks[:8])])
        if decision == "approve":
            content_lines.append("The custom inference surface is approved for this fingerprint.")
            self._append_resume_after_review_result(
                action,
                content_lines=content_lines,
                review_kind="inference",
            )
        else:
            content_lines.append("The custom inference surface is blocked. Revise the code before trusted training/campaign evaluation.")
        return {
            "messages": [
                AIMessage(
                    content="\n".join(content_lines),
                    additional_kwargs={"phase": PHASE_COMMENTARY, "inference_review_decision": decision},
                )
            ],
            "commentary_retries": 0,
        }

    def _build_implementation_evaluator_context(self, action: Dict[str, Any]) -> Dict[str, Any]:
        algorithm_id = str(action.get("algorithm_id") or "").strip().lower()
        proposal_id = str(action.get("proposal_id") or "").strip()
        purpose = str(action.get("purpose") or "").strip()
        review_hash = str(action.get("review_hash") or "").strip()
        review_hash_scope = str(action.get("review_hash_scope") or "proposal_semantics").strip()
        implementation_review_policy_version = int(
            action.get("implementation_review_policy_version") or IMPLEMENTATION_REVIEW_POLICY_VERSION
        )
        campaign_id = str(action.get("campaign_id") or "").strip()
        campaign_trial_count = int(action.get("campaign_trial_count") or 0)
        review_policy = str(action.get("review_policy") or "initial_or_proposal_semantics_change").strip()
        due_reasons = [str(item).strip() for item in (action.get("due_reasons") or []) if str(item).strip()]
        relevant_paths = [
            str(path).strip()
            for path in (action.get("relevant_paths") or [])
            if str(path).strip()
        ]
        fingerprint_payload = dict(action.get("fingerprint_payload") or {})
        task = (
            f"Review custom algorithm `{algorithm_id}` implementation alignment "
            f"(purpose={purpose or 'training'}, proposal_id={proposal_id or 'unknown'}, review_hash={review_hash}). "
            "Decide whether the current code/config faithfully implements the approved proposal, rather than a simplified "
            "or silently approximated algorithm."
        )
        context_notes = (
            "Review scope:\n"
            "- Read the approved proposal version first, then the current algorithm workspace files.\n"
            "- Treat IMPLEMENTATION_MAP.md as the planner's required self-check. Audit it directly against the proposal and code.\n"
            "- Verify that each proposal pseudocode step is mapped to concrete implementation files, package-default behavior, or hook/config choices. "
            "Line references are useful when present, but absence of exact line numbers is not by itself a blocking issue.\n"
            "- Verify every row's Deviation Status. `acceptable approximation` is valid only for semantics-preserving mini-batch/stochastic-estimator/vectorization/caching/streaming style changes.\n"
            "- Treat the approved proposal as an implementation contract, not as loose inspiration.\n"
            "- Check objective, mathematical abstraction, algorithm semantics table, unbalanced/mass decision, stochasticity decision, "
            "distribution recovery argument, implementation pseudocode, and evaluation plan against code/config.\n"
            "- Check mathematical formulas literally, not only component names. Verify identity terms, signs, constants, normalizations, "
            "regularization placement, and scaling factors. For example, a metric specified as `I + ...` is not implemented by "
            "`eps * I + ...` unless the proposal explicitly allows replacing the identity baseline.\n"
            "- The only default acceptable approximations are mini-batch/chunking/streaming/stochastic-estimator/vectorization/caching style changes "
            "that preserve the same mathematical problem and estimator semantics.\n"
            "- Block simplified first-pass implementations that omit proposal-required losses, couplings, growth/mass dynamics, "
            "conditioning variables, stochastic terms, constraints, or inference semantics.\n"
            "- Block post-hoc substitutes that make the method easier while changing the proposal's algorithmic meaning.\n"
            "- Block generalization shortcuts: do not approve code/config that depends on a previously trained model, cached fitted state, "
            "or fine-tuning from an already trained run to make a trial look fast. Campaign evidence must reflect training from the "
            "configured initialization on the target dataset so a new dataset would still work.\n"
            "- Check epoch discipline. Builtin flow-matching methods generally use 3000 epochs; custom flow-matching configs should start "
            "from the seeded 3000-epoch default unless there is concrete convergence evidence that fewer epochs fully converge. "
            "Reducing epochs merely to satisfy the wall-clock budget is not valid evidence.\n"
            "- Use sparse config overrides in campaign/training tools. If you tune one parameter, pass only that leaf path "
            "(for example `training.plan[0].lr` or `training.plan[0].flow_matching.coupling.beta_context`), not a copied "
            "training/model config block. Copied defaults create stale/noisy trial records and can trip fairness guards.\n"
            "- Audit efficiency separately from correctness: identify avoidable CPU-bound or repeated large pairwise computations, "
            "data transfers, serial loops over cells/timepoints, and missing mini-batch/chunking/GPU/vectorized paths. "
            "Give non-blocking acceleration recommendations when they preserve proposal semantics.\n"
            "- If the implementation adds trainable conditional-path modules or extra neural heads, verify optimizer ownership and "
            "device placement. Trainable modules should normally be attached by model_builder and reused by the backend via "
            "FlowMatchingBuildContext.model; backend-created trainable modules are suspect unless explicitly frozen. "
            "Require code or log evidence that proposal-required trainable modules are selected through "
            "cytobridge_trainable_parameters(...), model.cytobridge_component_modules, component_trainable_modules, "
            "trainable_modules, extra_trainable_modules, or standard model submodules. If a trainable path/conditioner/"
            "geometry head is created only inside a backend/path/coupling object and is not optimizer-owned, block "
            "trusted evidence as implementation drift.\n"
            "- Also audit gradient ownership inside custom training loops. If the code uses multiple losses, separate backward "
            "calls, gradient accumulation, or manual stage runners, verify that every proposal-required trainable head receives "
            "a gradient that survives until the corresponding optimizer.step(). Block implementations that compute a loss for "
            "a module but clear it with zero_grad(), detach/no_grad(), or an unrelated optimizer step before the module can be "
            "updated.\n"
            "- Compare custom OT/UOT/WFR coupling implementations against package builtin examples in "
            "`CytoBridge-main/CytoBridge/tl/flow_matching_backends.py`: `BalancedOTCouplingStrategy`, "
            "`UnbalancedOTCouplingStrategy`, `WFROETCouplingStrategy`, `ChunkedTransportCouplingStrategy`, "
            "`_split_transport_chunks(...)`, `_solve_uot_from_pairwise_cost(...)`, and "
            "`_solve_balanced_ot_from_pairwise_cost(...)`. "
            "Check whether POT solver tensors stay on GPU when CUDA is used, whether costs are built per chunk rather than "
            "densely before chunking, and whether chunk sampling/weighting matches the builtin semantics.\n"
            "- Do not block merely because the implementation overrides `build_state(...)` / `sample_pairs(...)`, lacks a "
            "mini-batch marker, or stores non-dense plan metadata. Those are not evidence of failure. Judge scalability from "
            "the actual code path, preview/runtime behavior, and whether bounded `CouplingPlanStore` / chunked / sparse / "
            "streaming state is used correctly.\n"
            "- If the implementation hand-writes an OT/UOT/WFR solver where a package POT/builtin strategy would preserve the same "
            "objective, call that out. Block it only when it changes semantics, is numerically suspect, or creates a clear scalability "
            "failure; otherwise put it in advisory_risks or efficiency_recommendations.\n"
            "- However, definite dense OT/UOT/WFR scalability failures are blocking: if code materializes full all-pairs cost/plan/mask/kernel "
            "state for real benchmark-scale time pairs and only chunks after dense allocation, or stitches chunked "
            "`CouplingPlanStore.sub_plans` back into one global dense plan, return revise/reject with this in blocking_issues.\n"
            "- If implementation difficulty makes the proposal infeasible, require proposal revision before training evidence is trusted.\n"
            "- Prefer `revise` when the implementation is close but missing required pieces; use `reject` only when it is a different method.\n"
            "- Finish by calling submit_implementation_review(...). Do not use free-text findings as the authoritative verdict.\n"
            "- submit_implementation_review fields must include: decision (approve|revise|reject), summary, reviewer_feedback, "
            "findings, blocking_issues, advisory_risks, efficiency_recommendations, generalization_shortcut_risks, "
            "and optional risk_assessment markdown.\n"
            f"- Campaign context: campaign_id={campaign_id or 'none'}, completed_trial_count={campaign_trial_count}, "
            f"review_policy={review_policy}, due_reasons={due_reasons or ['initial review']}.\n"
            f"- Implementation review policy version: {implementation_review_policy_version}.\n"
            f"- Fingerprint payload for traceability:\n{json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True)[:9000]}"
        )
        success_criteria = [
            "Identify the proposal version and the current implementation/config under review.",
            "Audit IMPLEMENTATION_MAP.md row by row against the proposal and current source/config.",
            "Verify whether all proposal-required algorithmic components are implemented without silent semantic simplification.",
            "Distinguish acceptable engineering approximations such as mini-batching from unacceptable changes to the mathematical method.",
            "Check whether config/runtime choices preserve fair scratch training on new datasets, including the default 3000-epoch flow-matching discipline.",
            "Block only evidence-backed dense OT/UOT/WFR all-pairs scalability failures; do not block on static markers alone.",
            "Compare custom OT/UOT/WFR code with builtin chunked/POT solver examples and report implementation-specific risks for risk.md.",
            "Return approve/revise/reject with concrete code-grounded blocking issues and advisory risks.",
        ]
        return {
            "algorithm_id": algorithm_id,
            "proposal_id": proposal_id,
            "purpose": purpose,
            "review_hash": review_hash,
            "review_hash_scope": review_hash_scope,
            "implementation_review_policy_version": implementation_review_policy_version,
            "campaign_id": campaign_id,
            "campaign_trial_count": campaign_trial_count,
            "review_policy": review_policy,
            "relevant_paths": relevant_paths,
            "task": task,
            "context_notes": context_notes,
            "success_criteria": success_criteria,
        }

    def _run_implementation_agent_review(self, action: Dict[str, Any]) -> Dict[str, Any]:
        manager = getattr(self.tools_handler, "subagent_manager", None)
        ctx = self._build_implementation_evaluator_context(action)
        if manager is None:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Implementation evaluator could not run because the planner runtime has no subagent manager. "
                            "Do not proceed to trusted training/campaign evaluation until implementation alignment is reviewed."
                        ),
                        additional_kwargs={"phase": PHASE_COMMENTARY},
                    )
                ],
                "commentary_retries": 0,
            }
        self._emit_event(
            "algorithm_implementation_agent_review_started",
            {
                "algorithm_id": ctx["algorithm_id"],
                "proposal_id": ctx["proposal_id"],
                "purpose": ctx["purpose"],
                "review_hash": ctx["review_hash"],
                "campaign_id": ctx["campaign_id"],
                "campaign_trial_count": ctx["campaign_trial_count"],
                "relevant_paths": list(ctx["relevant_paths"]),
            },
        )
        result = self._run_structured_review_subagent(
            manager,
            subagent_type="implementation_evaluator",
            task=ctx["task"],
            success_criteria=list(ctx["success_criteria"]),
            context_notes=ctx["context_notes"],
            relevant_paths=list(ctx["relevant_paths"]),
            review_key="implementation_review",
            required_text_fields=("reviewer_feedback",),
            required_present_fields=(
                "blocking_issues",
                "advisory_risks",
                "efficiency_recommendations",
                "generalization_shortcut_risks",
            ),
            repair_instruction=(
                "For implementation reviews, call submit_implementation_review(...) and include all list fields, "
                "using empty lists when there are no findings in a category."
            ),
        )
        review_payload = _extract_review_payload(result, "implementation_review")
        _, contract_errors = _validate_structured_review_result(
            result,
            "implementation_review",
            required_text_fields=("reviewer_feedback",),
            required_present_fields=(
                "blocking_issues",
                "advisory_risks",
                "efficiency_recommendations",
                "generalization_shortcut_risks",
            ),
        )
        decision = str(review_payload.get("decision") or "").strip().lower()
        feedback = str(review_payload.get("reviewer_feedback") or "").strip()
        blocking_issues = _coerce_review_list(review_payload.get("blocking_issues"))
        advisory_risks = _coerce_review_list(review_payload.get("advisory_risks"))
        efficiency_recommendations = _coerce_review_list(review_payload.get("efficiency_recommendations"))
        generalization_shortcut_risks = _coerce_review_list(review_payload.get("generalization_shortcut_risks"))
        risk_assessment = str(review_payload.get("risk_assessment") or "").strip()
        findings = [str(item).strip() for item in (result.get("findings") or []) if str(item).strip()]
        if contract_errors or result.get("status") != "completed" or decision not in _REVIEW_DECISIONS or not feedback:
            summary = str(result.get("summary") or "Implementation evaluator did not produce a usable verdict.").strip()
            contract_details = "\n".join(f"- {item}" for item in contract_errors)
            content = (
                f"Implementation evaluator could not complete a usable review for `{ctx['algorithm_id']}`.\n"
                f"Summary: {summary}\n"
            )
            if contract_details:
                content += f"Structured contract errors:\n{contract_details}\n"
            content += "Do not proceed to trusted training or campaign evaluation until this is resolved."
            return {
                "messages": [
                    AIMessage(
                        content=content,
                        additional_kwargs={"phase": PHASE_COMMENTARY},
                    )
                ],
                "commentary_retries": 0,
            }
        record = self.tools_handler.planner_file_tools.record_implementation_review(
            ctx["algorithm_id"],
            proposal_id=ctx["proposal_id"],
            review_hash=ctx["review_hash"],
            decision=decision,
            reviewer_feedback=feedback,
            findings=findings,
            blocking_issues=blocking_issues,
            advisory_risks=advisory_risks,
            efficiency_recommendations=efficiency_recommendations,
            generalization_shortcut_risks=generalization_shortcut_risks,
            risk_assessment=risk_assessment,
            subagent_id=str(result.get("subagent_id") or ""),
            reviewed_paths=list(ctx["relevant_paths"]),
            review_summary=str(result.get("summary") or "").strip(),
            campaign_id=ctx["campaign_id"],
            campaign_trial_count=ctx["campaign_trial_count"],
            review_interval=0,
            review_hash_scope=str(ctx.get("review_hash_scope") or "proposal_semantics"),
            implementation_review_policy_version=int(ctx.get("implementation_review_policy_version") or 0),
        )
        self._emit_event(
            "algorithm_implementation_agent_review_completed",
            {
                "algorithm_id": ctx["algorithm_id"],
                "proposal_id": ctx["proposal_id"],
                "purpose": ctx["purpose"],
                "decision": decision,
                "status": record.get("status"),
                "review_hash": ctx["review_hash"],
                "review_hash_scope": ctx.get("review_hash_scope") or "proposal_semantics",
                "campaign_id": ctx["campaign_id"],
                "campaign_trial_count": ctx["campaign_trial_count"],
                "blocking_issues": blocking_issues,
                "advisory_risks": advisory_risks,
                "efficiency_recommendations": efficiency_recommendations,
                "generalization_shortcut_risks": generalization_shortcut_risks,
                "implementation_risk_path": record.get("implementation_risk_path", ""),
            },
        )
        content_lines = [
            f"Implementation evaluator review completed for `{ctx['algorithm_id']}`.",
            f"Decision: {decision}",
            f"Summary: {str(result.get('summary') or '').strip() or '(no summary)'}",
            "Reviewer feedback:",
            feedback,
        ]
        if findings:
            content_lines.extend(["Findings:", "\n".join(f"- {item}" for item in findings[:8])])
        if blocking_issues:
            content_lines.extend(["Blocking issues:", "\n".join(f"- {item}" for item in blocking_issues[:8])])
        if advisory_risks:
            content_lines.extend(["Advisory risks:", "\n".join(f"- {item}" for item in advisory_risks[:8])])
        if efficiency_recommendations:
            content_lines.extend(["Efficiency recommendations:", "\n".join(f"- {item}" for item in efficiency_recommendations[:8])])
        if generalization_shortcut_risks:
            content_lines.extend(["Generalization shortcut risks:", "\n".join(f"- {item}" for item in generalization_shortcut_risks[:8])])
        if risk_assessment:
            content_lines.extend(["Implementation risk notes saved to risk.md:", risk_assessment[:1200]])
        elif advisory_risks or efficiency_recommendations or generalization_shortcut_risks or blocking_issues:
            content_lines.append(f"Implementation review risks updated in risk.md: {record.get('implementation_risk_path','')}")
        if decision == "approve":
            content_lines.append("The implementation is approved for this proposal/fingerprint.")
            self._append_resume_after_review_result(
                action,
                content_lines=content_lines,
                review_kind="implementation",
            )
        else:
            content_lines.append("The implementation is blocked. Either align the code with the proposal or patch/review the proposal before trusted training.")
        return {
            "messages": [
                AIMessage(
                    content="\n".join(content_lines),
                    additional_kwargs={"phase": PHASE_COMMENTARY, "implementation_review_decision": decision},
                )
            ],
            "commentary_retries": 0,
        }

    def _append_resume_after_review_result(
        self,
        action: Dict[str, Any],
        *,
        content_lines: List[str],
        review_kind: str,
    ) -> None:
        resume = action.get("resume_after_review")
        if not isinstance(resume, dict):
            content_lines.append("No deferred training/campaign call was attached to this review.")
            return
        tool_name = str(resume.get("tool") or "").strip()
        args = dict(resume.get("args") or {})
        if tool_name != "run_campaign_trial":
            content_lines.append(f"Deferred resume tool `{tool_name or 'unknown'}` is not supported; planner should continue manually.")
            return

        campaign_id = str(args.get("campaign_id") or "").strip()
        dataset_config_overrides = args.get("dataset_config_overrides")
        if dataset_config_overrides is not None and not isinstance(dataset_config_overrides, dict):
            dataset_config_overrides = {}
        try:
            resumed = self.tools_handler.run_campaign_trial(
                campaign_id=campaign_id,
                dataset_config_overrides=dataset_config_overrides,
            )
        except Exception as exc:
            logger.exception("Failed to resume run_campaign_trial after %s review", review_kind)
            content_lines.extend(
                [
                    "Automatic resume failed.",
                    f"Error: {exc}",
                    "The planner should inspect the error before retrying the campaign trial.",
                ]
            )
            self._emit_event(
                "runtime_review_resume_failed",
                {
                    "review_kind": review_kind,
                    "tool": tool_name,
                    "campaign_id": campaign_id,
                    "error": str(exc),
                },
            )
            return

        text = str(resumed or "").strip()
        content_lines.extend(
            [
                "Automatically resumed `run_campaign_trial(...)` after review approval.",
                "Resume result:",
                text[:6000] if text else "(empty result)",
            ]
        )
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {}
        self._emit_event(
            "runtime_review_resume_completed",
            {
                "review_kind": review_kind,
                "tool": tool_name,
                "campaign_id": campaign_id,
                "status": str((parsed or {}).get("status") or ""),
                "reason": str((parsed or {}).get("reason") or ""),
                "trial_id": str((parsed or {}).get("trial_id") or ""),
                "decision": str((parsed or {}).get("decision") or ""),
            },
        )

    def _build_research_idea_evaluator_context(self, action: Dict[str, Any]) -> Dict[str, Any]:
        idea_id = str(action.get("idea_id") or "").strip().lower()
        title = str(action.get("title") or "").strip()
        ideas = self.state.get("research_ideas") or {}
        record = dict(ideas.get(idea_id) or {})
        idea_path = str(action.get("idea_path") or record.get("idea_path") or "").strip()
        idea_json_path = str(action.get("idea_json_path") or record.get("idea_json_path") or "").strip()
        relevant_paths = [path for path in [idea_path, idea_json_path] if path]
        task = (
            f"Review research idea `{idea_id}`"
            f"{f' ({title})' if title else ''}. Judge whether the idea is a well-defined, meaningful, and "
            "appropriately scoped scientific problem for its declared track, with falsifiable success criteria and "
            "bounded feasible direction families."
        )
        context_notes = (
            "Review scope:\n"
            "- Read the idea artifact before judging it.\n"
            "- Evaluate the idea as a research-problem asset, not as a full algorithm proposal.\n"
            "- Check whether the problem is concrete, scientifically meaningful, track-appropriate, and evidence-backed.\n"
            "- Check whether the question is narrow enough to be owned by one paper rather than a whole subfield.\n"
            "- Check whether the current CytoBridge scope is respected by default: multi-timepoint snapshot single-cell omics, optional multimodal data, and no perturbation-first framing unless the user explicitly asked for it.\n"
            "- Check whether the idea stays inside the CytoBridge package frame: deep-learning continuous generative dynamics on time-resolved snapshot data rather than generic traditional bioinformatics.\n"
            "- Check whether the idea actually exceeds the current package boundary rather than merely renaming a builtin family or rerunning an already-supported modeling object.\n"
            "- Verify that the scientific object is explicit, the current-method failure mode is concrete, and the success criteria are falsifiable.\n"
            "- Verify that the prior-work section states what earlier methods already achieved, what they still fail to resolve, and what concrete improvement space remains.\n"
            "- Verify that the prior-work section also makes clear what CytoBridge already supports today and why the proposed idea is still genuinely open.\n"
            "- Verify that the feasible direction families are bounded, plausible, and concrete enough to indicate a realistic next algorithm step rather than decorative buzzwords.\n"
            "- If the idea's natural endpoint is a static traditional bioinformatics method rather than a learned continuous dynamics model, prefer `revise` or `reject`.\n"
            "- If multiple direction families are listed, check whether they are ordered by feasibility and whether the most feasible option is clearly identifiable.\n"
            "- If the prior-work section is missing, generic, or does not reveal remaining headroom, prefer `revise` over `approve`.\n"
            "- If the idea is promising but underspecified, prefer `revise` over `reject`.\n"
            "- Use `approve` only if the question is sharp enough that downstream algorithm work could proceed without redefining the problem.\n"
            "- Finish by calling submit_research_idea_review(...) with decision, summary, reviewer_feedback, optional findings, and confidence."
        )
        success_criteria = [
            "Read the idea artifact before judging it.",
            "Judge the idea on problem formulation quality rather than implementation detail.",
            "Explicitly assess narrowness, scope fit, prior-work grounding, and direction-family feasibility.",
            "Return approve/revise/reject with actionable reviewer feedback.",
            "Finish with submit_research_idea_review(...).",
        ]
        return {
            "idea_id": idea_id,
            "title": title,
            "relevant_paths": relevant_paths,
            "task": task,
            "context_notes": context_notes,
            "success_criteria": success_criteria,
        }

    def _run_research_idea_agent_review(self, action: Dict[str, Any]) -> Dict[str, Any]:
        manager = getattr(self.tools_handler, "subagent_manager", None)
        if manager is None:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Research-idea evaluator could not run because the planner runtime has no subagent manager. "
                            "Do not proceed until the idea is reviewed."
                        ),
                        additional_kwargs={"phase": PHASE_COMMENTARY},
                    )
                ],
                "commentary_retries": 0,
            }

        ctx = self._build_research_idea_evaluator_context(action)
        self._emit_event(
            "research_idea_agent_review_started",
            {
                "idea_id": ctx["idea_id"],
                "title": ctx["title"],
                "idea_paths": list(ctx["relevant_paths"]),
            },
        )
        result = self._run_structured_review_subagent(
            manager,
            subagent_type="idea_evaluator",
            task=ctx["task"],
            success_criteria=list(ctx["success_criteria"]),
            context_notes=ctx["context_notes"],
            relevant_paths=list(ctx["relevant_paths"]),
            review_key="research_idea_review",
            required_text_fields=("reviewer_feedback",),
            repair_instruction=(
                "For research-idea reviews, call submit_research_idea_review(...) and include decision, "
                "summary, reviewer_feedback, findings, and confidence."
            ),
        )
        review_payload = _extract_review_payload(result, "research_idea_review")
        _, contract_errors = _validate_structured_review_result(
            result,
            "research_idea_review",
            required_text_fields=("reviewer_feedback",),
        )
        decision = str(review_payload.get("decision") or "").strip().lower()
        feedback = str(review_payload.get("reviewer_feedback") or "").strip()
        confidence = review_payload.get("confidence")
        if contract_errors or result.get("status") != "completed" or decision not in {"approve", "revise", "reject"} or not feedback:
            summary = str(result.get("summary") or "Research-idea evaluator did not produce a usable verdict.").strip()
            findings = [str(item).strip() for item in (result.get("findings") or []) if str(item).strip()]
            details = "\n".join(f"- {item}" for item in findings[:6])
            contract_details = "\n".join(f"- {item}" for item in contract_errors)
            content = (
                f"Research-idea evaluator could not complete a usable review for `{ctx['idea_id']}`.\n"
                f"Summary: {summary}\n"
            )
            if contract_details:
                content += f"Structured contract errors:\n{contract_details}\n"
            content += "Do not proceed yet. Diagnose the idea framing or retry the evaluator."
            if details:
                content += f"\nFindings:\n{details}"
            return {
                "messages": [AIMessage(content=content, additional_kwargs={"phase": PHASE_COMMENTARY})],
                "commentary_retries": 0,
            }

        apply_result = self.tools_handler.review_research_idea(
            idea_id=ctx["idea_id"],
            decision=decision,
            reviewer_feedback=feedback,
        )
        if isinstance(apply_result, str) and apply_result.startswith("Error:"):
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"Research-idea evaluator produced a verdict for `{ctx['idea_id']}`, but applying the review failed:\n"
                            f"{apply_result}\n"
                            "Do not proceed until this is resolved."
                        ),
                        additional_kwargs={"phase": PHASE_COMMENTARY},
                    )
                ],
                "commentary_retries": 0,
            }

        self._emit_event(
            "research_idea_agent_review_completed",
            {
                "idea_id": ctx["idea_id"],
                "title": ctx["title"],
                "decision": decision,
                "confidence": confidence,
                "summary": str(result.get("summary") or "").strip(),
            },
        )
        findings = [str(item).strip() for item in (result.get("findings") or []) if str(item).strip()]
        findings_block = "\n".join(f"- {item}" for item in findings[:8])
        next_step = {
            "approve": "The idea is approved. It can now seed downstream algorithm exploration without redefining the problem.",
            "revise": "The idea is marked revise. Tighten the problem framing before algorithm work continues.",
            "reject": "The idea is rejected. Reformulate the scientific question before proceeding.",
        }[decision]
        content_lines = [
            f"Research-idea evaluator review completed for `{ctx['idea_id']}`.",
            f"Decision: {decision}",
            f"Summary: {str(result.get('summary') or '').strip() or '(no summary)'}",
            "Reviewer feedback:",
            feedback,
        ]
        if findings_block:
            content_lines.extend(["Findings:", findings_block])
        content_lines.append(next_step)
        return {
            "messages": [
                AIMessage(
                    content="\n".join(content_lines),
                    additional_kwargs={"phase": PHASE_COMMENTARY, "research_idea_review_decision": decision},
                )
            ],
            "commentary_retries": 0,
        }

    def _synchronize_planner_phase(self, phase: str, content: str) -> str:
        if self.agent_role != "planner":
            return "completed"

        if phase == PHASE_NEEDS_INPUT:
            existing_need = dict(self.state.get("planner_need") or {})
            if str(self.state.get("planner_phase") or "working") == "needs_input" and existing_need:
                if content and not str(existing_need.get("question") or "").strip():
                    existing_need["question"] = content
                self.state["planner_phase"] = "needs_input"
                self.state["planner_need"] = existing_need
                return "needs_input"

            need = {
                "source": "cytobridge",
                "question": content or "Please provide additional information.",
                "reason": "runtime_v2_needs_user_input",
            }
            self.state["planner_phase"] = "needs_input"
            self.state["planner_need"] = need
            self._emit_event("planner_needs_input", dict(need))
            return "needs_input"

        if str(self.state.get("planner_phase") or "working") == "needs_input":
            self._emit_event("planner_need_resolved", dict(self.state.get("planner_need") or {}))
        self.state["planner_phase"] = "working"
        self.state["planner_need"] = {}
        return "completed"

    def run(
        self,
        user_instruction: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
        *,
        max_turns: Optional[int] = None,
        _allow_checkpoint_reseed_retry: bool = True,
    ) -> Dict[str, Any]:
        attachments = list(attachments or [])
        if attachments and not bool(self.state.get("enable_multimodal", True)):
            raise ValueError("Multimodal input is disabled for this session.")
        turn_input_messages: List[BaseMessage] = []
        if not self.chat_history:
            if self.agent_role == "subagent":
                initial_msg = HumanMessage(content=self._build_user_message_content(user_instruction, attachments))
            else:
                input_path = self.state.get("input_path") or "(not set yet)"
                initial = (
                    f"Current Session Context:\n"
                    f"Input Data: {input_path}\n"
                    f"User Goal: {self.state.get('user_goal', {})}\n\n"
                    f"User Message: {user_instruction}"
                )
                initial_msg = HumanMessage(content=self._build_user_message_content(initial, attachments))
            self.chat_history = add_messages([], [initial_msg])
            turn_input_messages = [initial_msg]
        elif user_instruction:
            user_msg = HumanMessage(content=self._build_user_message_content(user_instruction, attachments))
            self.chat_history = add_messages(self.chat_history, [user_msg])
            turn_input_messages = [user_msg]

        # NOTE:
        # With a checkpointer, graph state is recovered by thread_id. Normally we pass
        # only turn-incremental messages to avoid duplicating history. After an explicit
        # history rewrite such as manual compaction, we temporarily seed the graph with
        # the full rewritten history so the fresh checkpoint matches local chat_history.
        seed_full_history = bool(self._seed_full_history_next_run)
        graph_messages = list(self.chat_history) if seed_full_history else turn_input_messages
        fallback_base_messages = (
            list(self.chat_history)
            if seed_full_history
            else add_messages(self.checkpoint_messages(), turn_input_messages)
        )
        if seed_full_history:
            self._seed_full_history_next_run = False
        turn_system_prompt = self.prompt_builder.build(
            agent_role=self.agent_role,
            agent_id=self.agent_id,
            parent_agent_id=self.parent_agent_id,
            subagent_type=self.subagent_type,
            tool_policy=self.tool_policy,
        )
        graph_state = {
            "messages": graph_messages,
            "commentary_retries": 0,
            "stop_hook_triggers": 0,
            "turn_system_prompt": turn_system_prompt,
            "enable_multimodal": bool(self.state.get("enable_multimodal", True)),
        }
        self._emit_event(
            "status",
            {
                "agent": "cytobridge" if self.agent_role == "planner" else self.agent_id,
                "message": f"Workflow phase: {self.state.get('workflow_phase', 'intake')}",
            },
        )
        final_messages: List[BaseMessage] = []
        graph_config = {
            "recursion_limit": _runtime_recursion_limit(
                max_turns=max_turns,
                enforce_turn_bound=self.agent_role == "planner",
            ),
            "configurable": {
                "thread_id": self.thread_id,
                "checkpoint_ns": self.checkpoint_ns,
            },
        }
        try:
            try:
                graph_events = self.graph.stream(
                    graph_state,
                    stream_mode="updates",
                    config=graph_config,
                    durability="async",
                )
            except TypeError as exc:
                if "durability" not in str(exc):
                    raise
                graph_events = self.graph.stream(graph_state, stream_mode="updates", config=graph_config)
            for event in graph_events:
                if self.stop_check and self.stop_check():
                    stop_msg = AIMessage(content="Stopped by user.", additional_kwargs={"phase": PHASE_FINAL_ANSWER})
                    self.chat_history = add_messages(self.chat_history, [stop_msg])
                    final_messages.append(stop_msg)
                    break
                for node_payload in event.values():
                    if not isinstance(node_payload, dict):
                        continue
                    new_messages = node_payload.get("messages", [])
                    if new_messages:
                        fallback_base_messages = self._apply_message_updates(
                            new_messages,
                            fallback_base_messages=fallback_base_messages,
                        )
                        final_messages.extend(m for m in new_messages if not isinstance(m, RemoveMessage))
        except Exception as exc:
            if seed_full_history:
                self._seed_full_history_next_run = True
            saw_tool_side_effects = any(isinstance(msg, ToolMessage) for msg in final_messages)
            if (
                _allow_checkpoint_reseed_retry
                and self._is_delete_missing_message_error(exc)
                and not saw_tool_side_effects
            ):
                self._emit_event(
                    "status",
                    {
                        "agent": "cytobridge" if self.agent_role == "planner" else self.agent_id,
                        "message": "Detected checkpoint/history drift during runtime update. Resetting checkpoint and retrying the current turn once.",
                    },
                )
                self.replace_chat_history(self.chat_history, reset_checkpoint=True)
                return self.run(
                    "",
                    max_turns=max_turns,
                    _allow_checkpoint_reseed_retry=False,
                )
            raise
        else:
            if seed_full_history:
                self._seed_full_history_next_run = False
        self.state["messages"] = [self._serialize_message(m) for m in self.chat_history]
        if self.checkpoint_callback:
            self.checkpoint_callback()
        final_ai = next((m for m in reversed(final_messages) if isinstance(m, AIMessage)), None)
        status = "completed"
        content = strip_tool_call_transcript(getattr(final_ai, "content", "")) if final_ai else ""
        phase = (
            resolve_message_phase(final_ai)
            or (PHASE_COMMENTARY if final_ai and (getattr(final_ai, "tool_calls", None) or []) else PHASE_FINAL_ANSWER)
        ) if final_ai else PHASE_FINAL_ANSWER
        if content:
            self._emit_event(
                "agent_thought",
                {
                    "agent": "cytobridge" if self.agent_role == "planner" else self.agent_id,
                    "content": content,
                    "phase": phase,
                },
            )
        status = self._synchronize_planner_phase(phase, content)
        return {"status": status, "messages": final_messages, "content": content, "phase": phase}

    @staticmethod
    def _serialize_message(message: BaseMessage) -> dict:
        return {
            "type": message.__class__.__name__,
            "content": getattr(message, "content", ""),
            "additional_kwargs": dict(getattr(message, "additional_kwargs", {}) or {}),
            "tool_calls": list(getattr(message, "tool_calls", []) or []),
        }
