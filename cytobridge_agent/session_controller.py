"""Shared session controller for CLI, interactive, and Web adapters."""
from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .conversation_store import ConversationStore
from .job_registry import JobRegistry
from .runtime_events import runtime_event
from .runtime_snapshot import build_runtime_snapshot, build_saved_session_snapshot, legacy_status_from_snapshot
from .runtime_state import RuntimeState, RuntimeStatus
from .session_ownership import SessionOwnershipRegistry
from .utils.llm_factory import normalize_llm_reasoning_effort
from .utils.llm_providers import get_llm_provider, normalize_llm_provider

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from .interactive import InteractiveSession
else:
    BaseChatModel = Any
    InteractiveSession = Any


LlmFactory = Callable[[Optional[Dict[str, Any]]], BaseChatModel]


@dataclass
class TurnResult:
    status: str
    session_id: str
    response: str


class SessionController:
    """Process-local controller around ``InteractiveSession``.

    This is intentionally UI-agnostic. It owns one active session in the current
    process and exposes the same operations that CLI and Web adapters need.
    """

    def __init__(
        self,
        llm_factory: LlmFactory,
        *,
        initial_config: Optional[Dict[str, Any]] = None,
        store: Optional[ConversationStore] = None,
    ) -> None:
        self.llm_factory = llm_factory
        self.initial_config: Dict[str, Any] = dict(initial_config or {})
        self.store = store or ConversationStore()
        self.jobs = JobRegistry()
        self.ownership = SessionOwnershipRegistry()
        self.owner_id = f"{os.getpid()}-{uuid.uuid4().hex[:10]}"
        self.owner_kind = "runtime"
        self.runtime_state = RuntimeState()
        self.active_session: Optional[InteractiveSession] = None
        self.stop_requested = False
        self._lock = threading.RLock()

    def _apply_session_runtime_config(self, session: InteractiveSession) -> None:
        """Apply UI-agnostic runtime settings that must survive CLI resume.

        Web sessions already install these values when the session is created.
        CLI sessions share the same controller now, so the controller is the
        right place to make stop-hook state consistent across create/resume.
        """
        runtime_keys = (
            "llm_provider",
            "llm_model",
            "llm_base_url",
            "llm_thinking_level",
            "stop_hook_enabled",
            "stop_hook_mode",
            "stop_hook_prompt",
            "stop_hook_max_triggers",
        )
        changed = False
        for key in runtime_keys:
            if key in self.initial_config and self.initial_config[key] is not None:
                session.state[key] = self.initial_config[key]
                changed = True
        if changed:
            planner = getattr(session, "planner", None)
            if planner is not None:
                planner.refresh_tooling()
            try:
                session._save_conversation()
            except Exception:
                pass

    def _make_llm(self, overrides: Optional[Dict[str, Any]] = None) -> BaseChatModel:
        cfg = dict(self.initial_config)
        cfg.update({k: v for k, v in dict(overrides or {}).items() if v is not None})
        return self.llm_factory(cfg)

    def _install_stop_check(self, session: InteractiveSession) -> None:
        def check_stop() -> bool:
            return self.stop_requested

        planner = getattr(session, "planner", None)
        if planner is not None:
            planner.stop_check = check_stop

    def _emit_runtime_event(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        session: Optional[InteractiveSession] = None,
        turn_id: Optional[str] = None,
    ) -> None:
        target = session or self.active_session
        session_id = getattr(target, "session_id", None)
        if not session_id:
            return
        event = runtime_event(
            event_type,
            data or {},
            session_id=session_id,
            turn_id=turn_id,
            source="runtime",
        )
        try:
            self.store.append_timeline_event(session_id, event)
        except Exception:
            # Runtime events should improve observability, not break agent turns.
            pass

    def new_session(
        self,
        *,
        input_path: Optional[str],
        user_goal: Dict[str, Any],
        output_dir: Optional[str] = None,
        enable_multimodal: bool = True,
    ) -> InteractiveSession:
        with self._lock:
            from .interactive import InteractiveSession

            llm = self._make_llm()
            session = InteractiveSession(
                input_path=input_path,
                user_goal=user_goal,
                llm=llm,
                output_dir=output_dir,
                enable_multimodal=enable_multimodal,
            )
            self._install_stop_check(session)
            self._apply_session_runtime_config(session)
            self.active_session = session
            self.stop_requested = False
            self.owner_kind = "runtime"
            self.ownership.acquire(session_id=session.session_id, owner_id=self.owner_id, owner_kind=self.owner_kind)
            self.runtime_state.set(RuntimeStatus.IDLE, session_id=session.session_id, message="session started")
            self._emit_runtime_event("session_started", {"session_id": session.session_id}, session=session)
            return session

    def resume_session(self, session_id: str) -> InteractiveSession:
        with self._lock:
            from .interactive import InteractiveSession

            llm = self._make_llm()
            session = InteractiveSession.from_checkpoint(session_id, llm)
            self._install_stop_check(session)
            self._apply_session_runtime_config(session)
            self.active_session = session
            self.stop_requested = False
            self.owner_kind = "runtime"
            self.ownership.acquire(session_id=session.session_id, owner_id=self.owner_id, owner_kind=self.owner_kind)
            self.runtime_state.set(RuntimeStatus.IDLE, session_id=session.session_id, message="session resumed")
            self._emit_runtime_event("session_resumed", {"session_id": session.session_id}, session=session)
            return session

    def adopt_session(
        self,
        session: InteractiveSession,
        *,
        stop_check: Optional[Callable[[], bool]] = None,
        owner_kind: str = "runtime",
    ) -> InteractiveSession:
        """Adopt a session created by another adapter, such as the Web UI."""
        with self._lock:
            if stop_check is None:
                self._install_stop_check(session)
            else:
                planner = getattr(session, "planner", None)
                if planner is not None:
                    planner.stop_check = stop_check
            self._apply_session_runtime_config(session)
            self.active_session = session
            self.stop_requested = False
            self.owner_kind = str(owner_kind or "runtime")
            self.ownership.acquire(session_id=session.session_id, owner_id=self.owner_id, owner_kind=self.owner_kind)
            self.runtime_state.set(RuntimeStatus.IDLE, session_id=session.session_id, message="session adopted")
            self._emit_runtime_event("session_adopted", {"session_id": session.session_id}, session=session)
            return session

    def run_turn(self, message: str, attachments: Optional[List[Dict[str, Any]]] = None) -> TurnResult:
        session = self.active_session
        if session is None:
            raise RuntimeError("Session not initialized.")
        self.ownership.heartbeat(session.session_id, self.owner_id)
        turn_id = uuid.uuid4().hex
        self.stop_requested = False
        self.runtime_state.begin_turn(session_id=session.session_id, turn_id=turn_id, message="turn running")
        self._emit_runtime_event(
            "turn_started",
            {
                "session_id": session.session_id,
                "message_chars": len(str(message or "")),
                "attachment_count": len(attachments or []),
            },
            session=session,
            turn_id=turn_id,
        )
        try:
            response = session.run_turn(message, attachments=attachments)
        except Exception as exc:
            elapsed = self.runtime_state.finish_turn(status=RuntimeStatus.FAILED, message="turn failed")
            self.runtime_state.last_error = str(exc)
            self._emit_runtime_event(
                "turn_failed",
                {
                    "session_id": session.session_id,
                    "elapsed_seconds": elapsed,
                    "error": str(exc),
                },
                session=session,
                turn_id=turn_id,
            )
            raise
        elapsed = self.runtime_state.finish_turn(status=RuntimeStatus.IDLE, message="turn complete")
        self._emit_runtime_event(
            "turn_complete",
            {
                "session_id": session.session_id,
                "elapsed_seconds": elapsed,
                "response_chars": len(str(response or "")),
            },
            session=session,
            turn_id=turn_id,
        )
        return TurnResult(status="success", session_id=session.session_id, response=response)

    def request_stop(self) -> Dict[str, Any]:
        self.stop_requested = True
        session_id = getattr(self.active_session, "session_id", None)
        self.runtime_state.set(RuntimeStatus.STOPPING, session_id=session_id, message="stop requested")
        self._emit_runtime_event("stop_requested", {"session_id": session_id})
        return {"status": "success", "message": "Stop requested for the active session."}

    def switch_model(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        thinking_level: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        auth_mode: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        if provider is not None:
            updates["llm_provider"] = normalize_llm_provider(provider)
        if model is not None:
            updates["llm_model"] = model
        if thinking_level is not None:
            updates["llm_thinking_level"] = normalize_llm_reasoning_effort(thinking_level, default="low")
        if base_url is not None:
            updates["llm_base_url"] = base_url
        if api_key is not None:
            updates["llm_api_key"] = api_key
        if auth_mode is not None:
            updates["llm_auth_mode"] = auth_mode
        if profile_id is not None:
            updates["llm_profile_id"] = profile_id

        self.initial_config.update(updates)
        session = self.active_session
        restore_diag: Dict[str, Any] = {}
        if session is not None:
            llm = self._make_llm()
            restore_diag = session.update_llm(llm)
            session.planner.refresh_tooling()
            self._emit_runtime_event(
                "model_switched",
                {
                    "provider": self.initial_config.get("llm_provider"),
                    "model": self.initial_config.get("llm_model"),
                    "thinking_level": self.initial_config.get("llm_thinking_level"),
                    "resume_diagnostics": restore_diag,
                },
                session=session,
            )
        return {
            "status": "success",
            "provider": self.initial_config.get("llm_provider"),
            "model": self.initial_config.get("llm_model"),
            "thinking_level": self.initial_config.get("llm_thinking_level"),
            "resume_diagnostics": restore_diag,
        }

    def compact_active_session(self) -> Dict[str, Any]:
        session = self.active_session
        if session is None:
            return {"status": "error", "message": "Session not initialized."}
        result = session.compact_context()
        self._emit_runtime_event("session_compacted", {"session_id": session.session_id, "compact_result": result}, session=session)
        return {"status": "success", "session_id": session.session_id, "compact_result": result}

    def status(self) -> Dict[str, Any]:
        session = self.active_session
        session_id = getattr(session, "session_id", None)
        ownership = self.ownership.heartbeat(session_id, self.owner_id) if session_id else {}
        snapshot = build_runtime_snapshot(
            session=session,
            runtime_state=self.runtime_state,
            jobs=self.jobs,
            initial_config=self.initial_config,
            stop_requested=self.stop_requested,
            ownership=ownership,
        )
        status = legacy_status_from_snapshot(snapshot)
        status["snapshot"] = snapshot
        return status

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.store.list_conversations(limit=limit)

    def latest_session_id(self) -> Optional[str]:
        sessions = self.list_sessions(limit=1)
        if not sessions:
            return None
        return str(sessions[0].get("session_id") or "") or None

    def session_status(self, session_id: str) -> Dict[str, Any]:
        data = self.store.get_conversation(session_id, degrade_large_snapshot=True)
        if not data:
            return {"status": "error", "message": f"Session {session_id} not found."}
        metadata = dict(data.get("metadata") or {})
        agent_state = dict(data.get("agent_state") or {})
        snapshot = build_saved_session_snapshot(
            session_id=session_id,
            metadata=metadata,
            agent_state=agent_state,
            jobs=self.jobs,
            ownership=self.ownership.get(session_id),
        )
        return {
            "status": "success",
            "session_id": session_id,
            "metadata": metadata,
            "input_path": metadata.get("input_path") or agent_state.get("input_path"),
            "output_dir": metadata.get("output_dir") or agent_state.get("output_dir"),
            "conversation_turn": agent_state.get("conversation_turn"),
            "planner_phase": agent_state.get("planner_phase"),
            "preview": data.get("preview"),
            "resume_diagnostics": data.get("resume_diagnostics"),
            "snapshot": snapshot,
        }

    def tail_events(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        return self.store.get_timeline_events(session_id, limit=limit)

    def compact_session(self, session_id: str, keep_last_turns: int = 6) -> Dict[str, Any]:
        if self.active_session and self.active_session.session_id == session_id:
            return self.compact_active_session()
        result = self.store.compact_conversation(session_id, keep_last_turns=keep_last_turns)
        status = "success" if result.get("ok", True) else "error"
        return {"status": status, "session_id": session_id, "compact_result": result}

    def release_active_session(self) -> Dict[str, Any]:
        session_id = getattr(self.active_session, "session_id", None)
        if not session_id:
            return {"status": "noop", "message": "No active session to release."}
        released = self.ownership.release(session_id, self.owner_id)
        self._emit_runtime_event("session_released", {"session_id": session_id, "ownership": released})
        return {"status": "success" if released else "noop", "session_id": session_id, "ownership": released}

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        deleted = self.store.delete_conversation(session_id)
        return {
            "status": "success" if deleted else "error",
            "session_id": session_id,
            "message": "Session deleted." if deleted else "Session not found.",
        }


def print_jsonl_event(event_type: str, payload: Dict[str, Any]) -> None:
    print(json.dumps({"type": event_type, **payload}, ensure_ascii=False), flush=True)


def format_session_listing(sessions: List[Dict[str, Any]]) -> str:
    from .cli_terminal import format_session_listing as _format_session_listing

    return _format_session_listing(sessions, numbered=True)
