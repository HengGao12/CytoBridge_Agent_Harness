"""Shared runtime status primitives for CellCompass adapters."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .runtime_events import now_iso


class RuntimeStatus:
    """Canonical process-local session states."""

    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    COMPLETED = "completed"
    NEEDS_USER = "needs_user"
    ACTION_REQUIRED = "action_required"


TERMINAL_STATUSES = {
    RuntimeStatus.FAILED,
    RuntimeStatus.COMPLETED,
    RuntimeStatus.NEEDS_USER,
    RuntimeStatus.ACTION_REQUIRED,
}


@dataclass
class RuntimeState:
    """Small status object shared by CLI, TUI, Web, and supervisor adapters."""

    status: str = RuntimeStatus.IDLE
    session_id: Optional[str] = None
    message: str = ""
    since: str = field(default_factory=now_iso)
    last_error: Optional[str] = None
    active_turn_id: Optional[str] = None
    active_turn_started_at: Optional[str] = None
    _active_turn_monotonic: Optional[float] = field(default=None, repr=False)

    def set(
        self,
        status: str,
        *,
        session_id: Optional[str] = None,
        message: str = "",
        error: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> None:
        self.status = str(status or RuntimeStatus.IDLE)
        if session_id is not None:
            self.session_id = str(session_id)
        self.message = str(message or "")
        self.since = now_iso()
        self.last_error = str(error) if error else None
        if turn_id is not None:
            self.active_turn_id = str(turn_id)

    def begin_turn(self, *, session_id: str, turn_id: str, message: str = "") -> None:
        self.status = RuntimeStatus.RUNNING
        self.session_id = str(session_id)
        self.message = str(message or "turn running")
        self.since = now_iso()
        self.last_error = None
        self.active_turn_id = str(turn_id)
        self.active_turn_started_at = self.since
        self._active_turn_monotonic = time.perf_counter()

    def finish_turn(self, *, status: str = RuntimeStatus.IDLE, message: str = "") -> float:
        elapsed = self.active_elapsed_seconds()
        self.status = str(status or RuntimeStatus.IDLE)
        self.message = str(message or "")
        self.since = now_iso()
        self.active_turn_id = None
        self.active_turn_started_at = None
        self._active_turn_monotonic = None
        return elapsed

    def active_elapsed_seconds(self) -> float:
        if self._active_turn_monotonic is None:
            return 0.0
        return max(0.0, time.perf_counter() - self._active_turn_monotonic)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "session_id": self.session_id,
            "message": self.message,
            "since": self.since,
            "last_error": self.last_error,
            "active_turn_id": self.active_turn_id,
            "active_turn_started_at": self.active_turn_started_at,
            "active_elapsed_seconds": self.active_elapsed_seconds(),
            "is_terminal": self.status in TERMINAL_STATUSES,
        }

