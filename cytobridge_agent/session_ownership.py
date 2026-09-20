"""Process-level ownership records for active runtime sessions."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .runtime_events import now_iso


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except Exception:
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


class SessionOwnershipError(RuntimeError):
    """Raised when a live process already owns a session."""


class SessionOwnershipRegistry:
    """Atomic JSON registry preventing accidental cross-process session reuse."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else Path.home() / ".cellcompass" / "runtime_sessions.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> Dict[str, Dict[str, Any]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        sessions = raw.get("sessions", raw) if isinstance(raw, dict) else {}
        if not isinstance(sessions, dict):
            return {}
        return {str(key): dict(value or {}) for key, value in sessions.items()}

    def _write(self, sessions: Dict[str, Dict[str, Any]]) -> None:
        payload = {"schema_version": 1, "updated_at": now_iso(), "sessions": sessions}
        fd, tmp_name = tempfile.mkstemp(prefix=self.path.name, suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(tmp_name, self.path)
        finally:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except Exception:
                pass

    def _decorate(self, record: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(record or {})
        item["pid_alive"] = _pid_alive(item.get("pid")) if item.get("pid") is not None else None
        item["is_current_process"] = item.get("pid") == os.getpid()
        return item

    def acquire(self, *, session_id: str, owner_id: str, owner_kind: str, force: bool = False) -> Dict[str, Any]:
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        sessions = self._read()
        existing = self._decorate(sessions.get(session_id) or {})
        existing_pid_alive = bool(existing.get("pid_alive"))
        existing_same_process = bool(existing.get("is_current_process"))
        existing_owner = str(existing.get("owner_id") or "")
        if (
            existing
            and existing_pid_alive
            and not existing_same_process
            and existing_owner != str(owner_id)
            and not force
        ):
            raise SessionOwnershipError(
                f"Session {session_id} is already owned by live process pid={existing.get('pid')} owner={existing_owner}"
            )
        now = now_iso()
        record = {
            "session_id": session_id,
            "owner_id": str(owner_id),
            "owner_kind": str(owner_kind or "runtime"),
            "pid": os.getpid(),
            "status": "active",
            "acquired_at": existing.get("acquired_at") if existing_same_process and existing_owner == str(owner_id) else now,
            "updated_at": now,
            "heartbeat_at": now,
            "replaced_owner_id": existing_owner if existing and existing_owner != str(owner_id) else "",
        }
        sessions[session_id] = record
        self._write(sessions)
        return self._decorate(record)

    def heartbeat(self, session_id: str, owner_id: str) -> Optional[Dict[str, Any]]:
        sessions = self._read()
        record = dict(sessions.get(str(session_id)) or {})
        if not record or str(record.get("owner_id") or "") != str(owner_id):
            return None
        now = now_iso()
        record["heartbeat_at"] = now
        record["updated_at"] = now
        sessions[str(session_id)] = record
        self._write(sessions)
        return self._decorate(record)

    def release(self, session_id: str, owner_id: str) -> Optional[Dict[str, Any]]:
        sessions = self._read()
        record = dict(sessions.get(str(session_id)) or {})
        if not record or str(record.get("owner_id") or "") != str(owner_id):
            return None
        record["status"] = "released"
        record["released_at"] = now_iso()
        record["updated_at"] = record["released_at"]
        sessions[str(session_id)] = record
        self._write(sessions)
        return self._decorate(record)

    def get(self, session_id: str) -> Dict[str, Any]:
        return self._decorate(self._read().get(str(session_id)) or {})

    def list(self) -> Dict[str, Dict[str, Any]]:
        return {session_id: self._decorate(record) for session_id, record in self._read().items()}
