"""Session-scoped ownership records for mutable algorithm artifacts."""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .runtime_events import now_iso


class AlgorithmOwnershipError(RuntimeError):
    """Raised when another saved session already owns an algorithm."""


class AlgorithmOwnershipRegistry:
    """Atomic JSON registry preventing cross-session algorithm writes.

    Ownership is keyed by saved CytoBridge session id, not process id. This
    intentionally allows a normal resume of the same session to continue an
    algorithm, while blocking a different session from mutating that algorithm's
    workspace, campaign, or experiment registry.
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else Path.home() / ".cellcompass" / "algorithm_ownership.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _normalize_algorithm_id(algorithm_id: str) -> str:
        value = str(algorithm_id or "").strip().lower()
        if not value:
            raise ValueError("algorithm_id is required")
        return value

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        value = str(session_id or "").strip()
        if not value:
            raise ValueError("session_id is required for algorithm ownership")
        return value

    def _read_unlocked(self) -> Dict[str, Dict[str, Any]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        records = raw.get("algorithms", raw) if isinstance(raw, dict) else {}
        if not isinstance(records, dict):
            return {}
        return {str(key).strip().lower(): dict(value or {}) for key, value in records.items()}

    def _write_unlocked(self, records: Dict[str, Dict[str, Any]]) -> None:
        payload = {"schema_version": 1, "updated_at": now_iso(), "algorithms": records}
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

    def acquire(
        self,
        *,
        algorithm_id: str,
        session_id: str,
        action: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        algo_id = self._normalize_algorithm_id(algorithm_id)
        owner_session_id = self._normalize_session_id(session_id)
        with self._locked():
            records = self._read_unlocked()
            existing = dict(records.get(algo_id) or {})
            existing_session = str(existing.get("owner_session_id") or "").strip()
            existing_status = str(existing.get("status") or "active").strip().lower() or "active"
            if (
                existing
                and existing_status == "active"
                and existing_session
                and existing_session != owner_session_id
                and not force
            ):
                raise AlgorithmOwnershipError(
                    f"Algorithm '{algo_id}' is owned by session '{existing_session}' "
                    f"(last_action={existing.get('last_action') or 'unknown'}). "
                    f"Session '{owner_session_id}' cannot {action or 'mutate'} it. "
                    "Resume the owner session, create a new algorithm_id, or perform an explicit supervised ownership transfer."
                )

            now = now_iso()
            record = dict(existing) if existing_session == owner_session_id else {}
            record.update(
                {
                    "algorithm_id": algo_id,
                    "owner_session_id": owner_session_id,
                    "owner_pid": os.getpid(),
                    "status": "active",
                    "updated_at": now,
                    "last_action": str(action or "").strip(),
                }
            )
            record.setdefault("acquired_at", now)
            records[algo_id] = record
            self._write_unlocked(records)
            return dict(record)

    def get(self, algorithm_id: str) -> Dict[str, Any]:
        algo_id = self._normalize_algorithm_id(algorithm_id)
        with self._locked():
            return dict(self._read_unlocked().get(algo_id) or {})

    def list(self) -> Dict[str, Dict[str, Any]]:
        with self._locked():
            return {key: dict(value) for key, value in self._read_unlocked().items()}
