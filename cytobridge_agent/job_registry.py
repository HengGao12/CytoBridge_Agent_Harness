"""Lightweight local job registry for long-running runtime tasks."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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


@dataclass
class RuntimeJob:
    job_id: str
    kind: str
    session_id: Optional[str] = None
    status: str = "running"
    pid: Optional[int] = None
    command: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    heartbeat_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "session_id": self.session_id,
            "status": self.status,
            "pid": self.pid,
            "pid_alive": _pid_alive(self.pid) if self.pid is not None else None,
            "command": self.command,
            "artifacts": dict(self.artifacts or {}),
            "metadata": dict(self.metadata or {}),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "heartbeat_at": self.heartbeat_at,
        }


class JobRegistry:
    """Atomic JSON registry for process and training jobs.

    This is intentionally simple: it gives CLI/Web/supervisors a single place to
    inspect active long-running jobs before introducing heavier queueing.
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else Path.home() / ".cellcompass" / "runtime_jobs.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> Dict[str, Dict[str, Any]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if isinstance(raw, dict):
            jobs = raw.get("jobs", raw)
            if isinstance(jobs, dict):
                return {str(k): dict(v or {}) for k, v in jobs.items()}
        return {}

    def _write(self, jobs: Dict[str, Dict[str, Any]]) -> None:
        payload = {"schema_version": 1, "updated_at": now_iso(), "jobs": jobs}
        fd, tmp_name = tempfile.mkstemp(prefix=self.path.name, suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_name, self.path)
        finally:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except Exception:
                pass

    def upsert(
        self,
        *,
        job_id: str,
        kind: str,
        session_id: Optional[str] = None,
        status: str = "running",
        pid: Optional[int] = None,
        command: Optional[str] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        jobs = self._read()
        existing = dict(jobs.get(job_id) or {})
        now = now_iso()
        record = {
            **existing,
            "job_id": str(job_id),
            "kind": str(kind),
            "session_id": session_id,
            "status": str(status or "running"),
            "pid": pid,
            "command": command,
            "artifacts": dict(artifacts or existing.get("artifacts") or {}),
            "metadata": dict(metadata or existing.get("metadata") or {}),
            "created_at": existing.get("created_at") or now,
            "updated_at": now,
            "heartbeat_at": now,
        }
        jobs[str(job_id)] = record
        self._write(jobs)
        return self._decorate(record)

    def mark(self, job_id: str, status: str, *, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        jobs = self._read()
        record = dict(jobs.get(job_id) or {"job_id": str(job_id), "kind": "unknown"})
        record["status"] = str(status)
        record["updated_at"] = now_iso()
        if metadata:
            merged = dict(record.get("metadata") or {})
            merged.update(metadata)
            record["metadata"] = merged
        jobs[str(job_id)] = record
        self._write(jobs)
        return self._decorate(record)

    def heartbeat(self, job_id: str) -> Optional[Dict[str, Any]]:
        jobs = self._read()
        if job_id not in jobs:
            return None
        jobs[job_id]["heartbeat_at"] = now_iso()
        jobs[job_id]["updated_at"] = jobs[job_id]["heartbeat_at"]
        self._write(jobs)
        return self._decorate(jobs[job_id])

    def list(self, *, session_id: Optional[str] = None, active_only: bool = False) -> List[Dict[str, Any]]:
        records = [self._decorate(record) for record in self._read().values()]
        if session_id:
            records = [item for item in records if item.get("session_id") == session_id]
        if active_only:
            records = [
                item
                for item in records
                if item.get("status") in {"running", "queued", "stopping"} or item.get("pid_alive") is True
            ]
        records.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return records

    def stale_jobs(self) -> List[Dict[str, Any]]:
        stale = []
        for item in self.list():
            if item.get("status") in {"running", "stopping"} and item.get("pid") is not None and not item.get("pid_alive"):
                stale.append(item)
        return stale

    @staticmethod
    def _decorate(record: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(record or {})
        item["pid_alive"] = _pid_alive(item.get("pid")) if item.get("pid") is not None else None
        return item

