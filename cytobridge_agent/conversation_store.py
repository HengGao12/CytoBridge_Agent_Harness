"""Conversation persistence with V2 snapshots + append-only events."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.load import dumpd, load
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy should exist, but persistence should not hard-fail if it does not.
    np = None


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _pid_is_alive(pid: Any) -> bool:
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


def _json_safe(value: Any) -> Any:
    """Recursively convert values into JSON-serializable Python primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if np is not None:
        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


class ConversationStore:
    """Persist and restore conversations under ``~/.cellcompass/conversations``."""

    schema_version = 2
    _LLM_USAGE_PREVIEW_CHARS = 4000
    _TOOL_OUTPUT_PREVIEW_CHARS = 12000
    _NON_TIMELINE_EVENT_TYPES = {"resume_migrated", "turn_started"}
    _LIST_SNAPSHOT_MAX_BYTES = int(os.environ.get("CYTOBRIDGE_CONVERSATION_LIST_SNAPSHOT_MAX_BYTES", 16 * 1024 * 1024))
    _RESUME_SNAPSHOT_MAX_BYTES = int(os.environ.get("CYTOBRIDGE_CONVERSATION_RESUME_SNAPSHOT_MAX_BYTES", 128 * 1024 * 1024))

    def __init__(self, base_dir: Optional[str] = None, stale_ms: int = 30 * 60 * 1000):
        self.base_dir = Path(base_dir) if base_dir else Path.home() / ".cellcompass"
        self.conversations_dir = self.base_dir / "conversations"
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.stale_ms = stale_ms

    def generate_session_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def _get_conversation_path(self, session_id: str) -> Path:
        return self.conversations_dir / f"{session_id}.json"

    def _get_events_path(self, session_id: str) -> Path:
        return self.conversations_dir / f"{session_id}.events.jsonl"

    def _get_lock_path(self, session_id: str) -> Path:
        return self.conversations_dir / f"{session_id}.lock"

    @contextmanager
    def _session_lock(self, session_id: str, timeout_ms: int = 5000):
        lock_path = self._get_lock_path(session_id)
        token = f"{os.getpid()}-{time.time_ns()}"
        start = time.time()

        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                payload = {
                    "pid": os.getpid(),
                    "created_at": _utc_now_iso(),
                    "created_at_ms": int(time.time() * 1000),
                    "token": token,
                }
                os.write(fd, json.dumps(payload).encode("utf-8"))
                os.close(fd)
                break
            except FileExistsError:
                stale = False
                try:
                    raw = lock_path.read_text(encoding="utf-8")
                    info = json.loads(raw)
                    created_ms = int(info.get("created_at_ms", 0))
                    lock_pid = info.get("pid")
                    pid_alive = _pid_is_alive(lock_pid)
                    stale = not pid_alive
                    if not stale:
                        stale = created_ms > 0 and (int(time.time() * 1000) - created_ms) > self.stale_ms
                except Exception:
                    stale = True

                if stale:
                    try:
                        lock_path.unlink(missing_ok=True)
                        continue
                    except Exception:
                        pass

                if (time.time() - start) * 1000 > timeout_ms:
                    raise TimeoutError(f"Timeout acquiring lock for session {session_id}")
                time.sleep(0.05)

        try:
            yield
        finally:
            try:
                if lock_path.exists():
                    raw = lock_path.read_text(encoding="utf-8")
                    info = json.loads(raw)
                    if info.get("token") == token:
                        lock_path.unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to release lock for %s", session_id, exc_info=True)

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except Exception:
            return 0

    @staticmethod
    def _file_mtime_iso(path: Path) -> str:
        try:
            return datetime.utcfromtimestamp(path.stat().st_mtime).replace(microsecond=0).isoformat() + "Z"
        except Exception:
            return _utc_now_iso()

    @staticmethod
    def _format_bytes(size: int) -> str:
        value = float(max(size, 0))
        units = ["B", "KB", "MB", "GB", "TB"]
        for unit in units:
            if value < 1024 or unit == units[-1]:
                if unit == "B":
                    return f"{int(value)} {unit}"
                return f"{value:.1f} {unit}"
            value /= 1024

    def _large_snapshot_warning(self, size: int) -> str:
        return (
            f"Snapshot is {self._format_bytes(size)}, so Web skipped full JSON parsing "
            "and will recover a minimal view from the event log when opened."
        )

    def _large_snapshot_event_summary(self, session_id: str) -> Dict[str, str]:
        title = ""
        preview = ""
        updated_at = ""
        events_path = self._get_events_path(session_id)
        if not events_path.exists():
            return {}

        try:
            with events_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    payload = record.get("payload") or {}
                    updated_at = str(record.get("ts") or updated_at)
                    event_type = str(record.get("event_type") or "")
                    if event_type == "user_message" and not title:
                        content = str(payload.get("content") or "").strip()
                        if content:
                            title = content[:50] + ("..." if len(content) > 50 else "")
                    elif event_type in {"assistant_message", "turn_complete"}:
                        content = str(payload.get("content") or payload.get("response") or "").strip()
                        if content:
                            preview = content[:100] + ("..." if len(content) > 100 else "")
                    elif event_type == "turn_finished":
                        metadata = payload.get("metadata") or {}
                        if metadata.get("title"):
                            title = str(metadata.get("title") or title)
                        if metadata.get("updated_at"):
                            updated_at = str(metadata.get("updated_at") or updated_at)
        except Exception:
            return {}

        return {
            "title": title,
            "preview": preview,
            "updated_at": updated_at,
        }

    def _large_snapshot_listing(self, conv_file: Path) -> Dict[str, Any]:
        session_id = conv_file.stem
        size = self._file_size(conv_file)
        summary = self._large_snapshot_event_summary(session_id)
        updated_at = summary.get("updated_at") or self._file_mtime_iso(conv_file)
        return {
            "session_id": session_id,
            "metadata": {
                "title": summary.get("title") or f"Large conversation {session_id}",
                "created_at": updated_at,
                "updated_at": updated_at,
                "message_count": 0,
                "is_resumable": True,
                "resume_warning": self._large_snapshot_warning(size),
                "snapshot_size_bytes": size,
            },
            "preview": summary.get("preview")
            or "Preview unavailable because the saved conversation snapshot is very large.",
        }

    def _large_snapshot_placeholder(self, session_id: str, conv_path: Path) -> Dict[str, Any]:
        size = self._file_size(conv_path)
        updated_at = self._file_mtime_iso(conv_path)
        return {
            "schema_version": 2,
            "session_id": session_id,
            "metadata": {
                "title": f"Large conversation {session_id}",
                "created_at": updated_at,
                "updated_at": updated_at,
                "message_count": 0,
                "is_resumable": False,
                "resume_warning": self._large_snapshot_warning(size),
                "snapshot_size_bytes": size,
            },
            "messages": [],
            "agent_state": {},
            "agent_histories": {},
            "agent_snapshots": {},
            "events_log": [],
            "history_revision": 0,
            "compaction_stats": self._empty_compaction_stats(),
        }

    def _read_snapshot_for_update(self, session_id: str, conv_path: Path) -> Dict[str, Any]:
        del session_id
        if not conv_path.exists():
            return {}
        return self._read_json(conv_path) or {}

    def _write_json_atomic(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
        safe_payload = _json_safe(payload)
        tmp.write_text(json.dumps(safe_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _append_event_line(self, session_id: str, record: Dict[str, Any]) -> None:
        events_path = self._get_events_path(session_id)
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")

    _RESUME_TIMELINE_EVENT_LIMIT = int(os.environ.get("CYTOBRIDGE_CONVERSATION_RESUME_TIMELINE_EVENT_LIMIT", 2000))

    def _iter_events(self, session_id: str, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
        events_path = self._get_events_path(session_id)
        if not events_path.exists():
            return []

        max_items = int(limit or 0)
        records: Any = deque(maxlen=max_items) if max_items > 0 else []
        try:
            with events_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            return []
        return list(records)

    def _event_record(
        self,
        session_id: str,
        revision: int,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "ts": ts or _utc_now_iso(),
            "session_id": session_id,
            "revision": int(revision),
            "event_type": event_type,
            "payload": payload or {},
        }

    @staticmethod
    def _compact_text(value: Any, max_chars: int) -> str:
        text = str(value or "")
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]"

    def _compact_timeline_payload(self, event_type: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        data = dict(payload or {})
        if event_type == "llm_usage":
            if "response_text" in data:
                data["response_text"] = self._compact_text(
                    data.get("response_text"),
                    self._LLM_USAGE_PREVIEW_CHARS,
                )
            return data
        if event_type == "tool_output":
            display_content = data.get("display_content")
            content = display_content if display_content else data.get("content")
            data["content"] = self._compact_text(
                content,
                self._TOOL_OUTPUT_PREVIEW_CHARS,
            )
            data.pop("display_content", None)
            return data
        return data

    def _timeline_event_from_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        event = {
            "type": str(record.get("event_type") or "timeline_event"),
            "timestamp": record.get("ts") or _utc_now_iso(),
            "data": dict(record.get("payload") or {}),
        }
        if record.get("event_id"):
            event["event_id"] = str(record.get("event_id") or "")
        if record.get("turn_id"):
            event["turn_id"] = str(record.get("turn_id") or "")
        if record.get("process_id"):
            event["process_id"] = str(record.get("process_id") or "")
        return event

    def _is_timeline_record(self, record: Dict[str, Any]) -> bool:
        if str(record.get("stream") or "") == "timeline":
            return True
        event_type = str(record.get("event_type") or "")
        return event_type not in self._NON_TIMELINE_EVENT_TYPES

    def _get_preview(self, messages: List[Dict[str, str]], max_length: int = 100) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                return content[:max_length] + ("..." if len(content) > max_length else "")
        return ""

    def _derive_title(self, metadata: Dict[str, Any], messages: List[Dict[str, Any]]) -> str:
        title = metadata.get("title")
        if title:
            return str(title)

        for msg in messages:
            if msg.get("role") == "user":
                content = str(msg.get("content", "")).strip()
                if content:
                    return content[:50] + ("..." if len(content) > 50 else "")

        input_path = metadata.get("input_path")
        return Path(input_path).name if input_path else "Untitled"

    def _empty_compaction_stats(self) -> Dict[str, Any]:
        return {
            "count": 0,
            "last_at": "",
            "last_before_tokens": 0,
            "last_after_tokens": 0,
            "last_reason": "",
        }

    def _to_v2_snapshot(self, session_id: str, raw: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        if int(raw.get("schema_version", 1)) >= 2:
            data = dict(raw)
            data.setdefault("schema_version", 2)
            data.setdefault("history_revision", 0)
            data.setdefault("compaction_stats", self._empty_compaction_stats())
            data.setdefault("agent_snapshots", {})
            data.setdefault("events_log", [])
            return data, False

        metadata = dict(raw.get("metadata") or {})
        agent_histories = dict(raw.get("agent_histories") or {})
        agent_snapshots = dict(raw.get("agent_snapshots") or {})
        for name, hist in agent_histories.items():
            agent_snapshots.setdefault(name, {"chat_history": hist})

        migrated = {
            "schema_version": 2,
            "session_id": raw.get("session_id") or session_id,
            "metadata": metadata,
            "messages": list(raw.get("messages") or []),
            "agent_state": dict(raw.get("agent_state") or {}),
            "agent_histories": agent_histories,
            "agent_snapshots": agent_snapshots,
            "events_log": list(raw.get("events_log") or []),
            "history_revision": int(raw.get("history_revision", 0)),
            "compaction_stats": dict(raw.get("compaction_stats") or self._empty_compaction_stats()),
        }
        return migrated, True

    def _replay_minimal_snapshot(self, session_id: str) -> Optional[Dict[str, Any]]:
        events = list(self._iter_events(session_id))
        if not events:
            return None

        messages: List[Dict[str, str]] = []
        metadata: Dict[str, Any] = {}
        revision = 0

        def append_message(role: str, content: Any) -> None:
            text = str(content or "")
            if not text:
                return
            if messages and messages[-1].get("role") == role and messages[-1].get("content") == text:
                return
            messages.append({"role": role, "content": text})

        for ev in events:
            revision = max(revision, int(ev.get("revision", 0)))
            payload = ev.get("payload") or {}
            etype = ev.get("event_type")
            if etype in {"user_message", "assistant_message"}:
                role = "user" if etype == "user_message" else "assistant"
                append_message(role, payload.get("content", ""))
            elif etype == "turn_complete":
                append_message("assistant", payload.get("response", ""))
            if etype == "turn_finished":
                metadata.update(payload.get("metadata", {}))

        now = _utc_now_iso()
        return {
            "schema_version": 2,
            "session_id": session_id,
            "metadata": {
                "title": metadata.get("title", "Recovered Session"),
                "created_at": metadata.get("created_at", now),
                "updated_at": now,
                "message_count": len(messages),
                "is_resumable": True,
                "resume_warning": "Snapshot unavailable; recovered minimal state from event log.",
            },
            "messages": messages,
            "agent_state": {},
            "agent_histories": {},
            "agent_snapshots": {},
            "events_log": [],
            "history_revision": revision,
            "compaction_stats": self._empty_compaction_stats(),
        }

    def save_conversation(
        self,
        session_id: str,
        metadata: Dict[str, Any],
        messages: List[Dict[str, str]],
        agent_state: Optional[Dict[str, Any]] = None,
        agent_histories: Optional[Dict[str, List]] = None,
        events_log: Optional[List[Dict[str, Any]]] = None,
        agent_snapshots: Optional[Dict[str, Any]] = None,
        compaction_stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        conv_path = self._get_conversation_path(session_id)

        with self._session_lock(session_id):
            existing = self._read_snapshot_for_update(session_id, conv_path)
            existing_v2, migrated = self._to_v2_snapshot(session_id, existing) if existing else ({}, False)

            now = _utc_now_iso()
            prev_meta = dict(existing_v2.get("metadata") or {})
            created_at = prev_meta.get("created_at", now)
            revision = int(existing_v2.get("history_revision", 0)) + 1

            snapshots = dict(existing_v2.get("agent_snapshots") or {})
            if agent_snapshots:
                for name, snap in agent_snapshots.items():
                    merged = dict(snapshots.get(name) or {})
                    merged.update(dict(snap or {}))
                    snapshots[name] = merged
            for snap in snapshots.values():
                if isinstance(snap, dict):
                    snap.pop("chat_history", None)

            cs = dict(existing_v2.get("compaction_stats") or self._empty_compaction_stats())
            if compaction_stats:
                cs.update(compaction_stats)

            merged_metadata = {
                **prev_meta,
                **(metadata or {}),
                "title": self._derive_title(metadata or {}, messages),
                "created_at": created_at,
                "updated_at": now,
                "message_count": len(messages),
                "is_resumable": True,
            }

            current_events = list(events_log) if events_log is not None else []
            merged_metadata["web_event_count"] = len(current_events)

            snapshot = {
                "schema_version": 2,
                "session_id": session_id,
                "metadata": merged_metadata,
                "messages": list(messages or []),
                "agent_state": dict(agent_state or {}),
                "agent_histories": dict(agent_histories or {}),
                "agent_snapshots": snapshots,
                "events_log": current_events,
                "history_revision": revision,
                "compaction_stats": cs,
            }

            self._write_json_atomic(conv_path, snapshot)

            if migrated:
                self._append_event_line(
                    session_id,
                    self._event_record(session_id, revision, "resume_migrated", {"migrated_from_v1": True}),
                )

            self._append_event_line(
                session_id,
                self._event_record(
                    session_id,
                    revision,
                    "turn_finished",
                    {
                        "message_count": len(messages),
                        "metadata": {
                            "updated_at": merged_metadata.get("updated_at"),
                            "title": merged_metadata.get("title"),
                        },
                    },
                ),
            )

        return str(conv_path)

    def append_event(
        self,
        session_id: str,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        revision: Optional[int] = None,
    ) -> None:
        with self._session_lock(session_id):
            conv_path = self._get_conversation_path(session_id)
            raw = self._read_snapshot_for_update(session_id, conv_path)
            snapshot, _ = self._to_v2_snapshot(session_id, raw) if raw else ({}, False)
            rev = int(revision if revision is not None else snapshot.get("history_revision", 0))
            self._append_event_line(session_id, self._event_record(session_id, rev, event_type, payload))

    def append_timeline_event(
        self,
        session_id: str,
        event: Dict[str, Any],
        revision: Optional[int] = None,
    ) -> None:
        event_type = str(event.get("type") or "timeline_event")
        payload = self._compact_timeline_payload(event_type, event.get("data") or {})
        timestamp = event.get("timestamp")
        with self._session_lock(session_id):
            if revision is not None:
                rev = int(revision)
            else:
                conv_path = self._get_conversation_path(session_id)
                raw = self._read_snapshot_for_update(session_id, conv_path)
                snapshot, _ = self._to_v2_snapshot(session_id, raw) if raw else ({}, False)
                rev = int(snapshot.get("history_revision", 0))
            record = self._event_record(session_id, rev, event_type, payload, ts=timestamp)
            record["stream"] = "timeline"
            if event.get("event_id"):
                record["event_id"] = str(event.get("event_id") or "")
            if event.get("turn_id"):
                record["turn_id"] = str(event.get("turn_id") or "")
            if event.get("process_id"):
                record["process_id"] = str(event.get("process_id") or "")
            self._append_event_line(session_id, record)

    def get_events(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return list(self._iter_events(session_id, limit=limit))

    def get_timeline_events(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        records = self.get_events(session_id, limit=limit)
        return [
            self._timeline_event_from_record(record)
            for record in records
            if self._is_timeline_record(record)
        ]

    def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        conversations: List[Dict[str, Any]] = []
        max_items = max(1, int(limit or 50))

        conv_files = [
            path
            for path in self.conversations_dir.glob("*.json")
            if not path.name.endswith(".events.json")
        ]
        conv_files.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)

        for conv_file in conv_files:
            if conv_file.name.endswith(".events.json"):
                continue
            if self._file_size(conv_file) > self._LIST_SNAPSHOT_MAX_BYTES:
                conversations.append(self._large_snapshot_listing(conv_file))
                if len(conversations) >= max_items:
                    break
                continue
            data = self._read_json(conv_file)
            if not data:
                continue
            session_id = data.get("session_id") or conv_file.stem
            v2, _ = self._to_v2_snapshot(session_id, data)
            conversations.append(
                {
                    "session_id": session_id,
                    "metadata": v2.get("metadata", {}),
                    "preview": self._get_preview(v2.get("messages", [])),
                }
            )
            if len(conversations) >= max_items:
                break

        conversations.sort(key=lambda x: x.get("metadata", {}).get("updated_at", ""), reverse=True)
        return conversations[:max_items]

    def get_conversation(
        self,
        session_id: str,
        *,
        degrade_large_snapshot: bool = False,
    ) -> Optional[Dict[str, Any]]:
        conv_path = self._get_conversation_path(session_id)
        diagnostics = {
            "source": "snapshot",
            "migrated_from_v1": False,
            "degraded": False,
            "warnings": [],
        }
        skipped_large_snapshot = False

        if conv_path.exists():
            snapshot_size = self._file_size(conv_path)
            if degrade_large_snapshot and snapshot_size > self._RESUME_SNAPSHOT_MAX_BYTES:
                skipped_large_snapshot = True
                diagnostics["source"] = "replay"
                diagnostics["degraded"] = True
                diagnostics["warnings"].append(self._large_snapshot_warning(snapshot_size))
            else:
                raw = self._read_json(conv_path)
                if raw is not None:
                    snapshot, migrated = self._to_v2_snapshot(session_id, raw)
                    diagnostics["migrated_from_v1"] = migrated
                    timeline_events = self.get_timeline_events(
                        session_id,
                        limit=self._RESUME_TIMELINE_EVENT_LIMIT,
                    )
                    if timeline_events:
                        snapshot["events_log"] = timeline_events
                    snapshot["resume_diagnostics"] = diagnostics
                    return snapshot

                diagnostics["warnings"].append("Snapshot JSON unreadable; trying event replay.")

        replayed = self._replay_minimal_snapshot(session_id)
        if replayed is not None:
            diagnostics["source"] = "replay"
            diagnostics["degraded"] = True
            diagnostics["warnings"].append("Recovered from event replay; agent sub-state may be incomplete.")
            timeline_events = self.get_timeline_events(
                session_id,
                limit=self._RESUME_TIMELINE_EVENT_LIMIT,
            )
            if timeline_events:
                replayed["events_log"] = timeline_events
            replayed["resume_diagnostics"] = diagnostics
            return replayed

        if skipped_large_snapshot:
            placeholder = self._large_snapshot_placeholder(session_id, conv_path)
            placeholder["resume_diagnostics"] = diagnostics
            return placeholder

        logger.warning("Conversation %s not found", session_id)
        return None

    def compact_conversation(self, session_id: str, keep_last_turns: int = 6) -> Dict[str, Any]:
        from .tools.context_compaction import compact_history, estimate_tokens, get_context_policy

        with self._session_lock(session_id):
            conv_path = self._get_conversation_path(session_id)
            raw = self._read_json(conv_path)
            if not raw:
                return {"ok": False, "error": "Session not found"}

            snapshot, _ = self._to_v2_snapshot(session_id, raw)
            agent_histories = dict(snapshot.get("agent_histories") or {})
            agent_state = dict(snapshot.get("agent_state") or {})
            context_policy = get_context_policy(agent_state.get("context_policy"))
            agent_state["context_policy"] = dict(context_policy)
            snapshots = dict(snapshot.get("agent_snapshots") or {})
            before_tokens = 0
            after_tokens = 0
            removed_messages = 0
            any_compacted = False
            last_info: Dict[str, Any] = {}

            all_agent_names = sorted(set(agent_histories.keys()) | set(snapshots.keys()))
            if not all_agent_names and snapshot.get("messages"):
                synthesized_history = []
                for item in list(snapshot.get("messages") or []):
                    role = str(item.get("role") or "").strip().lower()
                    content = str(item.get("content") or "")
                    if role == "user":
                        synthesized_history.append(HumanMessage(content=content))
                    elif role == "assistant":
                        synthesized_history.append(AIMessage(content=content))
                    elif role == "system":
                        synthesized_history.append(SystemMessage(content=content))
                if synthesized_history:
                    agent_histories["planner"] = [dumpd(msg) for msg in synthesized_history]
                    all_agent_names = ["planner"]

            for name in all_agent_names:
                history_raw = list(agent_histories.get(name) or (dict(snapshots.get(name) or {}).get("chat_history") or []))
                if not history_raw:
                    continue
                try:
                    history_messages = [load(msg) for msg in history_raw]
                except Exception:
                    logger.debug("Failed to load agent history for offline compact: %s", name, exc_info=True)
                    continue
                compacted_view, info = compact_history(
                    history_messages,
                    keep_last_turns=keep_last_turns,
                    llm=None,
                    llm_enabled=False,
                    llm_input_max_chars=int(context_policy.get("llm_compact_input_max_chars", 24000)),
                    short_history_max_tool_chars=int(context_policy.get("max_tool_chars", 1500)),
                    preserve_leading_system=False,
                    state=agent_state,
                    trigger="manual",
                    force_full=True,
                    microcompact_enabled=bool(context_policy.get("microcompact_enabled", True)),
                )
                history_updates = list(info.get("history_updates") or [])
                new_history = history_updates if history_updates else history_messages
                agent_histories[name] = [dumpd(msg) for msg in new_history]
                snap = dict(snapshots.get(name) or {})
                snap.pop("chat_history", None)
                snap["langgraph_checkpoint"] = None
                snapshots[name] = snap
                before_tokens += int(info.get("before_tokens", estimate_tokens(history_messages)))
                after_tokens += int(info.get("after_tokens", estimate_tokens(compacted_view)))
                removed_messages += int(info.get("removed_messages", 0))
                last_info = dict(info)
                if history_updates or str(info.get("mode") or "none") != "none":
                    any_compacted = True

            if not any_compacted:
                return {
                    "ok": True,
                    "before_tokens": before_tokens,
                    "after_tokens": after_tokens,
                    "removed_messages": int(removed_messages),
                    "changed": False,
                }

            snapshot["agent_histories"] = agent_histories
            snapshot["agent_snapshots"] = snapshots
            snapshot["agent_state"] = agent_state
            snapshot["history_revision"] = int(snapshot.get("history_revision", 0)) + 1
            cs = dict(snapshot.get("compaction_stats") or self._empty_compaction_stats())
            cs["count"] = int(cs.get("count", 0)) + 1
            cs["last_at"] = _utc_now_iso()
            cs["last_before_tokens"] = int(before_tokens)
            cs["last_after_tokens"] = int(after_tokens)
            cs["last_reason"] = "manual_api"
            snapshot["compaction_stats"] = cs
            agent_state["llm_context_usage"] = {
                "budget_tokens": int(after_tokens),
                "total_tokens": int(after_tokens),
                "prompt_tokens": int(after_tokens),
                "source": "estimate_after_offline_compact",
            }
            snapshot["agent_state"] = agent_state

            self._write_json_atomic(conv_path, snapshot)
            self._append_event_line(
                session_id,
                self._event_record(
                    session_id,
                    int(snapshot.get("history_revision", 0)),
                    "context_compacted",
                    {
                        "before_tokens": before_tokens,
                        "after_tokens": after_tokens,
                        "removed_messages": int(removed_messages),
                        "reason": "manual_api",
                        "mode": last_info.get("mode", "rule"),
                        "summary_text": last_info.get("summary_text", ""),
                        "summary_core": last_info.get("summary_core", ""),
                    },
                ),
            )

        return {
            "ok": True,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "removed_messages": int(removed_messages),
            "changed": True,
        }

    def delete_conversation(self, session_id: str) -> bool:
        deleted = False
        for path in [
            self._get_conversation_path(session_id),
            self._get_events_path(session_id),
            self._get_lock_path(session_id),
        ]:
            if path.exists():
                path.unlink(missing_ok=True)
                deleted = True
        return deleted
