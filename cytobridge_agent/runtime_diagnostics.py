"""Runtime diagnostics and lightweight performance reporting."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .conversation_store import ConversationStore
from .job_registry import JobRegistry


def run_doctor(*, store: Optional[ConversationStore] = None, jobs: Optional[JobRegistry] = None) -> Dict[str, Any]:
    store = store or ConversationStore()
    jobs = jobs or JobRegistry()
    conversations_dir = store.conversations_dir
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("conversation_store", conversations_dir.exists(), str(conversations_dir))
    add("conversation_store_writable", os.access(conversations_dir, os.W_OK), str(conversations_dir))
    add("job_registry", jobs.path.parent.exists(), str(jobs.path))
    add("job_registry_writable", os.access(jobs.path.parent, os.W_OK), str(jobs.path.parent))
    add("python_env", True, os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV") or "system")

    try:
        import fastapi  # noqa: F401

        add("fastapi", True, "available")
    except Exception as exc:
        add("fastapi", False, str(exc))

    try:
        import rich  # noqa: F401

        add("rich", True, "available")
    except Exception as exc:
        add("rich", False, str(exc))

    stale_jobs = jobs.stale_jobs()
    add("stale_jobs", not stale_jobs, f"{len(stale_jobs)} stale job(s)")

    try:
        sessions = store.list_conversations(limit=5)
        add("conversation_listing", True, f"{len(sessions)} recent session(s)")
    except Exception as exc:
        add("conversation_listing", False, str(exc))

    return {
        "status": "ok" if all(item["ok"] for item in checks) else "warning",
        "checks": checks,
        "active_jobs": jobs.list(active_only=True),
        "stale_jobs": stale_jobs,
    }


def session_perf_report(session_id: str, *, store: Optional[ConversationStore] = None) -> Dict[str, Any]:
    store = store or ConversationStore()
    events = store.get_timeline_events(session_id)
    counts: Dict[str, int] = {}
    durations: Dict[str, List[float]] = {}
    errors: List[Dict[str, Any]] = []

    for event in events:
        event_type = str(event.get("type") or "event")
        counts[event_type] = counts.get(event_type, 0) + 1
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        for key in ("elapsed_seconds", "duration_seconds", "latency_seconds"):
            if key in data:
                try:
                    durations.setdefault(event_type, []).append(float(data[key]))
                except Exception:
                    pass
        if "error" in event_type or data.get("error") or data.get("exception"):
            errors.append(event)

    duration_summary = {
        key: {
            "count": len(values),
            "total_seconds": round(sum(values), 3),
            "max_seconds": round(max(values), 3) if values else 0.0,
        }
        for key, values in durations.items()
    }

    return {
        "session_id": session_id,
        "event_count": len(events),
        "event_type_counts": counts,
        "duration_summary": duration_summary,
        "error_count": len(errors),
        "recent_errors": errors[-5:],
    }


def format_doctor_report(report: Dict[str, Any]) -> str:
    lines = [f"Runtime doctor: {report.get('status', 'unknown')}"]
    for item in report.get("checks", []):
        mark = "OK" if item.get("ok") else "WARN"
        detail = item.get("detail") or ""
        lines.append(f"- {mark} {item.get('name')}: {detail}")
    active_jobs = report.get("active_jobs") or []
    if active_jobs:
        lines.append("")
        lines.append("Active jobs:")
        for job in active_jobs[:10]:
            lines.append(f"- {job.get('job_id')} {job.get('kind')} {job.get('status')} pid={job.get('pid')}")
    return "\n".join(lines)


def format_perf_report(report: Dict[str, Any]) -> str:
    lines = [
        f"Session: {report.get('session_id')}",
        f"Events: {report.get('event_count', 0)}",
        f"Errors: {report.get('error_count', 0)}",
    ]
    counts = report.get("event_type_counts") or {}
    if counts:
        lines.extend(["", "Event counts:"])
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:20]:
            lines.append(f"- {key}: {value}")
    durations = report.get("duration_summary") or {}
    if durations:
        lines.extend(["", "Durations:"])
        for key, value in sorted(durations.items()):
            lines.append(
                f"- {key}: total={value.get('total_seconds')}s max={value.get('max_seconds')}s count={value.get('count')}"
            )
    return "\n".join(lines)

