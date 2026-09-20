from __future__ import annotations

import json
import os
import time
from pathlib import Path


AUTH_LOCK_PATH = Path.home() / ".cellcompass" / "auth_profiles.json.lock"
AUTH_LOCK_STALE_SEC = 90


def clear_stale_auth_lock(
    *,
    max_age_sec: int = AUTH_LOCK_STALE_SEC,
    ignore_live_owner: bool = False,
) -> bool:
    path = AUTH_LOCK_PATH
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    except Exception:
        payload = {}

    pid = int(payload.get("pid") or 0)
    created_at_ms = int(payload.get("created_at_ms") or 0)
    if created_at_ms > 0:
        age_sec = max(0.0, (time.time() * 1000.0 - created_at_ms) / 1000.0)
    else:
        try:
            age_sec = max(0.0, time.time() - path.stat().st_mtime)
        except FileNotFoundError:
            return False

    owner_alive = False
    if pid > 0:
        try:
            os.kill(pid, 0)
            owner_alive = True
        except OSError:
            owner_alive = False

    if owner_alive and not ignore_live_owner:
        return False
    if age_sec < float(max_age_sec):
        return False
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def clear_benchmark_auth_lock() -> bool:
    """Remove the Codex OAuth profile lock before an isolated benchmark run.

    Batch benchmark runs are deliberately serial. If a previous agent shell or
    sidecar leaves the file lock behind, the next run would otherwise block for
    minutes and fail without producing outputs.
    """

    return clear_stale_auth_lock(max_age_sec=0, ignore_live_owner=True)


def codex_error_is_retryable(message: str) -> bool:
    text = str(message or "").lower()
    return any(
        marker in text
        for marker in [
            "timed out waiting for auth store lock",
            "selected model is at capacity",
            "model is at capacity",
            "you've hit your usage limit",
            "usage limit",
            "rate limit",
            "429",
            "stream disconnected",
            "server_error",
            "service unavailable",
            "gateway timeout",
            "error sending request for url",
            "websocket",
        ]
    )
