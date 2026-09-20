from __future__ import annotations

import os
from pathlib import Path

import pytest

from cytobridge_agent.session_ownership import SessionOwnershipError, SessionOwnershipRegistry


def test_session_ownership_blocks_live_foreign_process(tmp_path: Path) -> None:
    registry = SessionOwnershipRegistry(tmp_path / "sessions.json")
    registry.acquire(session_id="s1", owner_id="owner1", owner_kind="cli")
    records = registry._read()
    records["s1"]["pid"] = 1
    records["s1"]["owner_id"] = "foreign"
    registry._write(records)

    with pytest.raises(SessionOwnershipError):
        registry.acquire(session_id="s1", owner_id="owner2", owner_kind="web")


def test_session_ownership_allows_same_process_takeover_and_release(tmp_path: Path) -> None:
    registry = SessionOwnershipRegistry(tmp_path / "sessions.json")
    first = registry.acquire(session_id="s1", owner_id="owner1", owner_kind="cli")
    second = registry.acquire(session_id="s1", owner_id="owner2", owner_kind="web")

    assert first["pid"] == os.getpid()
    assert second["owner_id"] == "owner2"
    assert second["replaced_owner_id"] == "owner1"
    released = registry.release("s1", "owner2")
    assert released is not None
    assert released["status"] == "released"
