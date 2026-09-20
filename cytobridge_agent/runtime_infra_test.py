from __future__ import annotations

from pathlib import Path

from cytobridge_agent.conversation_store import ConversationStore
from cytobridge_agent.job_registry import JobRegistry
from cytobridge_agent.runtime_diagnostics import run_doctor, session_perf_report
from cytobridge_agent.runtime_events import coerce_runtime_event, runtime_event
from cytobridge_agent.runtime_state import RuntimeState, RuntimeStatus


def test_runtime_event_envelope_is_json_safe() -> None:
    event = runtime_event(
        "tool_start",
        {"path": Path("/tmp/example"), "items": {1, 2}},
        session_id="abc123",
        turn_id="turn1",
        source="test",
    )

    assert event["type"] == "tool_start"
    assert event["session_id"] == "abc123"
    assert event["data"]["path"] == "/tmp/example"
    assert sorted(event["data"]["items"]) == [1, 2]

    coerced = coerce_runtime_event({"event_type": "legacy", "payload": {"ignored": True}}, source="web")
    assert coerced["type"] == "legacy"
    assert coerced["source"] == "web"


def test_runtime_state_turn_lifecycle() -> None:
    state = RuntimeState()

    state.begin_turn(session_id="s1", turn_id="t1")
    assert state.to_dict()["status"] == RuntimeStatus.RUNNING
    assert state.to_dict()["active_turn_id"] == "t1"

    elapsed = state.finish_turn(status=RuntimeStatus.IDLE, message="done")
    assert elapsed >= 0
    snapshot = state.to_dict()
    assert snapshot["status"] == RuntimeStatus.IDLE
    assert snapshot["active_turn_id"] is None


def test_job_registry_marks_stale_pid(tmp_path: Path) -> None:
    registry = JobRegistry(tmp_path / "jobs.json")
    registry.upsert(job_id="j1", kind="training", session_id="s1", pid=99999999)

    jobs = registry.list(session_id="s1")
    assert jobs[0]["job_id"] == "j1"
    assert jobs[0]["pid_alive"] is False
    assert registry.stale_jobs()[0]["job_id"] == "j1"

    registry.mark("j1", "completed")
    assert registry.stale_jobs() == []


def test_perf_report_aggregates_timeline_events(tmp_path: Path) -> None:
    store = ConversationStore(base_dir=str(tmp_path / "cellcompass"))
    store.append_timeline_event(
        "s1",
        runtime_event("turn_complete", {"elapsed_seconds": 1.25}, session_id="s1"),
    )
    store.append_timeline_event(
        "s1",
        runtime_event("tool_error", {"error": "boom"}, session_id="s1"),
    )

    report = session_perf_report("s1", store=store)

    assert report["event_count"] == 2
    assert report["event_type_counts"]["turn_complete"] == 1
    assert report["duration_summary"]["turn_complete"]["total_seconds"] == 1.25
    assert report["error_count"] == 1


def test_doctor_reports_warning_for_stale_jobs(tmp_path: Path) -> None:
    store = ConversationStore(base_dir=str(tmp_path / "cellcompass"))
    registry = JobRegistry(tmp_path / "jobs.json")
    registry.upsert(job_id="j1", kind="training", pid=99999999)

    report = run_doctor(store=store, jobs=registry)

    assert report["status"] == "warning"
    assert report["stale_jobs"][0]["job_id"] == "j1"

