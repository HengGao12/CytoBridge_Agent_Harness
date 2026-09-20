from __future__ import annotations

from pathlib import Path

from cytobridge_agent.conversation_store import ConversationStore
from cytobridge_agent.job_registry import JobRegistry
from cytobridge_agent.session_controller import SessionController
from cytobridge_agent.session_ownership import SessionOwnershipRegistry


class _FakeSession:
    session_id = "snap123"
    input_path = "/data/raw.h5ad"
    output_dir = "/tmp/out"

    def __init__(self) -> None:
        self.state = {
            "conversation_turn": 4,
            "planner_phase": "campaign",
            "workflow_phase": "stage2",
            "preprocessed_path": "/data/preprocessed.h5ad",
            "final_config": {"path": "/data/final.h5ad"},
            "active_algorithm_context": {"algorithm_id": "algo_snap", "workspace_path": "/tmp/algo_snap"},
            "latest_algorithm_proposal_id": "proposal_snap",
            "latest_training_algorithm_id": "algo_snap",
            "active_algorithm_campaign_id": "campaign_snap",
            "active_algorithm_campaign": {
                "campaign_id": "campaign_snap",
                "current_stage": "stage2_controlled",
                "status": "running",
                "baseline_policy": {"mode": "strict_all_builtin"},
            },
            "stop_hook_enabled": True,
            "stop_hook_mode": "algorithm_lifecycle",
            "stop_hook_max_triggers": 50,
            "report_path": "/tmp/out/report.html",
            "paper_path": "/tmp/out/paper/main.tex",
        }


def test_controller_status_exposes_canonical_snapshot(tmp_path: Path) -> None:
    controller = SessionController(
        lambda _cfg: None,  # type: ignore[arg-type,return-value]
        initial_config={
            "llm_provider": "xiaomi",
            "llm_model": "mimo-v2.5-pro",
            "llm_thinking_level": "high",
        },
    )
    controller.jobs = JobRegistry(tmp_path / "jobs.json")
    controller.ownership = SessionOwnershipRegistry(tmp_path / "sessions.json")
    controller.jobs.upsert(job_id="job1", kind="training", session_id="snap123", status="running")
    controller.active_session = _FakeSession()  # type: ignore[assignment]
    controller.ownership.acquire(session_id="snap123", owner_id=controller.owner_id, owner_kind="test")

    status = controller.status()
    snapshot = status["snapshot"]

    assert status["session_id"] == "snap123"
    assert status["provider"] == "xiaomi"
    assert status["active_jobs"][0]["job_id"] == "job1"
    assert snapshot["schema_version"] == 1
    assert snapshot["ownership"]["owner_id"] == controller.owner_id
    assert snapshot["session"]["session_id"] == "snap123"
    assert snapshot["model"]["thinking_level"] == "high"
    assert snapshot["workflow"]["planner_phase"] == "campaign"
    assert snapshot["data"]["active_data_path"] == "/data/final.h5ad"
    assert snapshot["algorithm"]["active_algorithm_id"] == "algo_snap"
    assert snapshot["algorithm"]["active_campaign_id"] == "campaign_snap"
    assert snapshot["algorithm"]["active_campaign_stage"] == "stage2_controlled"
    assert snapshot["algorithm"]["baseline_mode"] == "strict_all_builtin"
    assert snapshot["algorithm"]["lifecycle"]["status"] == "campaign_running"
    assert snapshot["algorithm"]["lifecycle"]["current_stage"] == "stage2_controlled"
    assert snapshot["algorithm"]["campaign_resolution"]["source"] == "session_state"
    assert snapshot["jobs"]["active_count"] == 1
    assert snapshot["artifacts"]["paper_path"] == "/tmp/out/paper/main.tex"
    assert snapshot["state_sources"]["workflow"] == "active_session.state"


def test_saved_session_status_includes_canonical_snapshot(tmp_path: Path) -> None:
    store = ConversationStore(base_dir=str(tmp_path / "cellcompass"))
    controller = SessionController(lambda _cfg: None, store=store)  # type: ignore[arg-type,return-value]
    controller.jobs = JobRegistry(tmp_path / "jobs.json")
    controller.ownership = SessionOwnershipRegistry(tmp_path / "sessions.json")
    controller.jobs.upsert(job_id="job_saved", kind="training", session_id="saved123", status="running")
    store.save_conversation(
        "saved123",
        metadata={"input_path": "/data/input.h5ad", "output_dir": "/tmp/out", "title": "Saved session"},
        messages=[],
        agent_state={
            "conversation_turn": 7,
            "planner_phase": "final_regression",
            "latest_training_algorithm_id": "saved_algo",
            "active_algorithm_campaign_id": "saved_campaign",
            "active_algorithm_campaign": {"campaign_id": "saved_campaign", "current_stage": "stage3_real"},
            "final_config": {"path": "/data/final_saved.h5ad"},
        },
    )

    status = controller.session_status("saved123")
    snapshot = status["snapshot"]

    assert status["status"] == "success"
    assert snapshot["runtime"]["status"] == "saved"
    assert "ownership" in snapshot
    assert snapshot["session"]["session_id"] == "saved123"
    assert snapshot["data"]["active_data_path"] == "/data/final_saved.h5ad"
    assert snapshot["algorithm"]["active_algorithm_id"] == "saved_algo"
    assert snapshot["algorithm"]["active_campaign_stage"] == "stage3_real"
    assert snapshot["algorithm"]["lifecycle"]["current_stage"] == "stage3_real"
    assert snapshot["jobs"]["active_count"] == 1
    assert snapshot["state_sources"]["workflow"] == "conversation_store.agent_state"
