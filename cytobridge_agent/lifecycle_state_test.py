from __future__ import annotations

from cytobridge_agent.lifecycle_state import summarize_algorithm_lifecycle


def test_lifecycle_requires_locked_release_for_completion() -> None:
    summary = summarize_algorithm_lifecycle(
        {
            "active_algorithm_context": {
                "algorithm_id": "algo1",
                "algorithm_lifecycle_status": "complete",
            },
            "active_algorithm_campaign_id": "campaign1",
            "active_algorithm_campaign": {
                "campaign_id": "campaign1",
                "algorithm_id": "algo1",
                "status": "active",
                "current_stage": "final_regression",
            },
        }
    )

    assert summary["status"] == "inconsistent"
    assert summary["lifecycle_complete"] is False
    assert "no locked release" in " ".join(summary["blockers"])


def test_lifecycle_marks_final_regression_locked_release_complete() -> None:
    summary = summarize_algorithm_lifecycle(
        {
            "active_algorithm_context": {
                "algorithm_id": "algo1",
                "algorithm_lifecycle_status": "complete",
            },
            "active_algorithm_campaign_id": "campaign1",
            "active_algorithm_campaign": {
                "campaign_id": "campaign1",
                "algorithm_id": "algo1",
                "status": "locked",
                "algorithm_lifecycle_status": "complete",
                "current_stage": "final_regression",
                "locked_release": {"trial_id": "trial_final", "run_ids": ["run_final"]},
                "stages": {
                    "final_regression": {
                        "status": "locked",
                        "active_best_trial_id": "trial_final",
                        "last_gate_check": {"ok": True},
                    }
                },
            },
        }
    )

    assert summary["status"] == "complete"
    assert summary["lifecycle_complete"] is True
    assert summary["release_locked"] is True
    assert summary["locked_release"]["trial_id"] == "trial_final"


def test_lifecycle_surfaces_stage_gate_blockers() -> None:
    summary = summarize_algorithm_lifecycle(
        {
            "active_algorithm_context": {
                "algorithm_id": "algo1",
                "algorithm_lifecycle_status": "developing",
            },
            "active_algorithm_campaign_id": "campaign1",
            "active_algorithm_campaign": {
                "campaign_id": "campaign1",
                "status": "active",
                "current_stage": "stage2_claim_validation",
                "stages": {
                    "stage1_feasibility": {"status": "passed", "gate_ready": True},
                    "stage2_claim_validation": {
                        "status": "active",
                        "active_best_trial_id": "trial2",
                        "last_gate_check": {
                            "ok": False,
                            "blockers": ["baseline claim metric missing"],
                        },
                    },
                },
                "stage_statuses": {
                    "stage2_claim_validation": {
                        "next_required_action": "compute baseline claim metric",
                    }
                },
            },
        }
    )

    assert summary["status"] == "gate_blocked"
    assert summary["current_stage"] == "stage2_claim_validation"
    assert summary["stages"]["stage1_feasibility"]["passed"] is True
    assert summary["stages"]["stage2_claim_validation"]["passed"] is False
    assert summary["blockers"] == ["baseline claim metric missing"]
    assert summary["next_actions"] == ["compute baseline claim metric"]


def test_lifecycle_marks_failed_as_needing_revision() -> None:
    summary = summarize_algorithm_lifecycle(
        {
            "active_algorithm_context": {
                "algorithm_id": "algo1",
                "algorithm_lifecycle_status": "failed",
                "algorithm_lifecycle_status_reason": "claim metric could not be validated",
            },
            "active_algorithm_campaign": {"status": "failed"},
        }
    )

    assert summary["status"] == "failed"
    assert "claim metric" in summary["blockers"][0]
    assert summary["next_actions"] == ["revise_proposal_or_restore_meaningful_algorithm"]
