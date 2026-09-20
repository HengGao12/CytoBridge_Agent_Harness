from __future__ import annotations

from cytobridge_agent.campaign_state import resolve_active_campaign, state_with_resolved_campaign


class _RegistryTools:
    def get_algorithm_campaign_status(self, campaign_id: str = ""):
        return {
            "campaign_id": campaign_id,
            "algorithm_id": "algo",
            "status": "locked",
            "current_stage": "final_regression",
            "locked_release": {"trial_id": "trial_final"},
        }


class _FailingRegistryTools:
    def get_algorithm_campaign_status(self, campaign_id: str = ""):
        raise FileNotFoundError(campaign_id)


def test_resolve_active_campaign_prefers_registry_over_session_state() -> None:
    state = {
        "active_algorithm_campaign_id": "campaign1",
        "active_algorithm_campaign": {
            "campaign_id": "campaign1",
            "status": "active",
            "current_stage": "stage2_claim_validation",
        },
    }

    resolved = resolve_active_campaign(state, file_tools=_RegistryTools())
    merged = state_with_resolved_campaign(state, resolved)

    assert resolved["source"] == "registry"
    assert resolved["used_state_fallback"] is False
    assert resolved["campaign"]["status"] == "locked"
    assert merged["active_algorithm_campaign"]["current_stage"] == "final_regression"


def test_resolve_active_campaign_falls_back_to_state_with_error() -> None:
    state = {
        "active_algorithm_campaign_id": "campaign1",
        "active_algorithm_campaign": {
            "campaign_id": "campaign1",
            "status": "active",
        },
    }

    resolved = resolve_active_campaign(state, file_tools=_FailingRegistryTools())

    assert resolved["source"] == "session_state"
    assert resolved["used_state_fallback"] is True
    assert "FileNotFoundError" in resolved["registry_error"]
