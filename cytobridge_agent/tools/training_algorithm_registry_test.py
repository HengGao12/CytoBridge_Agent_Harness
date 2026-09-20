from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from cytobridge_agent.tools.training_algorithm_registry import render_training_algorithm_catalog_context


def _write_algorithm(root: Path, algorithm_id: str, description: str) -> None:
    algo_dir = root / ".cellcompass" / "training_algorithms" / algorithm_id
    algo_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "algorithm_id": algorithm_id,
        "api_version": 1,
        "entrypoint": "algorithm.py",
        "entry_function": "build_training_algorithm",
        "description": description,
        "requirements": f"requirements for {algorithm_id}",
        "base_config": "default",
    }
    with (algo_dir / "manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f)
    (algo_dir / "algorithm.py").write_text(
        "def build_training_algorithm(context):\n"
        "    raise RuntimeError('not used by registry rendering tests')\n",
        encoding="utf-8",
    )


def _write_campaign(
    root: Path,
    algorithm_id: str,
    campaign_id: str,
    *,
    status: str,
    stage: str,
    locked_at: str = "",
) -> None:
    campaign_dir = root / ".cellcompass" / "algorithm_campaigns" / algorithm_id / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm_id": algorithm_id,
        "campaign_id": campaign_id,
        "status": status,
        "current_stage": stage,
        "updated_at": locked_at or "2026-01-01T00:00:00+00:00",
    }
    if locked_at:
        payload["locked_release"] = {
            "trial_id": f"trial_{algorithm_id}",
            "locked_at": locked_at,
        }
    (campaign_dir / "campaign.json").write_text(json.dumps(payload), encoding="utf-8")


class TrainingAlgorithmCatalogPromptTests(unittest.TestCase):
    def test_catalog_includes_only_latest_final_locked_algorithms(self) -> None:
        old_home = os.environ.get("HOME")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_algorithm(root, "proposal_only_algo", "Proposal-only algorithm")
            _write_algorithm(root, "failed_algo", "Failed algorithm")
            _write_algorithm(root, "old_locked_algo", "Old locked algorithm")
            _write_algorithm(root, "new_locked_algo", "New locked algorithm")

            _write_campaign(
                root,
                "failed_algo",
                "campaign_failed",
                status="failed",
                stage="stage1_feasibility",
                locked_at="",
            )
            _write_campaign(
                root,
                "old_locked_algo",
                "campaign_old",
                status="locked",
                stage="final_regression",
                locked_at="2026-01-02T00:00:00+00:00",
            )
            _write_campaign(
                root,
                "new_locked_algo",
                "campaign_new",
                status="locked",
                stage="final_regression",
                locked_at="2026-01-03T00:00:00+00:00",
            )

            os.environ["HOME"] = str(root)
            try:
                rendered = render_training_algorithm_catalog_context(max_items=1)
            finally:
                if old_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = old_home

        self.assertIn("latest final-regression locked releases only", rendered)
        self.assertIn("new_locked_algo", rendered)
        self.assertIn("campaign_new", rendered)
        self.assertNotIn("old_locked_algo", rendered)
        self.assertNotIn("proposal_only_algo", rendered)
        self.assertNotIn("failed_algo", rendered)
        self.assertIn("and 1 more final-regression locked", rendered)


if __name__ == "__main__":
    unittest.main()
