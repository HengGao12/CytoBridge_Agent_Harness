from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cytobridge_agent.runtime_v2.commit import WorkflowCommitter
from cytobridge_agent.runtime_v2.paper_review_freshness import build_paper_review_manifest, paper_review_manifest_status
from cytobridge_agent.runtime_v2.state import ensure_runtime_v2_state


class PaperReviewGateSyncTest(unittest.TestCase):
    def test_commit_workflow_state_syncs_approved_paper_reviewer_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paper_dir = output_dir / "outputs" / "paper"
            checks_dir = paper_dir / "checks"
            checks_dir.mkdir(parents=True)
            (paper_dir / "main.tex").write_text("\\input{sections/method}\n", encoding="utf-8")
            (paper_dir / "sections").mkdir()
            (paper_dir / "sections" / "method.tex").write_text("\\section{Method}\n", encoding="utf-8")
            gate_path = checks_dir / "paper_reviewer_gate.md"
            gate_path.write_text("Latest reviewer decision: revise.\nStatus: pending.\n", encoding="utf-8")

            manifest = build_paper_review_manifest([str(paper_dir)])
            self.assertTrue(paper_review_manifest_status(manifest)["fresh"])
            self.assertNotIn(
                "paper_reviewer_gate.md",
                {Path(str(item["path"])).name for item in manifest.get("files", [])},
            )

            state = ensure_runtime_v2_state({"session_id": "paper-gate-sync", "output_dir": str(output_dir)})
            committer = WorkflowCommitter(state)
            committer.commit(
                phase="reporting",
                updates={
                    "paper_review": {
                        "decision": "approve",
                        "reviewer_feedback": "Approved with no blocking issues.",
                        "blocking_issues": [],
                        "reviewed_file_hashes": manifest,
                    }
                },
                summary="record paper reviewer approval",
            )

            self.assertEqual(state["paper_reviewer_gate_sync"]["status"], "synced")
            gate_text = gate_path.read_text(encoding="utf-8")
            self.assertIn("Latest reviewer decision: approve.", gate_text)
            self.assertIn("Status: approved.", gate_text)
            self.assertTrue(paper_review_manifest_status(manifest)["fresh"])

    def test_stale_manifest_does_not_overwrite_existing_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paper_dir = output_dir / "paper"
            checks_dir = paper_dir / "checks"
            checks_dir.mkdir(parents=True)
            (paper_dir / "main.tex").write_text("before\n", encoding="utf-8")
            gate_path = checks_dir / "paper_reviewer_gate.md"
            original_gate = "Latest reviewer decision: revise.\nStatus: pending.\n"
            gate_path.write_text(original_gate, encoding="utf-8")

            manifest = build_paper_review_manifest([str(paper_dir)])
            (paper_dir / "main.tex").write_text("after\n", encoding="utf-8")
            state = ensure_runtime_v2_state({"session_id": "paper-gate-stale", "output_dir": str(output_dir)})
            committer = WorkflowCommitter(state)
            committer.commit(
                phase="reporting",
                updates={
                    "paper_review": {
                        "decision": "approve",
                        "blocking_issues": [],
                        "reviewed_file_hashes": manifest,
                    }
                },
                summary="record stale paper reviewer approval",
            )

            self.assertEqual(state["paper_reviewer_gate_sync"]["status"], "skipped")
            self.assertIn("hashes changed", state["paper_reviewer_gate_sync"]["reason"])
            self.assertEqual(gate_path.read_text(encoding="utf-8"), original_gate)


if __name__ == "__main__":
    unittest.main()
