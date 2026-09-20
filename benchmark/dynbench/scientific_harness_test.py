from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from benchmark.dynbench.scientific_harness import (
    apply_public_scientific_calibration,
    audit_public_outputs,
    build_public_data_profile,
    build_revision_prompt,
)


class ScientificHarnessTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        genes = ["Gene_1", "Gene_2", "Gene_3"]
        rng = np.random.default_rng(7)
        n_cells = 90
        x = rng.normal(size=(n_cells, len(genes))).astype(np.float32)
        times = np.repeat([0.0, 1.0, 2.0], n_cells // 3)
        labels = np.tile(["A", "B", "C"], n_cells // 3)
        obs = pd.DataFrame(
            {
                "time": times,
                "time_bin": times.astype(int),
                "fate_true": labels,
            },
            index=[f"cell_{idx}" for idx in range(n_cells)],
        )
        train_path = root / "train.h5ad"
        ad.AnnData(x, obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(train_path)

        output = root / "output"
        output.mkdir()
        pd.DataFrame(np.ones((n_cells, 3)), columns=[f"velocity_{g}" for g in genes]).to_csv(
            output / "velocity_field.csv", index=False
        )
        growth = np.square(x[:, 1]) + 0.02 * rng.normal(size=n_cells)
        pd.DataFrame({"growth_rate": growth}).to_csv(
            output / "growth_rates.csv", index=False
        )
        holdout = pd.DataFrame(np.vstack([x[:30], x[30:60]]), columns=genes)
        holdout["time"] = [0.5] * 30 + [1.5] * 30
        holdout.to_csv(output / "holdout_prediction.csv", index=False)
        (output / "per_cell_fate.json").write_text(
            json.dumps({"cell_0": {"B": 0.8, "C": 0.2}, "cell_1": {"B": 0.2, "C": 0.8}}),
            encoding="utf-8",
        )
        perturbations = []
        for idx, gene in enumerate(genes):
            delta = 0.2 - 0.02 * idx
            perturbations.append(
                {
                    "gene_name": gene,
                    "control": {"B": 0.5, "C": 0.5},
                    "perturbed": {"B": 0.5 + delta, "C": 0.5 - delta},
                    "delta": {"B": delta, "C": -delta},
                }
            )
        (output / "perturbation_results.json").write_text(
            json.dumps(perturbations), encoding="utf-8"
        )
        edges = [
            {"source": source, "target": target, "score": 0.5}
            for source in genes
            for target in genes
            if source != target
        ]
        (output / "driver_genes.json").write_text(
            json.dumps(
                {
                    "grn_edges": edges,
                    "growth_drivers": {"Gene_1": 0.9, "Gene_2": 0.2, "Gene_3": 0.1},
                }
            ),
            encoding="utf-8",
        )
        return train_path, output

    def test_profile_contains_only_public_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_path, _ = self._fixture(Path(tmp))
            profile = build_public_data_profile(train_path)
            self.assertTrue(profile["public_inputs_only"])
            self.assertEqual(profile["n_cells"], 90)
            self.assertEqual(profile["genes"], ["Gene_1", "Gene_2", "Gene_3"])
            self.assertNotIn("ground_truth", profile)

    def test_audit_detects_unselective_perturbations_and_driver_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_path, output = self._fixture(Path(tmp))
            audit = audit_public_outputs(train_path, output)
            codes = {item["code"] for item in audit["issues"]}
            self.assertIn("perturbation_unselective", codes)
            self.assertIn("growth_driver_evidence_disagreement", codes)
            self.assertTrue(audit["public_inputs_only"])
            prompt = build_revision_prompt(audit, output / "audit.json")
            self.assertIn("public-data-only", prompt)
            self.assertIn("Do not fabricate values", prompt)

    def test_calibration_is_auditable_and_uses_model_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_path, output = self._fixture(Path(tmp))
            manifest = apply_public_scientific_calibration(
                train_path, output, n_growth_resamples=4
            )
            self.assertTrue(manifest["applied"])
            self.assertTrue((output / "scientific_harness_raw" / "driver_genes.json").exists())
            self.assertTrue((output / "scientific_harness_raw" / "perturbation_results.json").exists())
            actions = {item["artifact"] for item in manifest["actions"]}
            self.assertEqual(
                actions,
                {"driver_genes.json", "perturbation_results.json"},
            )
            drivers = json.loads((output / "driver_genes.json").read_text(encoding="utf-8"))
            self.assertEqual(
                max(drivers["growth_drivers"], key=drivers["growth_drivers"].get),
                "Gene_2",
            )
            perturbations = json.loads(
                (output / "perturbation_results.json").read_text(encoding="utf-8")
            )
            self.assertTrue(all("scientific_harness_raw_delta" in row for row in perturbations))


if __name__ == "__main__":
    unittest.main()
