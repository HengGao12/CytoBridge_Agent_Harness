from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np

from cytobridge_agent.tools.training_isolation import (
    _default_timeout_sec,
    _estimate_preview_timeout_from_counts,
    _estimate_preview_timeout_sec,
    _estimate_training_budget_sec,
    run_training_in_subprocess,
    should_isolate_training_target,
)
from cytobridge_agent.tools.training_run_manager import create_training_run_bundle


class TrainingIsolationTests(unittest.TestCase):
    def test_custom_training_isolated_by_default(self) -> None:
        old = os.environ.pop("CYTOBRIDGE_ISOLATE_CUSTOM_TRAINING", None)
        old_all = os.environ.pop("CYTOBRIDGE_ISOLATE_ALL_TRAINING", None)
        try:
            self.assertTrue(should_isolate_training_target("custom"))
            self.assertTrue(should_isolate_training_target("builtin"))
        finally:
            if old is not None:
                os.environ["CYTOBRIDGE_ISOLATE_CUSTOM_TRAINING"] = old
            if old_all is not None:
                os.environ["CYTOBRIDGE_ISOLATE_ALL_TRAINING"] = old_all

    def test_isolated_worker_returns_structured_error(self) -> None:
        old_timeout = os.environ.get("CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC")
        os.environ["CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC"] = "60"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                bundle = create_training_run_bundle(Path(tmp), stage="pilot", name="missing-data")
                metrics, error, isolation = run_training_in_subprocess(
                    adata_path=str(Path(tmp) / "missing.h5ad"),
                    stage="pilot",
                    device="cpu",
                    bundle=bundle,
                    training_algorithm_id="missing_algo",
                    output_dir=tmp,
                    workspace_root=str(Path.cwd()),
                    purpose="preview",
                )
        finally:
            if old_timeout is None:
                os.environ.pop("CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC", None)
            else:
                os.environ["CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC"] = old_timeout
        self.assertTrue(error)
        self.assertIn("error", metrics)
        self.assertTrue(isolation.get("isolated"))
        self.assertFalse(bundle.trained_model_path.exists())

    def test_preview_inspection_worker_returns_structured_error(self) -> None:
        old_timeout = os.environ.get("CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC")
        os.environ["CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC"] = "60"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                bundle = create_training_run_bundle(Path(tmp), stage="pilot", name="missing-preview-inspect")
                metrics, error, isolation = run_training_in_subprocess(
                    adata_path=str(Path(tmp) / "missing.h5ad"),
                    stage="pilot",
                    device="cpu",
                    bundle=bundle,
                    training_algorithm_id="missing_algo",
                    output_dir=tmp,
                    workspace_root=str(Path.cwd()),
                    purpose="preview_inspect",
                )
        finally:
            if old_timeout is None:
                os.environ.pop("CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC", None)
            else:
                os.environ["CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC"] = old_timeout
        self.assertTrue(error)
        self.assertIn("error", metrics)
        self.assertTrue(isolation.get("isolated"))
        self.assertEqual(isolation.get("timeout_sec"), 60)
        self.assertFalse(bundle.trained_model_path.exists())

    def test_preview_inspection_default_timeout_is_short(self) -> None:
        old_timeout = os.environ.pop("CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC", None)
        try:
            self.assertEqual(_default_timeout_sec(purpose="preview_inspect", adata_path="/missing.h5ad"), 300)
            self.assertEqual(_default_timeout_sec(purpose="preview", adata_path="/missing.h5ad"), 300)
        finally:
            if old_timeout is not None:
                os.environ["CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC"] = old_timeout

    def test_preview_subprocess_timeout_tracks_large_dataset_scale(self) -> None:
        old_timeout = os.environ.pop("CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC", None)
        old_max = os.environ.pop("CYTOBRIDGE_PREVIEW_SUBPROCESS_MAX_TIMEOUT_SEC", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                counts = [9000, 9000, 9000]
                adata = ad.AnnData(np.zeros((sum(counts), 1), dtype=np.float32))
                adata.obs["time_point_processed"] = np.repeat([0.0, 1.0, 2.0], counts)
                path = Path(tmp) / "large_preview.h5ad"
                adata.write_h5ad(path)

                estimated = _estimate_preview_timeout_sec(str(path))
                self.assertIsNotNone(estimated)
                self.assertGreater(estimated, 300)
                self.assertEqual(_default_timeout_sec(purpose="preview_inspect", adata_path=str(path)), estimated)
                os.environ["CYTOBRIDGE_PREVIEW_SUBPROCESS_MAX_TIMEOUT_SEC"] = "360"
                self.assertEqual(_default_timeout_sec(purpose="preview", adata_path=str(path)), 360)
        finally:
            if old_timeout is not None:
                os.environ["CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC"] = old_timeout
            if old_max is None:
                os.environ.pop("CYTOBRIDGE_PREVIEW_SUBPROCESS_MAX_TIMEOUT_SEC", None)
            else:
                os.environ["CYTOBRIDGE_PREVIEW_SUBPROCESS_MAX_TIMEOUT_SEC"] = old_max

    def test_preview_timeout_for_retinal_scale_is_bounded(self) -> None:
        estimate = _estimate_preview_timeout_from_counts([90053, 73261])
        self.assertGreaterEqual(estimate, 600)
        self.assertLessEqual(estimate, 660)

    def test_training_subprocess_timeout_tracks_dataset_budget(self) -> None:
        old_timeout = os.environ.pop("CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                counts = [4638, 14985, 29679]
                adata = ad.AnnData(np.zeros((sum(counts), 1), dtype=np.float32))
                adata.obs["time_point_processed"] = np.repeat([0.0, 1.0, 2.0], counts)
                path = Path(tmp) / "weinreb_like.h5ad"
                adata.write_h5ad(path)

                self.assertEqual(_estimate_training_budget_sec(str(path)), 780)
                self.assertEqual(_default_timeout_sec(purpose="training", adata_path=str(path)), 1680)

                large_counts = [120000, 120000]
                large = ad.AnnData(np.zeros((sum(large_counts), 1), dtype=np.float32))
                large.obs["time_point_processed"] = np.repeat([0.0, 1.0], large_counts)
                large_path = Path(tmp) / "large_training.h5ad"
                large.write_h5ad(large_path)
                self.assertEqual(_default_timeout_sec(purpose="training", adata_path=str(large_path)), 3600)
        finally:
            if old_timeout is not None:
                os.environ["CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC"] = old_timeout

    def test_training_subprocess_timeout_honors_env_override(self) -> None:
        old_timeout = os.environ.get("CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC")
        os.environ["CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC"] = "77"
        try:
            self.assertEqual(_default_timeout_sec(purpose="training", adata_path="/missing.h5ad"), 77)
        finally:
            if old_timeout is None:
                os.environ.pop("CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC", None)
            else:
                os.environ["CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC"] = old_timeout


if __name__ == "__main__":
    unittest.main()
