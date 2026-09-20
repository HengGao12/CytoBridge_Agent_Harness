from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np

from cytobridge_agent.runtime_v2.tool_registry import SingleAgentTools
from cytobridge_agent.tools.adata_manager import AnnDataManager
from cytobridge_agent.tools.code_executor import CodeExecutor


class AnnDataIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        AnnDataManager.reset_instance()

    def tearDown(self) -> None:
        AnnDataManager.reset_instance()

    def _write_adata(self, path: Path) -> None:
        adata = ad.AnnData(np.ones((3, 2)))
        adata.obs["time_point_processed"] = ["0", "1", "1"]
        adata.write_h5ad(path)

    def test_load_or_switch_adata_binds_path_without_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.h5ad"
            self._write_adata(path)
            handler = object.__new__(SingleAgentTools)
            handler.state = {}
            handler._emit_event = lambda *args, **kwargs: None

            message = handler.load_or_switch_adata(str(path))

            manager = AnnDataManager()
            self.assertIn("Bound adata path", message)
            self.assertEqual(manager.get_path(), str(path.resolve()))
            self.assertFalse(hasattr(manager, "get"))
            self.assertEqual(handler.state["input_path"], str(path.resolve()))

    def test_inspect_adata_state_reports_bound_path_not_loaded_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.h5ad"
            self._write_adata(path)
            AnnDataManager().bind_path(str(path))
            handler = object.__new__(SingleAgentTools)
            handler._emit_event = lambda *args, **kwargs: None

            payload = json.loads(handler.inspect_adata_state())

            self.assertFalse(payload["loaded_in_main_process"])
            self.assertTrue(payload["path_bound_for_isolated_tools"])
            self.assertEqual(payload["path"], str(path.resolve()))

    def test_code_executor_loads_bound_input_path_only_in_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.h5ad"
            self._write_adata(path)
            output_dir = Path(tmp) / "out"
            AnnDataManager().bind_path(str(path))
            executor = CodeExecutor(adata=None, output_dir=output_dir, input_path=str(path))

            result = executor.execute("print('n_obs', adata.n_obs)", timeout=20, timeout_mode="isolated")

            self.assertTrue(result["success"], result)
            self.assertIn("n_obs 3", result["stdout"])
            self.assertIsNone(executor.adata)
            self.assertFalse(hasattr(AnnDataManager(), "get"))
            self.assertEqual(AnnDataManager().get_path(), str(output_dir / "converted_input.h5ad"))


if __name__ == "__main__":
    unittest.main()
