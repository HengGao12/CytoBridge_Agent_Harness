from __future__ import annotations

import tempfile
import textwrap
import time
import unittest
import os
from pathlib import Path

from cytobridge_agent.tools.code_executor import CodeExecutor


class CodeExecutorTimeoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old_start_method = os.environ.get("CYTOBRIDGE_CODE_EXECUTOR_START_METHOD")
        self._old_home = os.environ.get("HOME")
        self._old_guard_budget = os.environ.get("CYTOBRIDGE_EXECUTE_PYTHON_MANAGED_GUARD_BUDGET_SEC")
        self._home_tmp = tempfile.TemporaryDirectory()
        os.environ["CYTOBRIDGE_CODE_EXECUTOR_START_METHOD"] = "fork"
        os.environ["HOME"] = self._home_tmp.name

    def tearDown(self) -> None:
        if self._old_start_method is None:
            os.environ.pop("CYTOBRIDGE_CODE_EXECUTOR_START_METHOD", None)
        else:
            os.environ["CYTOBRIDGE_CODE_EXECUTOR_START_METHOD"] = self._old_start_method
        if self._old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self._old_home
        if self._old_guard_budget is None:
            os.environ.pop("CYTOBRIDGE_EXECUTE_PYTHON_MANAGED_GUARD_BUDGET_SEC", None)
        else:
            os.environ["CYTOBRIDGE_EXECUTE_PYTHON_MANAGED_GUARD_BUDGET_SEC"] = self._old_guard_budget
        self._home_tmp.cleanup()

    def test_isolated_execute_python_times_out_by_wall_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = CodeExecutor(adata=None, output_dir=Path(tmp), input_path=None)

            started = time.monotonic()
            result = executor.execute(
                "import time\nprint('started')\ntime.sleep(5)\nprint('finished')",
                timeout=1,
                timeout_mode="isolated",
            )
            elapsed = time.monotonic() - started

            self.assertFalse(result["success"], result)
            self.assertTrue(result["timed_out"], result)
            self.assertIn("Timed Out", result["error"])
            self.assertLess(elapsed, 4.0)

    def test_isolated_saved_script_times_out_by_wall_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script = tmp_path / "slow_script.py"
            script.write_text(
                textwrap.dedent(
                    """
                    import time
                    print('started')
                    time.sleep(5)
                    print('finished')
                    """
                ).strip()
            )
            executor = CodeExecutor(adata=None, output_dir=tmp_path, input_path=None)

            started = time.monotonic()
            result = executor.execute_script_file(
                str(script),
                timeout=1,
                timeout_mode="isolated",
            )
            elapsed = time.monotonic() - started

            self.assertFalse(result["success"], result)
            self.assertTrue(result["timed_out"], result)
            self.assertIn("timed out", result["error"])
            self.assertLess(elapsed, 4.0)

    def test_isolated_saved_script_honors_requested_start_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script = tmp_path / "start_method_script.py"
            script.write_text(
                textwrap.dedent(
                    """
                    import multiprocessing as mp
                    print('start_method', mp.get_start_method())
                    """
                ).strip()
            )
            executor = CodeExecutor(adata=None, output_dir=tmp_path, input_path=None)

            os.environ["CYTOBRIDGE_CODE_EXECUTOR_START_METHOD"] = "forkserver"
            result = executor.execute_script_file(
                str(script),
                timeout=40,
                timeout_mode="isolated",
            )

            self.assertTrue(result["success"], result)
            self.assertIn("start_method forkserver", result["stdout"])

    def test_managed_write_guard_scopes_snapshot_to_active_owner(self) -> None:
        home = Path(self._home_tmp.name)
        target = home / ".cellcompass" / "training_algorithms" / "target_algo" / "config.yaml"
        target.parent.mkdir(parents=True)
        target.write_text("value: original\n")

        # Simulate a large portfolio that would make global managed snapshots
        # expensive while the active owner remains small.
        crowded = home / ".cellcompass" / "training_algorithms" / "other_algo"
        crowded.mkdir(parents=True)
        for idx in range(1500):
            (crowded / f"file_{idx:04d}.md").write_text(f"other {idx}\n")

        os.environ["CYTOBRIDGE_EXECUTE_PYTHON_MANAGED_GUARD_BUDGET_SEC"] = "0.05"
        with tempfile.TemporaryDirectory() as tmp:
            executor = CodeExecutor(
                adata=None,
                output_dir=Path(tmp),
                input_path=None,
                managed_write_owner="target_algo",
            )
            result = executor.execute(
                "from pathlib import Path\n"
                "Path.home().joinpath('.cellcompass','training_algorithms','target_algo','config.yaml').write_text('value: changed\\n')\n",
                timeout=20,
                timeout_mode="isolated",
            )

        self.assertFalse(result["success"], result)
        self.assertIn("read-only for managed CytoBridge", result["error"])
        self.assertEqual(target.read_text(), "value: original\n")


if __name__ == "__main__":
    unittest.main()
