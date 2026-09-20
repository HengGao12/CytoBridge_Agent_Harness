from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cytobridge_agent.tools import file_tools


class FileToolsReadTests(unittest.TestCase):
    def _write_text(self, root: str, name: str, content: str) -> str:
        path = Path(root) / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    def test_read_text_file_fast_path_reports_actual_total_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_text(
                tmpdir,
                "sample.txt",
                "line 1\nline 2\nline 3\nline 4\nline 5\n",
            )

            result = file_tools.read_text_file(path=path, start_line=2, end_line=3, max_chars=20000)

            self.assertIn("Line range: 2-3", result)
            self.assertIn("Total lines: 5", result)
            self.assertIn("L2: line 2", result)
            self.assertIn("L3: line 3", result)
            self.assertNotIn("L4: line 4", result)

    def test_read_text_file_streaming_path_truncates_large_selected_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "\n".join(
                [
                    "a" * 120,
                    "b" * 120,
                    "c" * 120,
                ]
            )
            path = self._write_text(tmpdir, "large_lines.txt", content)

            with patch.object(file_tools, "READ_TEXT_FAST_PATH_MAX_BYTES", 1):
                result = file_tools.read_text_file(path=path, start_line=1, end_line=3, max_chars=80)

            self.assertIn("Truncated: true", result)
            self.assertIn("...[truncated ", result)
            self.assertIn("L1: ", result)
            self.assertIn("cccccccc", result)
            self.assertLess(len(result), 500)

    def test_read_file_keeps_default_line_window_on_streaming_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "\n".join(f"line {idx}" for idx in range(1, 2601))
            path = self._write_text(tmpdir, "many_lines.txt", content)

            with patch.object(file_tools, "READ_TEXT_FAST_PATH_MAX_BYTES", 1):
                result = file_tools.read_file(file_path=path, offset=2500)

            self.assertIn("Line range: 2500-2600", result)
            self.assertIn("L2500: line 2500", result)
            self.assertIn("L2600: line 2600", result)
            self.assertNotIn("L1: line 1", result)


if __name__ == "__main__":
    unittest.main()
