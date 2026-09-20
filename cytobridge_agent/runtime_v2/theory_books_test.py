from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cytobridge_agent.rag.theory_books import TheoryBookIndex, format_theory_results
from cytobridge_agent.runtime_v2 import tool_registry


class _FakeEmbeddingModel:
    def encode(self, texts, show_progress_bar=False):
        vectors = []
        for text in texts:
            lowered = str(text).lower()
            vectors.append(
                [
                    1.0 if "benamou" in lowered or "dynamic ot" in lowered else 0.0,
                    1.0 if "schrodinger" in lowered else 0.0,
                    1.0,
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


class TheoryBookIndexTests(unittest.TestCase):
    def test_search_theory_returns_page_ranges_and_read_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            books_dir = root / "books"
            books_dir.mkdir()
            pdf_path = books_dir / "ot_book.pdf"
            pdf_path.write_bytes(b"%PDF-1.7\n")

            pages = [
                {"page": 1, "text": "Preface and unrelated material."},
                {"page": 2, "text": "Chapter 3 Dynamic OT\nBenamou Brenier dynamic OT minimizes kinetic energy."},
                {"page": 3, "text": "Schrodinger bridge material."},
            ]

            with patch("cytobridge_agent.rag.theory_books._extract_pdf_pages", return_value=(pages, {"Title": "OT Book"})):
                    with patch("cytobridge_agent.rag.theory_books.get_sentence_transformer_model", return_value=_FakeEmbeddingModel()):
                        index = TheoryBookIndex(books_dir=books_dir, base_dir=root / "index")
                        payload = index.search("Benamou Brenier dynamic OT", top_k=1)

            self.assertEqual(len(payload["results"]), 1)
            item = payload["results"][0]
            self.assertEqual(item["title"], "OT Book")
            self.assertEqual(item["page"], 2)
            self.assertEqual(item["recommended_read_pages"], "1-3")
            self.assertIn("read_file(file_path=", item["read_text_command"])
            self.assertIn("pdf_mode=\"text\"", item["read_text_command"])

            formatted = format_theory_results(payload)
            self.assertIn("Theory Search Results", formatted)
            self.assertIn("recommended_read_pages: 1-3", formatted)

    def test_search_theory_indexes_extra_theory_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            books_dir = root / "books"
            books_dir.mkdir()
            extra_pdf = root / "uot_dynamic.pdf"
            extra_pdf.write_bytes(b"%PDF-1.7\n")

            pages = [
                {
                    "page": 4,
                    "text": "Dynamic unbalanced optimal transport uses a continuity equation with growth.",
                }
            ]

            with patch("cytobridge_agent.rag.theory_books._extract_pdf_pages", return_value=(pages, {"Title": "UOT Dynamic"})):
                with patch("cytobridge_agent.rag.theory_books.get_sentence_transformer_model", return_value=_FakeEmbeddingModel()):
                    index = TheoryBookIndex(
                        books_dir=books_dir,
                        base_dir=root / "index",
                        extra_pdf_paths=[extra_pdf],
                    )
                    payload = index.search("dynamic unbalanced optimal transport growth", top_k=1)

            self.assertEqual(len(payload["results"]), 1)
            item = payload["results"][0]
            self.assertEqual(item["title"], "UOT Dynamic")
            self.assertEqual(item["pdf_path"], str(extra_pdf.resolve()))

    def test_search_theory_relocates_stale_pdf_paths_by_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            books_dir = root / "books"
            books_dir.mkdir()
            pdf_path = books_dir / "ot_book.pdf"
            pdf_path.write_bytes(b"%PDF-1.7\n")

            pages = [
                {
                    "page": 2,
                    "text": "Chapter 3 Dynamic OT\nBenamou Brenier dynamic OT minimizes kinetic energy.",
                }
            ]

            with patch("cytobridge_agent.rag.theory_books._extract_pdf_pages", return_value=(pages, {"Title": "OT Book"})):
                with patch("cytobridge_agent.rag.theory_books.get_sentence_transformer_model", return_value=_FakeEmbeddingModel()):
                    index = TheoryBookIndex(books_dir=books_dir, base_dir=root / "index", extra_pdf_paths=[])
                    index.build(force_rebuild=True)

                    index_path = root / "index" / "book_index.json"
                    payload = json.loads(index_path.read_text(encoding="utf-8"))
                    book = payload["books"]["ot_book"]
                    book["pdf_path"] = "/Users/zhenyizhang/old-machine/ot_book.pdf"
                    for chunk in book["chunks"]:
                        chunk["pdf_path"] = "/Users/zhenyizhang/old-machine/ot_book.pdf"
                    index_path.write_text(json.dumps(payload), encoding="utf-8")

                    index = TheoryBookIndex(books_dir=books_dir, base_dir=root / "index", extra_pdf_paths=[])
                    payload = index.search("Benamou Brenier dynamic OT", top_k=1)

            self.assertEqual(len(payload["results"]), 1)
            item = payload["results"][0]
            self.assertEqual(item["pdf_path"], str(pdf_path.resolve()))
            self.assertNotIn("zhenyizhang", item["read_text_command"])

    def test_search_theory_tool_is_registered(self) -> None:
        tools = tool_registry.SingleAgentTools(llm=None, state={}, agent_role="planner").get_tools()
        names = {tool.name for tool in tools}
        self.assertIn("search_theory", names)


if __name__ == "__main__":
    unittest.main()
