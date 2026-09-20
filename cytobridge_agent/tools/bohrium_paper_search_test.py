from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from cytobridge_agent.tools.bohrium_paper_search import get_bohrium_paper_search_status, search_bohrium_papers


class _FakeBohriumResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class BohriumPaperSearchTests(unittest.TestCase):
    def test_search_bohrium_papers_returns_clear_error_without_key(self) -> None:
        with patch.dict(os.environ, {"ACCESS_KEY": "", "BOHRIUM_ACCESS_KEY": ""}, clear=False), patch(
            "cytobridge_agent.tools.bohrium_paper_search.httpx.post"
        ) as mock_post:
            result = search_bohrium_papers(query="single cell trajectory", page_size=2)

        self.assertFalse(result["success"])
        self.assertEqual(result["backend"], "bohrium-paper-search")
        self.assertEqual(result["results"], [])
        self.assertIn("ACCESS_KEY or BOHRIUM_ACCESS_KEY is not set", result["error"])
        self.assertFalse(result["availability"]["available"])
        self.assertFalse(result["availability"]["safe_to_call"])
        mock_post.assert_not_called()

    def test_bohrium_status_reports_availability_without_secret_value(self) -> None:
        secret = "test-secret-value"
        with patch.dict(os.environ, {"BOHRIUM_ACCESS_KEY": secret, "ACCESS_KEY": ""}, clear=False):
            status = get_bohrium_paper_search_status()

        self.assertTrue(status["available"])
        self.assertTrue(status["safe_to_call"])
        self.assertEqual(status["configured_env_var"], "BOHRIUM_ACCESS_KEY")
        self.assertNotIn(secret, str(status))
        self.assertFalse(status["secret_value_exposed"])

    def test_search_bohrium_papers_parses_successful_response(self) -> None:
        payload = {
            "code": 0,
            "data": [
                {
                    "enName": "Trajectory inference with optimal transport",
                    "doi": "10.1234/example",
                    "paperId": "paper-1",
                    "enAbstract": "Abstract text",
                    "authors": [{"name": "A. Author"}],
                    "publicationEnName": "Example Journal",
                    "coverDateStart": "2026-01-01",
                    "impactFactor": 12.3,
                    "citationNums": 42,
                    "popularity": 0.9,
                    "pieces": [{"text": "snippet"}],
                    "figures": [{"caption": "figure"}],
                }
            ],
        }

        with patch.dict(os.environ, {"BOHRIUM_ACCESS_KEY": "test-key"}, clear=False), patch(
            "cytobridge_agent.tools.bohrium_paper_search.httpx.post",
            return_value=_FakeBohriumResponse(payload),
        ) as mock_post:
            result = search_bohrium_papers(query="single cell optimal transport", page_size=2)

        self.assertTrue(result["success"])
        self.assertEqual(result["results"][0]["title"], "Trajectory inference with optimal transport")
        self.assertEqual(result["results"][0]["doi"], "10.1234/example")
        self.assertIn("Trajectory inference", result["summary"])
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["accessKey"], "test-key")
        self.assertLessEqual(kwargs["json"]["pageSize"], 100)


if __name__ == "__main__":
    unittest.main()
