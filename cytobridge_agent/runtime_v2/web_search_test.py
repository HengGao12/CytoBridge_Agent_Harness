from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from cytobridge_agent.runtime_v2.state import ensure_runtime_v2_state
from cytobridge_agent.runtime_v2.tool_registry import SingleAgentTools
from cytobridge_agent.tools.web_search import search_web_free


class DummyLLM:
    pass


class DummyCodexLLM:
    _llm_type = "codex-oauth-sidecar"
    model_name = "gpt-5.4"
    preferred_profile_id = "openai-codex:test"


class _FakeHttpxResponse:
    def __init__(self, *, text: str = "", json_payload=None):
        self.text = text
        self._json_payload = json_payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._json_payload


class WebSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.state = ensure_runtime_v2_state(
            {
                "session_id": "web-search-test",
                "output_dir": self.tmpdir.name,
            }
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_search_web_free_duckduckgo_parses_results_and_filters_domains(self) -> None:
        html = """
        <html><body>
          <div class="result">
            <a class="result__a" href="https://example.com/paper">Example Result</a>
            <div class="result__snippet">Example snippet</div>
          </div>
          <div class="result">
            <a class="result__a" href="https://other.org/post">Other Result</a>
            <div class="result__snippet">Other snippet</div>
          </div>
        </body></html>
        """

        with patch("cytobridge_agent.tools.web_search.httpx.get", return_value=_FakeHttpxResponse(text=html)):
            result = search_web_free(
                "test query",
                count=5,
                allowed_domains=["example.com"],
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["backend"], "duckduckgo")
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["url"], "https://example.com/paper")

    def test_search_web_free_prefers_configured_searxng(self) -> None:
        payload = {
            "results": [
                {"title": "One", "url": "https://example.com/a", "content": "alpha"},
                {"title": "Two", "url": "https://blocked.net/b", "content": "beta"},
            ]
        }

        with patch("cytobridge_agent.tools.web_search.httpx.get", return_value=_FakeHttpxResponse(json_payload=payload)):
            result = search_web_free(
                "test query",
                count=5,
                allowed_domains=["example.com"],
                searxng_base_url="http://localhost:8888",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["backend"], "searxng")
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["title"], "One")

    def test_system_prompt_tools_web_search_uses_free_backend_for_regular_models(self) -> None:
        tools = SingleAgentTools(DummyLLM(), self.state, agent_role="planner", agent_id="planner")
        with patch(
            "cytobridge_agent.runtime_v2.tool_registry.search_web_free",
            return_value={"success": True, "backend": "duckduckgo", "query": "q", "summary": "", "results": []},
        ) as mock_search:
            result = tools.web_search("q", count=3, allowed_domains=["example.com"])

        self.assertEqual(result["backend"], "duckduckgo")
        mock_search.assert_called_once()

    def test_system_prompt_tools_web_search_uses_codex_native_backend_for_codex_oauth(self) -> None:
        tools = SingleAgentTools(DummyCodexLLM(), self.state, agent_role="planner", agent_id="planner")
        with patch(
            "cytobridge_agent.runtime_v2.tool_registry.codex_native_web_search",
            return_value={
                "backend": "openai_codex_native",
                "content": "Grounded answer",
                "sources": [{"title": "Source", "url": "https://example.com"}],
                "usage": {"total_tokens": 42},
                "mode": "cached",
            },
        ) as mock_search:
            result = tools.web_search("q", count=3, allowed_domains=["example.com"])

        self.assertTrue(result["success"])
        self.assertEqual(result["backend"], "openai_codex_native")
        self.assertEqual(result["summary"], "Grounded answer")
        self.assertEqual(result["results"][0]["url"], "https://example.com")
        mock_search.assert_called_once()

    def test_system_prompt_tools_web_search_falls_back_when_codex_native_search_fails(self) -> None:
        tools = SingleAgentTools(DummyCodexLLM(), self.state, agent_role="planner", agent_id="planner")
        with patch(
            "cytobridge_agent.runtime_v2.tool_registry.codex_native_web_search",
            side_effect=RuntimeError("native search unavailable"),
        ) as mock_native, patch(
            "cytobridge_agent.runtime_v2.tool_registry.search_web_free",
            return_value={
                "success": True,
                "backend": "duckduckgo",
                "query": "q",
                "summary": "Fallback result",
                "results": [{"title": "Fallback", "url": "https://example.com", "snippet": ""}],
                "notes": [],
            },
        ) as mock_fallback:
            result = tools.web_search("q", count=3, allowed_domains=["example.com"])

        self.assertTrue(result["success"])
        self.assertEqual(result["backend"], "duckduckgo")
        self.assertEqual(result["fallback_from"], "openai_codex_native")
        self.assertIn("native search unavailable", " ".join(result.get("notes") or []))
        mock_native.assert_called_once()
        mock_fallback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
