from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import httpx
from bs4 import BeautifulSoup as RealBeautifulSoup
from bs4 import FeatureNotFound

from cytobridge_agent.runtime_v2.state import ensure_runtime_v2_state
from cytobridge_agent.runtime_v2.tool_registry import SingleAgentTools
from cytobridge_agent.tools.web_fetch import fetch_web_content


class DummyLLM:
    pass


class _FakeHttpxResponse:
    def __init__(
        self,
        *,
        text: str = "",
        headers: dict | None = None,
        status_code: int = 200,
        url: str = "https://example.com/article",
    ):
        self.text = text
        self.headers = headers or {}
        self.status_code = status_code
        self.url = url

    def raise_for_status(self) -> None:
        return None


class WebFetchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.state = ensure_runtime_v2_state(
            {
                "session_id": "web-fetch-test",
                "output_dir": self.tmpdir.name,
            }
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_fetch_web_content_html_readable_prefers_trafilatura(self) -> None:
        html = """
        <html>
          <head><title>Example Article</title></head>
          <body><article><p>Ignored by primary extractor.</p></article></body>
        </html>
        """
        response = _FakeHttpxResponse(text=html, headers={"content-type": "text/html; charset=utf-8"})

        with patch("cytobridge_agent.tools.web_fetch._validate_resolved_addresses", return_value=None), patch(
            "cytobridge_agent.tools.web_fetch.httpx.get",
            return_value=response,
        ), patch(
            "cytobridge_agent.tools.web_fetch._extract_with_trafilatura",
            return_value="Primary extracted body.",
        ):
            result = fetch_web_content("https://example.com/article")

        self.assertTrue(result["success"])
        self.assertEqual(result["backend"], "managed_http_fetch")
        self.assertEqual(result["title"], "Example Article")
        self.assertEqual(result["content"], "Primary extracted body.")
        self.assertIn("trafilatura", " ".join(result.get("notes") or []).lower())

    def test_fetch_web_content_raw_text_uses_visible_text_fallback(self) -> None:
        html = """
        <html>
          <head><title>Example Raw Text</title><script>bad()</script></head>
          <body><main><h1>Heading</h1><p>Paragraph one.</p><p>Paragraph two.</p></main></body>
        </html>
        """
        response = _FakeHttpxResponse(text=html, headers={"content-type": "text/html"})

        with patch("cytobridge_agent.tools.web_fetch._validate_resolved_addresses", return_value=None), patch(
            "cytobridge_agent.tools.web_fetch.httpx.get",
            return_value=response,
        ):
            result = fetch_web_content("https://example.com/raw", extract_mode="raw_text")

        self.assertTrue(result["success"])
        self.assertIn("Heading", result["content"])
        self.assertIn("Paragraph one.", result["content"])
        self.assertNotIn("bad()", result["content"])

    def test_fetch_web_content_falls_back_when_lxml_is_unavailable(self) -> None:
        html = """
        <html>
          <head><title>No lxml</title><meta name="description" content="Fallback parser"></head>
          <body><main><p>Visible fallback body.</p></main></body>
        </html>
        """
        response = _FakeHttpxResponse(text=html, headers={"content-type": "text/html"})

        def fake_bs4(markup, parser=None, *args, **kwargs):
            if parser == "lxml":
                raise FeatureNotFound("lxml unavailable")
            return RealBeautifulSoup(markup, parser, *args, **kwargs)

        with patch("cytobridge_agent.tools.web_fetch._validate_resolved_addresses", return_value=None), patch(
            "cytobridge_agent.tools.web_fetch.httpx.get",
            return_value=response,
        ), patch("cytobridge_agent.tools.web_fetch.BeautifulSoup", side_effect=fake_bs4):
            result = fetch_web_content("https://example.com/fallback", extract_mode="raw_text")

        self.assertTrue(result["success"])
        self.assertEqual(result["title"], "No lxml")
        self.assertEqual(result["description"], "Fallback parser")
        self.assertIn("Visible fallback body.", result["content"])

    def test_fetch_web_content_handles_text_response_directly(self) -> None:
        response = _FakeHttpxResponse(
            text="plain text body",
            headers={"content-type": "text/plain; charset=utf-8"},
            url="https://example.com/text",
        )
        with patch("cytobridge_agent.tools.web_fetch._validate_resolved_addresses", return_value=None), patch(
            "cytobridge_agent.tools.web_fetch.httpx.get",
            return_value=response,
        ):
            result = fetch_web_content("https://example.com/text")

        self.assertTrue(result["success"])
        self.assertEqual(result["content"], "plain text body")
        self.assertEqual(result["content_type"], "text/plain; charset=utf-8")

    def test_fetch_web_content_rejects_private_destination(self) -> None:
        result = fetch_web_content("http://localhost:8080/secret")
        self.assertFalse(result["success"])
        self.assertIn("localhost", result["error"].lower())

    def test_fetch_web_content_rejects_binary_content(self) -> None:
        response = _FakeHttpxResponse(
            text="%PDF-1.7",
            headers={"content-type": "application/pdf"},
            url="https://example.com/file.pdf",
        )
        with patch("cytobridge_agent.tools.web_fetch._validate_resolved_addresses", return_value=None), patch(
            "cytobridge_agent.tools.web_fetch.httpx.get",
            return_value=response,
        ):
            result = fetch_web_content("https://example.com/file.pdf")

        self.assertFalse(result["success"])
        self.assertIn("unsupported content type", result["error"].lower())

    def test_fetch_web_content_http_error_returns_structured_error(self) -> None:
        request = httpx.Request("GET", "https://example.com/blocked")
        response = httpx.Response(403, request=request)
        error = httpx.HTTPStatusError("403 Forbidden", request=request, response=response)
        with patch("cytobridge_agent.tools.web_fetch._validate_resolved_addresses", return_value=None), patch(
            "cytobridge_agent.tools.web_fetch.httpx.get",
            side_effect=error,
        ):
            result = fetch_web_content("https://example.com/blocked")

        self.assertFalse(result["success"])
        self.assertEqual(result["status_code"], 403)
        self.assertIn("status 403", result["error"].lower())

    def test_fetch_web_content_truncates_long_output(self) -> None:
        response = _FakeHttpxResponse(
            text="A" * 5000,
            headers={"content-type": "text/plain"},
            url="https://example.com/long",
        )
        with patch("cytobridge_agent.tools.web_fetch._validate_resolved_addresses", return_value=None), patch(
            "cytobridge_agent.tools.web_fetch.httpx.get",
            return_value=response,
        ):
            result = fetch_web_content("https://example.com/long", max_chars=1200)

        self.assertTrue(result["success"])
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(result["content"]), 1200)

    def test_system_prompt_tools_web_fetch_uses_managed_backend(self) -> None:
        tools = SingleAgentTools(DummyLLM(), self.state, agent_role="planner", agent_id="planner")
        with patch(
            "cytobridge_agent.runtime_v2.tool_registry.fetch_web_content",
            return_value={
                "success": True,
                "backend": "managed_http_fetch",
                "url": "https://example.com",
                "final_url": "https://example.com",
                "status_code": 200,
                "content_type": "text/html",
                "title": "Example",
                "description": "",
                "content": "hello",
                "truncated": False,
                "extract_mode": "readable",
                "content_length": 5,
                "notes": [],
            },
        ) as mock_fetch:
            result = tools.web_fetch("https://example.com")

        self.assertTrue(result["success"])
        self.assertEqual(result["backend"], "managed_http_fetch")
        mock_fetch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
