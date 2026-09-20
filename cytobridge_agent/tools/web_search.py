from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import httpx


DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
DEFAULT_TIMEOUT_SEC = 20.0
DEFAULT_COUNT = 5
MAX_COUNT = 10


def _normalize_count(count: int) -> int:
    try:
        value = int(count)
    except Exception:
        value = DEFAULT_COUNT
    return max(1, min(value, MAX_COUNT))


def _normalize_allowed_domains(value: Optional[List[str]]) -> List[str]:
    if not value:
        return []
    normalized = []
    seen = set()
    for item in value:
        text = str(item or "").strip().lower()
        if not text:
            continue
        if text.startswith("http://") or text.startswith("https://"):
            parsed = urlparse(text)
            text = (parsed.hostname or "").strip().lower()
        if text.startswith("www."):
            text = text[4:]
        if text and text not in seen:
            seen.add(text)
            normalized.append(text)
    return normalized


def _hostname(url: str) -> str:
    try:
        return (urlparse(str(url or "")).hostname or "").strip().lower()
    except Exception:
        return ""


def _is_allowed_domain(url: str, allowed_domains: List[str]) -> bool:
    if not allowed_domains:
        return True
    host = _hostname(url)
    if not host:
        return False
    if host.startswith("www."):
        host = host[4:]
    for allowed in allowed_domains:
        if host == allowed or host.endswith(f".{allowed}"):
            return True
    return False


def _detect_ddg_bot_challenge(html: str) -> bool:
    if re.search(r'class="[^"]*\bresult__a\b[^"]*"', html, flags=re.IGNORECASE):
        return False
    return bool(
        re.search(
            r"g-recaptcha|are you a human|id=\"challenge-form\"|name=\"challenge\"",
            html,
            flags=re.IGNORECASE,
        )
    )


def _decode_html_entities(text: str) -> str:
    return (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
        .replace("&#x2F;", "/")
        .replace("&nbsp;", " ")
        .replace("&ndash;", "-")
        .replace("&mdash;", "--")
        .replace("&hellip;", "...")
    )


def _strip_html(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def _read_href_attribute(tag_attributes: str) -> str:
    match = re.search(r'\bhref="([^"]*)"', tag_attributes, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _build_result_summary(results: List[Dict[str, str]], *, limit: int = 3) -> str:
    lines: List[str] = []
    for item in results[: max(1, int(limit or 3))]:
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if title and snippet:
            lines.append(f"{title}: {snippet}")
        elif title:
            lines.append(title)
        elif snippet:
            lines.append(snippet)
    return "\n".join(lines).strip()


def _parse_duckduckgo_results(html: str, *, allowed_domains: List[str], count: int) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    seen_urls = set()
    result_regex = re.compile(
        r'<a\b(?=[^>]*\bclass="[^"]*\bresult__a\b[^"]*")([^>]*)>([\s\S]*?)</a>',
        flags=re.IGNORECASE,
    )
    next_result_regex = re.compile(
        r'<a\b(?=[^>]*\bclass="[^"]*\bresult__a\b[^"]*")[^>]*>',
        flags=re.IGNORECASE,
    )
    snippet_regex = re.compile(
        r'<(?:a|div)\b(?=[^>]*\bclass="[^"]*\bresult__snippet\b[^"]*")[^>]*>([\s\S]*?)</(?:a|div)>',
        flags=re.IGNORECASE,
    )
    for match in result_regex.finditer(html):
        raw_attributes = match.group(1) or ""
        raw_title = match.group(2) or ""
        href = _decode_html_entities(_read_href_attribute(raw_attributes))
        title = _decode_html_entities(_strip_html(raw_title))
        if not href or not title:
            continue
        if href.startswith("//"):
            href = f"https:{href}"
        parsed = urlparse(href)
        uddg_values = parse_qs(parsed.query).get("uddg", [])
        uddg = unquote(uddg_values[0]) if uddg_values else ""
        if uddg:
            try:
                href = httpx.URL(uddg).unicode_string()
            except Exception:
                href = uddg
        if not title or not _is_allowed_domain(href, allowed_domains):
            continue
        match_end = match.end()
        trailing_html = html[match_end:]
        next_result_match = next_result_regex.search(trailing_html)
        scoped_trailing_html = (
            trailing_html[: next_result_match.start()] if next_result_match else trailing_html
        )
        raw_snippet_match = snippet_regex.search(scoped_trailing_html)
        raw_snippet = raw_snippet_match.group(1) if raw_snippet_match else ""
        snippet = _decode_html_entities(_strip_html(raw_snippet))
        if href in seen_urls:
            continue
        seen_urls.add(href)
        results.append(
            {
                "title": title,
                "url": href,
                "snippet": snippet,
            }
        )
        if len(results) >= count:
            break
    return results


def _search_duckduckgo(query: str, *, count: int, allowed_domains: List[str], timeout_sec: float) -> Dict[str, Any]:
    response = httpx.get(
        DDG_HTML_ENDPOINT,
        params={"q": query},
        timeout=timeout_sec,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            )
        },
        follow_redirects=True,
    )
    response.raise_for_status()
    html = response.text
    if _detect_ddg_bot_challenge(html):
        raise RuntimeError("DuckDuckGo returned a bot-detection challenge.")
    results = _parse_duckduckgo_results(html, allowed_domains=allowed_domains, count=count)
    return {
        "success": True,
        "backend": "duckduckgo",
        "query": query,
        "summary": _build_result_summary(results),
        "results": results,
        "notes": [
            "Free DuckDuckGo HTML fallback; results are snippet-level and may be less stable than API-backed search."
        ],
    }


def _search_searxng(
    query: str,
    *,
    count: int,
    allowed_domains: List[str],
    timeout_sec: float,
    base_url: str,
) -> Dict[str, Any]:
    trimmed = str(base_url or "").strip().rstrip("/")
    if not trimmed:
        raise RuntimeError("SEARXNG_BASE_URL is empty.")
    endpoint = f"{trimmed}/search"
    response = httpx.get(
        endpoint,
        params={"q": query, "format": "json"},
        timeout=timeout_sec,
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    results: List[Dict[str, str]] = []
    for item in payload.get("results", []) or []:
        url = str(item.get("url") or "").strip()
        if not url or not _is_allowed_domain(url, allowed_domains):
            continue
        results.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": url,
                "snippet": str(item.get("content") or item.get("snippet") or "").strip(),
            }
        )
        if len(results) >= count:
            break
    return {
        "success": True,
        "backend": "searxng",
        "query": query,
        "summary": _build_result_summary(results),
        "results": results,
        "notes": [f"Using configured SearXNG instance: {trimmed}"],
    }


def search_web_free(
    query: str,
    *,
    count: int = DEFAULT_COUNT,
    allowed_domains: Optional[List[str]] = None,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    searxng_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {
            "success": False,
            "backend": "none",
            "query": "",
            "summary": "",
            "results": [],
            "error": "query must not be empty",
        }

    normalized_count = _normalize_count(count)
    normalized_domains = _normalize_allowed_domains(allowed_domains)
    try:
        if str(searxng_base_url or "").strip():
            return _search_searxng(
                normalized_query,
                count=normalized_count,
                allowed_domains=normalized_domains,
                timeout_sec=timeout_sec,
                base_url=str(searxng_base_url or "").strip(),
            )
        return _search_duckduckgo(
            normalized_query,
            count=normalized_count,
            allowed_domains=normalized_domains,
            timeout_sec=timeout_sec,
        )
    except Exception as exc:
        return {
            "success": False,
            "backend": "searxng" if str(searxng_base_url or "").strip() else "duckduckgo",
            "query": normalized_query,
            "summary": "",
            "results": [],
            "error": str(exc),
        }
