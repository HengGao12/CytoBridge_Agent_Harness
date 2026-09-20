from __future__ import annotations

import ipaddress
import re
import socket
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

try:
    import trafilatura  # type: ignore
except Exception:  # noqa: BLE001
    trafilatura = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # noqa: BLE001
    BeautifulSoup = None


DEFAULT_TIMEOUT_SEC = 20.0
DEFAULT_MAX_CHARS = 12000
MAX_MAX_CHARS = 50000
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
TEXTISH_CONTENT_TYPES = (
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml+xml",
    "application/javascript",
    "application/x-javascript",
    "application/ld+json",
    "application/rtf",
)


def _normalize_max_chars(value: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = DEFAULT_MAX_CHARS
    return max(1000, min(parsed, MAX_MAX_CHARS))


def _normalize_allowed_domains(value: Optional[List[str]]) -> List[str]:
    if not value:
        return []
    normalized: List[str] = []
    seen = set()
    for item in value:
        text = str(item or "").strip().lower()
        if not text:
            continue
        if text.startswith("http://") or text.startswith("https://"):
            text = (urlparse(text).hostname or "").strip().lower()
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


def _is_ip_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return bool(ip.is_global)


def _validate_hostname(hostname: str) -> Optional[str]:
    host = str(hostname or "").strip().lower()
    if not host:
        return "URL is missing a hostname."
    if host == "localhost" or host.endswith(".localhost"):
        return "Localhost destinations are not allowed."
    try:
        if not _is_ip_public(host):
            return f"Non-public IP address '{host}' is not allowed."
        return None
    except ValueError:
        return None


def _validate_resolved_addresses(hostname: str) -> Optional[str]:
    try:
        addrinfo = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return f"Failed to resolve hostname '{hostname}': {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Could not validate hostname '{hostname}': {exc}"

    seen = set()
    for item in addrinfo:
        sockaddr = item[4]
        if not sockaddr:
            continue
        address = str(sockaddr[0] or "").strip()
        if not address or address in seen:
            continue
        seen.add(address)
        try:
            if not _is_ip_public(address):
                return f"Resolved address '{address}' for '{hostname}' is not public."
        except ValueError:
            return f"Resolved address '{address}' for '{hostname}' is invalid."
    if not seen:
        return f"Hostname '{hostname}' resolved to no usable addresses."
    return None


def _validate_public_url(url: str, *, allowed_domains: List[str]) -> Optional[str]:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return f"Only http/https URLs are allowed, got '{parsed.scheme or 'missing'}'."
    hostname = (parsed.hostname or "").strip().lower()
    host_error = _validate_hostname(hostname)
    if host_error:
        return host_error
    if not _is_allowed_domain(url, allowed_domains):
        return f"URL host '{hostname}' is outside the allowlist."
    return _validate_resolved_addresses(hostname)


def _truncate_content(text: str, *, max_chars: int) -> Tuple[str, bool]:
    normalized = str(text or "").strip()
    if len(normalized) <= max_chars:
        return normalized, False
    marker = f"\n\n...[truncated {len(normalized) - max_chars} chars]..."
    keep = max(0, max_chars - len(marker))
    return normalized[:keep].rstrip() + marker, True


def _is_html_content_type(content_type: str) -> bool:
    normalized = str(content_type or "").lower()
    return "text/html" in normalized or "application/xhtml+xml" in normalized


def _is_text_like_content_type(content_type: str) -> bool:
    normalized = str(content_type or "").lower()
    return any(normalized.startswith(prefix) for prefix in TEXTISH_CONTENT_TYPES)


def _clean_text(text: str) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", str(text or "")))
    return cleaned.strip()


def _html_soup(html: str) -> Any:
    if BeautifulSoup is None:
        return None
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:  # noqa: BLE001
        return BeautifulSoup(html, "html.parser")


def _extract_metadata_from_html(html: str) -> Tuple[str, str]:
    if BeautifulSoup is None:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
        return title, ""

    soup = _html_soup(html)
    title = ""
    if soup and soup.title and soup.title.string:
        title = str(soup.title.string).strip()
    description = ""
    meta = soup.find("meta", attrs={"name": re.compile("^description$", re.IGNORECASE)}) if soup else None
    if meta and meta.get("content"):
        description = str(meta.get("content")).strip()
    return title, description


def _extract_with_trafilatura(html: str, url: str) -> str:
    if trafilatura is None:
        return ""
    try:
        extracted = trafilatura.extract(
            html,
            url=url,
            output_format="txt",
            include_comments=False,
            include_tables=True,
            include_images=False,
            include_links=False,
            favor_recall=True,
        )
    except Exception:  # noqa: BLE001
        return ""
    return _clean_text(extracted or "")


def _node_text(node: Any) -> str:
    if not node:
        return ""
    blocks: List[str] = []
    for child in node.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre", "td", "th"]
    ):
        text = _clean_text(child.get_text("\n", strip=True))
        if text:
            blocks.append(text)
    if not blocks:
        return _clean_text(node.get_text("\n", strip=True))
    return _clean_text("\n\n".join(blocks))


def _extract_with_bs4(html: str, *, raw_text: bool = False) -> str:
    if BeautifulSoup is None:
        return _clean_text(re.sub(r"<[^>]+>", " ", html))
    soup = _html_soup(html)
    if soup is None:
        return _clean_text(re.sub(r"<[^>]+>", " ", html))
    for tag in soup(["script", "style", "noscript", "template", "svg", "canvas", "head"]):
        tag.decompose()
    if raw_text:
        return _clean_text(soup.get_text("\n", strip=True))

    for selector in ("article", "main", "[role='main']"):
        node = soup.select_one(selector)
        text = _node_text(node)
        if text:
            return text

    body = soup.body or soup
    best_node = None
    best_score = -1
    for candidate in body.find_all(["article", "main", "section", "div"], recursive=True):
        text = candidate.get_text(" ", strip=True)
        if len(text) < 200:
            continue
        score = text.count(" ") + (len(candidate.find_all("p")) * 100)
        if score > best_score:
            best_score = score
            best_node = candidate
    text = _node_text(best_node or body)
    if text:
        return text
    return _clean_text(body.get_text("\n", strip=True))


def _extract_html_content(html: str, *, url: str, extract_mode: str) -> Tuple[str, List[str]]:
    notes: List[str] = []
    normalized_mode = str(extract_mode or "readable").strip().lower()
    if normalized_mode == "html":
        notes.append("Returned raw HTML content.")
        return html, notes
    if normalized_mode == "raw_text":
        notes.append("Used raw HTML-to-text extraction.")
        return _extract_with_bs4(html, raw_text=True), notes

    extracted = _extract_with_trafilatura(html, url)
    if extracted:
        notes.append("Primary extraction via trafilatura.")
        return extracted, notes

    notes.append("Trafilatura unavailable or yielded no content; used bs4 fallback extraction.")
    return _extract_with_bs4(html, raw_text=False), notes


def fetch_web_content(
    url: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    extract_mode: str = "readable",
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    allowed_domains: Optional[List[str]] = None,
) -> Dict[str, Any]:
    normalized_url = str(url or "").strip()
    normalized_domains = _normalize_allowed_domains(allowed_domains)
    normalized_max_chars = _normalize_max_chars(max_chars)
    if not normalized_url:
        return {
            "success": False,
            "backend": "managed_http_fetch",
            "url": "",
            "error": "url must not be empty",
            "content": "",
        }

    validation_error = _validate_public_url(normalized_url, allowed_domains=normalized_domains)
    if validation_error:
        return {
            "success": False,
            "backend": "managed_http_fetch",
            "url": normalized_url,
            "error": validation_error,
            "content": "",
        }

    try:
        response = httpx.get(
            normalized_url,
            timeout=float(timeout_sec or DEFAULT_TIMEOUT_SEC),
            headers={"User-Agent": DEFAULT_USER_AGENT},
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status_code = int(exc.response.status_code) if exc.response is not None else None
        final_url = str(exc.response.url) if exc.response is not None else normalized_url
        return {
            "success": False,
            "backend": "managed_http_fetch",
            "url": normalized_url,
            "final_url": final_url,
            "status_code": status_code,
            "error": f"HTTP fetch failed with status {status_code}: {exc}",
            "content": "",
        }
    except httpx.HTTPError as exc:
        return {
            "success": False,
            "backend": "managed_http_fetch",
            "url": normalized_url,
            "error": f"HTTP fetch failed: {exc}",
            "content": "",
        }
    final_url = str(response.url)
    final_error = _validate_public_url(final_url, allowed_domains=normalized_domains)
    if final_error:
        return {
            "success": False,
            "backend": "managed_http_fetch",
            "url": normalized_url,
            "final_url": final_url,
            "error": f"Final URL validation failed: {final_error}",
            "content": "",
        }

    content_type = str(response.headers.get("content-type") or "").strip().lower()
    title = ""
    description = ""
    notes: List[str] = []

    if _is_html_content_type(content_type) or not content_type:
        html = response.text
        title, description = _extract_metadata_from_html(html)
        content, extraction_notes = _extract_html_content(
            html,
            url=final_url,
            extract_mode=extract_mode,
        )
        notes.extend(extraction_notes)
    elif _is_text_like_content_type(content_type):
        content = response.text
        notes.append("Returned text-like response body directly.")
    else:
        return {
            "success": False,
            "backend": "managed_http_fetch",
            "url": normalized_url,
            "final_url": final_url,
            "status_code": int(response.status_code),
            "content_type": content_type,
            "error": f"Unsupported content type for web_fetch: {content_type or 'unknown'}",
            "content": "",
        }

    truncated_content, truncated = _truncate_content(content, max_chars=normalized_max_chars)
    payload: Dict[str, Any] = {
        "success": True,
        "backend": "managed_http_fetch",
        "url": normalized_url,
        "final_url": final_url,
        "status_code": int(response.status_code),
        "content_type": content_type or "unknown",
        "title": title,
        "description": description,
        "content": truncated_content,
        "truncated": truncated,
        "extract_mode": str(extract_mode or "readable").strip().lower() or "readable",
        "content_length": len(str(content or "")),
        "notes": notes,
    }
    return payload
