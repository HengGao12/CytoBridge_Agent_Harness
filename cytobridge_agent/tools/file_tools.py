"""Lightweight filesystem tools for agent use."""
from __future__ import annotations

import base64
import fnmatch
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..utils.runtime_paths import get_workspace_root


DEFAULT_MAX_ENTRIES = 200
DEFAULT_MAX_CHARS = 20000
READ_FILE_DEFAULT_LIMIT = 2000
READ_FILE_MAX_LINE_LENGTH = 500
READ_TEXT_FAST_PATH_MAX_BYTES = 10 * 1024 * 1024
LIST_DIR_DEFAULT_LIMIT = 25
LIST_DIR_DEFAULT_DEPTH = 2
FIND_DEFAULT_LIMIT = 200
FIND_MAX_LIMIT = 5000
FIND_DEFAULT_TIMEOUT = 15
GREP_DEFAULT_LIMIT = 100
GREP_MAX_LIMIT = 2000
GREP_DEFAULT_TIMEOUT = 15
WORKSPACE_ROOT = get_workspace_root()
DENY_READ_PATHS_ENV = "CYTOBRIDGE_DENY_READ_PATHS"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
IMAGE_ATTACHMENT_MAX_EDGE = 1600
PDF_MAX_PAGES_WITHOUT_SELECTION = 10
PDF_MAX_RENDER_PAGES = 20
PDF_READ_MODES = {"render", "text"}
READ_ARTIFACT_ROOT = (Path.home() / ".cellcompass" / "read_artifacts").resolve()


@dataclass
class ReadResult:
    tool_text: str
    media: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class _HeadTailAccumulator:
    """Track a bounded head/tail view without storing the whole selected range."""

    def __init__(self, max_chars: int):
        self.max_chars = max(0, int(max_chars))
        self.total_chars = 0
        self._head = ""
        self._tail = ""

    def add(self, text: str) -> None:
        if not text:
            return
        self.total_chars += len(text)
        if len(self._head) < self.max_chars:
            remaining = self.max_chars - len(self._head)
            self._head += text[:remaining]
        if self.max_chars > 0:
            self._tail = (self._tail + text)[-self.max_chars :]

    def render(self) -> tuple[str, bool]:
        if self.max_chars <= 0:
            return "", self.total_chars > 0
        if self.total_chars <= self.max_chars:
            return self._head, False
        marker = f"\n...[truncated {self.total_chars - self.max_chars} chars]...\n"
        if len(marker) >= self.max_chars:
            return self._head[: self.max_chars], True
        keep = self.max_chars - len(marker)
        head = keep // 2
        tail = keep - head
        return self._head[:head] + marker + self._tail[-tail:], True


def _is_hidden(path: Path) -> bool:
    return path.name.startswith(".")


def _is_probably_binary(path: Path, sample_size: int = 4096) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(sample_size)
    except Exception:
        return False

    if not chunk:
        return False
    if b"\x00" in chunk:
        return True

    non_text = sum(1 for b in chunk if b < 9 or (13 < b < 32))
    return (non_text / max(len(chunk), 1)) > 0.3


def _safe_resolve_path(path: str) -> tuple[Optional[Path], Optional[str]]:
    try:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve(), None

        direct = candidate.resolve()
        if direct.exists():
            return direct, None

        workspace_relative = (WORKSPACE_ROOT / candidate).resolve()
        return workspace_relative, None
    except Exception as e:
        return None, f"Error: invalid path '{path}': {e}"


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _configured_read_denies() -> list[Path]:
    raw = os.environ.get(DENY_READ_PATHS_ENV, "")
    denied: list[Path] = []
    for item in raw.replace("\n", os.pathsep).split(os.pathsep):
        item = item.strip()
        if not item:
            continue
        try:
            denied.append(Path(item).expanduser().resolve())
        except Exception:
            continue
    return denied


def _read_denied_error(
    target: Path,
    *,
    operation: str,
    block_parent_scans: bool = False,
) -> Optional[str]:
    for denied_root in _configured_read_denies():
        if target == denied_root or _path_is_relative_to(target, denied_root):
            return (
                f"Error: {operation} is blocked by {DENY_READ_PATHS_ENV}: {target}. "
                "Use the explicitly allowed task files instead."
            )
        if block_parent_scans and target.is_dir() and _path_is_relative_to(denied_root, target):
            return (
                f"Error: {operation} would scan a blocked subtree configured by "
                f"{DENY_READ_PATHS_ENV}: {denied_root}. Narrow the path to an allowed directory."
            )
    return None


def _truncate_head_tail(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False

    marker = f"\n...[truncated {len(text) - max_chars} chars]...\n"
    if len(marker) >= max_chars:
        return text[:max_chars], True

    keep = max_chars - len(marker)
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + text[-tail:], True


def _ensure_read_artifact_dir() -> Path:
    READ_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="read_file_", dir=str(READ_ARTIFACT_ROOT)))


def _mime_type_for_path(path: Path) -> str:
    return str(mimetypes.guess_type(path.name)[0] or "application/octet-stream")


def _guess_file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in IMAGE_SUFFIXES or _mime_type_for_path(path).startswith("image/"):
        return "image"
    return "text"


def _encode_data_url(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_mime_type_for_path(path)};base64,{payload}"


def _probe_image_dimensions(path: Path) -> tuple[Optional[int], Optional[int]]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        return int(width), int(height)
    except Exception:
        return None, None


def _render_svg_to_png(path: Path) -> tuple[Optional[Path], Optional[str]]:
    try:
        import cairosvg
    except Exception as exc:  # noqa: BLE001
        return None, f"Error: SVG rendering requires cairosvg: {exc}"

    artifact_dir = _ensure_read_artifact_dir()
    output_path = artifact_dir / f"{path.stem}.png"
    try:
        cairosvg.svg2png(url=str(path), write_to=str(output_path))
    except Exception as exc:  # noqa: BLE001
        return None, f"Error: failed to rasterize SVG '{path}': {exc}"
    return output_path, None


def _prepare_image_attachment(path: Path) -> tuple[Path, str, Optional[int], Optional[int]]:
    """Return a model-friendly image attachment path and a note.

    Large publication figures can be cheap to probe but expensive to attach as
    base64 media because the media message is checkpointed and compacted. Keep
    the source image untouched and attach a bounded visual copy for inspection.
    """
    width, height = _probe_image_dimensions(path)
    if width is None or height is None:
        return path, "", width, height
    max_edge = max(width, height)
    if max_edge <= IMAGE_ATTACHMENT_MAX_EDGE:
        return path, "", width, height

    try:
        from PIL import Image

        artifact_dir = _ensure_read_artifact_dir()
        output_path = artifact_dir / f"{path.stem}.preview.png"
        with Image.open(path) as image:
            preview = image.copy()
            preview.thumbnail((IMAGE_ATTACHMENT_MAX_EDGE, IMAGE_ATTACHMENT_MAX_EDGE))
            if preview.mode not in {"RGB", "RGBA"}:
                preview = preview.convert("RGBA")
            preview.save(output_path, format="PNG", optimize=True)
        note = (
            f"Attached preview was downscaled from {width}x{height} to "
            f"{preview.width}x{preview.height}: {output_path}"
        )
        return output_path, note, width, height
    except Exception as exc:  # noqa: BLE001
        return path, f"Warning: failed to create downscaled preview, attached original image: {exc}", width, height


def _build_media_blocks(paths: list[Path]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for path in paths:
        blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _encode_data_url(path),
                    "detail": "auto",
                },
                "metadata": {
                    "path": str(path),
                    "mime_type": _mime_type_for_path(path),
                },
            }
        )
    return blocks


def _build_image_read_result(target: Path) -> ReadResult | str:
    source_path = target
    attachment_path = target
    render_note = ""
    if target.suffix.lower() == ".svg":
        attachment_path, err = _render_svg_to_png(target)
        if err:
            return err
        assert attachment_path is not None
        render_note = f"Rendered SVG attachment: {attachment_path}"

    source_width, source_height = _probe_image_dimensions(source_path)
    attachment_path, preview_note, width, height = _prepare_image_attachment(attachment_path)
    attachment_width, attachment_height = _probe_image_dimensions(attachment_path)
    size_bytes = source_path.stat().st_size
    attachment_size_bytes = attachment_path.stat().st_size
    summary_lines = [
        f"Path: {target}",
        "Type: image",
        f"MIME: {_mime_type_for_path(target)}",
        f"Attachment MIME: {_mime_type_for_path(attachment_path)}",
        f"Size bytes: {size_bytes}",
        f"Attachment size bytes: {attachment_size_bytes}",
    ]
    if source_width is not None and source_height is not None:
        summary_lines.append(f"Dimensions: {source_width}x{source_height}")
    if attachment_width is not None and attachment_height is not None:
        summary_lines.append(f"Attachment dimensions: {attachment_width}x{attachment_height}")
    if render_note:
        summary_lines.append(render_note)
    if preview_note:
        summary_lines.append(preview_note)
    summary_lines.append("Image content attached for inspection.")
    return ReadResult(
        tool_text="\n".join(summary_lines),
        media=_build_media_blocks([attachment_path]),
        metadata={
            "kind": "image",
            "source_path": str(target),
            "attachment_paths": [str(attachment_path)],
            "mime_type": _mime_type_for_path(target),
            "attachment_mime_type": _mime_type_for_path(attachment_path),
            "width": source_width,
            "height": source_height,
            "attachment_width": attachment_width,
            "attachment_height": attachment_height,
            "size_bytes": size_bytes,
            "attachment_size_bytes": attachment_size_bytes,
            "downscaled_attachment": attachment_path != source_path,
        },
    )


def _get_pdf_page_count(path: Path) -> tuple[Optional[int], Optional[str]]:
    pdfinfo_bin = shutil.which("pdfinfo")
    if pdfinfo_bin:
        try:
            proc = subprocess.run(
                [pdfinfo_bin, str(path)],
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
            if proc.returncode == 0:
                match = re.search(r"^Pages:\s*(\d+)\s*$", proc.stdout, flags=re.MULTILINE)
                if match:
                    return int(match.group(1)), None
        except Exception:
            pass

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return len(reader.pages), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Error: failed to inspect PDF '{path}': {exc}"


def _parse_pdf_pages(raw_pages: str, page_count: int) -> tuple[Optional[list[int]], Optional[str]]:
    selected: list[int] = []
    seen = set()
    for token in [part.strip() for part in raw_pages.split(",") if part.strip()]:
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            try:
                start = int(start_raw)
                end = int(end_raw)
            except ValueError:
                return None, f"Error: invalid page range '{token}'."
            if start < 1 or end < start:
                return None, f"Error: invalid page range '{token}'."
            for page in range(start, end + 1):
                if page > page_count:
                    return None, f"Error: page {page} exceeds PDF page count {page_count}."
                if page not in seen:
                    selected.append(page)
                    seen.add(page)
            continue
        try:
            page = int(token)
        except ValueError:
            return None, f"Error: invalid page token '{token}'."
        if page < 1 or page > page_count:
            return None, f"Error: page {page} exceeds PDF page count {page_count}."
        if page not in seen:
            selected.append(page)
            seen.add(page)
    if not selected:
        return None, "Error: pages must contain at least one page number."
    return selected, None


def _render_pdf_pages(path: Path, pages: list[int]) -> tuple[Optional[list[Path]], Optional[str]]:
    pdftoppm_bin = shutil.which("pdftoppm")
    if not pdftoppm_bin:
        return None, "Error: pdftoppm is required to read PDF pages visually."

    artifact_dir = _ensure_read_artifact_dir()
    rendered_paths: list[Path] = []
    for page in pages:
        output_prefix = artifact_dir / f"{path.stem}_page_{page:03d}"
        command = [
            pdftoppm_bin,
            "-singlefile",
            "-f",
            str(page),
            "-l",
            str(page),
            "-r",
            "144",
            "-png",
            str(path),
            str(output_prefix),
        ]
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except Exception as exc:  # noqa: BLE001
            return None, f"Error: failed to render PDF page {page} from '{path}': {exc}"
        if proc.returncode != 0:
            stderr = (proc.stderr or proc.stdout or "").strip()
            return None, f"Error: pdftoppm failed for page {page} of '{path}': {stderr}"
        rendered_path = output_prefix.with_suffix(".png")
        if not rendered_path.exists():
            return None, f"Error: expected rendered PDF page not found: {rendered_path}"
        rendered_paths.append(rendered_path)
    return rendered_paths, None


def _select_pdf_pages(target: Path, pages: Optional[str]) -> tuple[Optional[int], Optional[list[int]], Optional[str]]:
    page_count, err = _get_pdf_page_count(target)
    if err:
        return None, None, err
    assert page_count is not None

    if pages:
        selected_pages, parse_err = _parse_pdf_pages(str(pages), page_count)
        if parse_err:
            return None, None, parse_err
        assert selected_pages is not None
    else:
        if page_count > PDF_MAX_PAGES_WITHOUT_SELECTION:
            return None, None, (
                f"Error: PDF has {page_count} pages. "
                f"Please specify `pages`, for example `pages=\"1-3\"`."
            )
        selected_pages = list(range(1, page_count + 1))

    if len(selected_pages) > PDF_MAX_RENDER_PAGES:
        return None, None, (
            f"Error: requested {len(selected_pages)} PDF pages, but the maximum per read is "
            f"{PDF_MAX_RENDER_PAGES}. Narrow the selection and retry."
        )
    return page_count, selected_pages, None


def _build_pdf_read_result(target: Path, pages: Optional[str]) -> ReadResult | str:
    page_count, selected_pages, selection_err = _select_pdf_pages(target, pages)
    if selection_err:
        return selection_err
    assert page_count is not None
    assert selected_pages is not None

    rendered_paths, render_err = _render_pdf_pages(target, selected_pages)
    if render_err:
        return render_err
    assert rendered_paths is not None

    summary_lines = [
        f"Path: {target}",
        "Type: pdf",
        f"Total pages: {page_count}",
        f"Selected pages: {', '.join(str(page) for page in selected_pages)}",
        f"Rendered attachments: {len(rendered_paths)}",
        "Rendered page images attached for inspection.",
    ]
    for rendered_path in rendered_paths[:5]:
        summary_lines.append(f"- {rendered_path}")
    if len(rendered_paths) > 5:
        summary_lines.append(f"... and {len(rendered_paths) - 5} more rendered page(s).")

    return ReadResult(
        tool_text="\n".join(summary_lines),
        media=_build_media_blocks(rendered_paths),
        metadata={
            "kind": "pdf",
            "mode": "render",
            "source_path": str(target),
            "page_count": page_count,
            "selected_pages": selected_pages,
            "attachment_paths": [str(path) for path in rendered_paths],
        },
    )


def _read_pdf_page_text_with_pdfplumber(path: Path, pages: list[int]) -> tuple[Optional[list[str]], Optional[str]]:
    try:
        import pdfplumber
    except Exception as exc:  # noqa: BLE001
        return None, f"pdfplumber unavailable: {exc}"

    try:
        texts: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page_number in pages:
                page = pdf.pages[page_number - 1]
                texts.append(page.extract_text() or "")
        return texts, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _read_pdf_page_text_with_pypdf(path: Path, pages: list[int]) -> tuple[Optional[list[str]], Optional[str]]:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # noqa: BLE001
        return None, f"pypdf unavailable: {exc}"

    try:
        reader = PdfReader(str(path))
        texts: list[str] = []
        for page_number in pages:
            texts.append(reader.pages[page_number - 1].extract_text() or "")
        return texts, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _format_virtual_text_read(
    source_label: str,
    lines: list[str],
    offset: int,
    limit: int,
    extra_headers: Optional[list[str]] = None,
) -> str:
    if offset < 1:
        return "Error: offset must be a 1-indexed line number."
    if limit <= 0:
        return "Error: limit must be greater than zero."
    if not lines:
        lines = ["[No extractable text found.]"]
    if offset > len(lines):
        return "Error: offset exceeds extracted content length."

    end_line = min(len(lines), offset + limit - 1)
    selected = lines[offset - 1 : end_line]
    numbered = []
    for line_number, line in enumerate(selected, start=offset):
        text = line.rstrip("\r\n")
        if len(text) > READ_FILE_MAX_LINE_LENGTH:
            text = text[:READ_FILE_MAX_LINE_LENGTH]
        numbered.append(f"L{line_number}: {text}")

    content = "\n".join(numbered)
    content, truncated = _truncate_head_tail(content, DEFAULT_MAX_CHARS)

    header = [
        f"Path: {source_label}",
        f"Line range: {offset}-{end_line}",
        f"Total lines: {len(lines)}",
        f"Returned chars: {len(content)}",
        f"Truncated: {str(truncated).lower()}",
    ]
    if extra_headers:
        header.extend(extra_headers)
    return "\n".join(header + ["--- BEGIN CONTENT ---", content, "--- END CONTENT ---"])


def _normalize_text_read_lines(raw: str) -> list[str]:
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    return raw.splitlines()


def _finalize_text_read_result(
    target: Path,
    start_line: int,
    total_lines: int,
    end_line_used: int,
    accumulator: _HeadTailAccumulator,
) -> str:
    content, truncated = accumulator.render()
    header = [
        f"Path: {target}",
        f"Line range: {start_line}-{end_line_used}",
        f"Total lines: {total_lines}",
        f"Returned chars: {len(content)}",
        f"Truncated: {str(truncated).lower()}",
    ]
    return "\n".join(header + ["--- BEGIN CONTENT ---", content, "--- END CONTENT ---"])


def _read_text_file_fast(
    target: Path,
    start_line: int,
    end_line: Optional[int],
    max_chars: int,
    encoding: str,
) -> str:
    try:
        raw = target.read_text(encoding=encoding, errors="replace")
    except Exception as e:
        return f"Error: failed to read file '{target}': {e}"

    lines = _normalize_text_read_lines(raw)
    total_lines = len(lines)
    if total_lines < start_line:
        return "Error: start_line exceeds file length."

    stop_line = min(total_lines, end_line if end_line is not None else total_lines)
    accumulator = _HeadTailAccumulator(max_chars)
    for line_number in range(start_line, stop_line + 1):
        text = lines[line_number - 1].rstrip("\r\n")
        if len(text) > READ_FILE_MAX_LINE_LENGTH:
            text = text[:READ_FILE_MAX_LINE_LENGTH]
        if line_number > start_line:
            accumulator.add("\n")
        accumulator.add(f"L{line_number}: {text}")

    return _finalize_text_read_result(
        target=target,
        start_line=start_line,
        total_lines=total_lines,
        end_line_used=stop_line,
        accumulator=accumulator,
    )


def _read_text_file_streaming(
    target: Path,
    start_line: int,
    end_line: Optional[int],
    max_chars: int,
    encoding: str,
) -> str:
    accumulator = _HeadTailAccumulator(max_chars)
    scanned_total_lines = 0
    last_selected_line = start_line
    selected_any = False

    try:
        with target.open("r", encoding=encoding, errors="replace") as f:
            for line_number, line in enumerate(f, start=1):
                scanned_total_lines = line_number
                if line_number < start_line:
                    continue
                if end_line is not None and line_number > end_line:
                    scanned_total_lines = end_line
                    break
                text = line.rstrip("\r\n")
                if len(text) > READ_FILE_MAX_LINE_LENGTH:
                    text = text[:READ_FILE_MAX_LINE_LENGTH]
                if selected_any:
                    accumulator.add("\n")
                accumulator.add(f"L{line_number}: {text}")
                selected_any = True
                last_selected_line = line_number
    except Exception as e:
        return f"Error: failed to read file '{target}': {e}"

    if scanned_total_lines < start_line:
        return "Error: start_line exceeds file length."

    if not selected_any:
        last_selected_line = start_line

    return _finalize_text_read_result(
        target=target,
        start_line=start_line,
        total_lines=scanned_total_lines,
        end_line_used=last_selected_line,
        accumulator=accumulator,
    )


def _build_pdf_text_read_result(
    target: Path,
    pages: Optional[str],
    offset: int,
    limit: int,
) -> str:
    page_count, selected_pages, selection_err = _select_pdf_pages(target, pages)
    if selection_err:
        return selection_err
    assert page_count is not None
    assert selected_pages is not None

    extracted_pages, extract_err = _read_pdf_page_text_with_pdfplumber(target, selected_pages)
    backend = "pdfplumber"
    if extract_err:
        extracted_pages, fallback_err = _read_pdf_page_text_with_pypdf(target, selected_pages)
        backend = "pypdf"
        if fallback_err:
            return (
                f"Error: failed to extract PDF text from '{target}'. "
                f"pdfplumber: {extract_err}; pypdf: {fallback_err}"
            )
    assert extracted_pages is not None

    lines: list[str] = []
    for page_number, page_text in zip(selected_pages, extracted_pages):
        lines.append(f"=== Page {page_number} ===")
        page_lines = [segment.rstrip("\r\n") for segment in (page_text or "").splitlines()]
        if page_lines:
            lines.extend(page_lines)
        else:
            lines.append("[No extractable text on this page.]")
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()

    return _format_virtual_text_read(
        source_label=str(target),
        lines=lines,
        offset=offset,
        limit=limit,
        extra_headers=[
            "Type: pdf",
            "PDF mode: text",
            f"PDF text backend: {backend}",
            f"Total pages: {page_count}",
            f"Selected pages: {', '.join(str(page) for page in selected_pages)}",
        ],
    )


def list_path(
    path: str = ".",
    recursive: bool = False,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    show_hidden: bool = False,
) -> str:
    """List a file or directory in a compact human-readable format."""
    target, err = _safe_resolve_path(path)
    if err:
        return err
    assert target is not None
    denied = _read_denied_error(
        target,
        operation="list_path",
        block_parent_scans=bool(recursive),
    )
    if denied:
        return denied

    if not target.exists():
        return f"Error: path does not exist: {target}"

    if target.is_file():
        try:
            stat = target.stat()
        except Exception as e:
            return f"Error: failed to stat file '{target}': {e}"
        return "\n".join(
            [
                f"Path: {target}",
                "Type: file",
                f"Name: {target.name}",
                f"Size bytes: {stat.st_size}",
                f"Suffix: {target.suffix}",
            ]
        )

    depth = 64 if recursive else LIST_DIR_DEFAULT_DEPTH
    return list_dir(
        dir_path=str(target),
        offset=1,
        limit=max_entries,
        depth=depth,
        show_hidden=show_hidden,
    )


def list_dir(
    dir_path: str,
    offset: int = 1,
    limit: int = LIST_DIR_DEFAULT_LIMIT,
    depth: int = LIST_DIR_DEFAULT_DEPTH,
    show_hidden: bool = False,
) -> str:
    """List entries with codex-style pagination/depth semantics."""
    target, err = _safe_resolve_path(dir_path)
    if err:
        return err
    assert target is not None
    denied = _read_denied_error(
        target,
        operation="list_dir",
        block_parent_scans=depth > 1,
    )
    if denied:
        return denied

    if not target.exists():
        return f"Error: path does not exist: {target}"
    if not target.is_dir():
        return f"Error: not a directory: {target}"
    if offset < 1:
        return "Error: offset must be a 1-indexed entry number."
    if limit <= 0:
        return "Error: limit must be greater than zero."
    if depth <= 0:
        return "Error: depth must be greater than zero."

    rows: list[tuple[str, str]] = []
    queue: deque[tuple[Path, Path, int]] = deque([(target, Path("."), depth)])

    while queue:
        current, rel_prefix, remain = queue.popleft()
        try:
            children = list(current.iterdir())
        except Exception as e:
            return f"Error: failed to read directory '{current}': {e}"

        children.sort(key=lambda p: p.name.lower())
        for child in children:
            if not show_hidden and _is_hidden(child):
                continue

            child_rel = child.relative_to(target)
            entry_name = child.name
            kind_suffix = "/" if child.is_dir() else "@"
            if child.is_file():
                kind_suffix = ""
            elif not child.is_dir() and not child.is_symlink():
                kind_suffix = "?"
            indent = "  " * (len(child_rel.parts) - 1)
            rows.append((str(child_rel).replace("\\", "/"), f"{indent}{entry_name}{kind_suffix}"))

            if child.is_dir() and remain > 1:
                queue.append((child, rel_prefix / child.name, remain - 1))

    rows.sort(key=lambda x: x[0])
    start_idx = offset - 1
    if start_idx >= len(rows) and rows:
        return "Error: offset exceeds directory entry count."
    if not rows:
        return f"Absolute path: {target}"

    selected = rows[start_idx : start_idx + limit]
    lines = [f"Absolute path: {target}"] + [line for _, line in selected]
    if start_idx + limit < len(rows):
        lines.append(f"More than {limit} entries found")
    return "\n".join(lines)


def read_text_file(
    path: str,
    start_line: int = 1,
    end_line: Optional[int] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    encoding: str = "utf-8",
) -> str:
    """Read text file snippets with line numbers and safe truncation."""
    target, err = _safe_resolve_path(path)
    if err:
        return err
    assert target is not None
    denied = _read_denied_error(target, operation="read_text_file")
    if denied:
        return denied

    if not target.exists():
        return f"Error: file does not exist: {target}"
    if not target.is_file():
        return f"Error: not a file: {target}"
    if _is_probably_binary(target):
        return f"Error: file appears to be binary and cannot be read as text: {target}"

    if start_line < 1:
        start_line = 1
    if end_line is not None and end_line < start_line:
        return "Error: end_line must be >= start_line."
    try:
        size_bytes = target.stat().st_size
    except Exception as e:
        return f"Error: failed to stat file '{target}': {e}"

    if size_bytes <= READ_TEXT_FAST_PATH_MAX_BYTES:
        return _read_text_file_fast(
            target=target,
            start_line=start_line,
            end_line=end_line,
            max_chars=max_chars,
            encoding=encoding,
        )

    return _read_text_file_streaming(
        target=target,
        start_line=start_line,
        end_line=end_line,
        max_chars=max_chars,
        encoding=encoding,
    )


def read_file(
    file_path: str,
    offset: int = 1,
    limit: Optional[int] = None,
    pages: Optional[str] = None,
    pdf_mode: str = "render",
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    default_text_limit: Optional[int] = READ_FILE_DEFAULT_LIMIT,
) -> str | ReadResult:
    """Unified file reader for text, images, and PDFs."""
    target, err = _safe_resolve_path(file_path)
    if err:
        return err
    assert target is not None
    denied = _read_denied_error(target, operation="read_file")
    if denied:
        return denied

    if not target.exists():
        return f"Error: file does not exist: {target}"
    if not target.is_file():
        return f"Error: not a file: {target}"
    if offset < 1:
        return "Error: offset must be a 1-indexed line number."
    effective_limit = limit
    if effective_limit is None:
        effective_limit = default_text_limit
    if effective_limit is not None and effective_limit <= 0:
        return "Error: limit must be greater than zero."
    if pages and target.suffix.lower() != ".pdf":
        return "Error: `pages` is only supported for PDF files."
    pdf_mode = str(pdf_mode or "render").strip().lower()
    if pdf_mode not in PDF_READ_MODES:
        return f"Error: unsupported pdf_mode '{pdf_mode}'. Use one of: render, text."
    if pdf_mode != "render" and target.suffix.lower() != ".pdf":
        return "Error: `pdf_mode` is only supported for PDF files."

    kind = _guess_file_kind(target)
    if kind == "pdf":
        if pdf_mode == "text":
            pdf_limit = effective_limit if effective_limit is not None else READ_FILE_DEFAULT_LIMIT
            return _build_pdf_text_read_result(target, pages=pages, offset=offset, limit=int(pdf_limit))
        return _build_pdf_read_result(target, pages=pages)
    if kind == "image":
        return _build_image_read_result(target)
    if _is_probably_binary(target):
        return f"Error: unsupported binary file; use a specialized tool instead: {target}"

    end_line = None if effective_limit is None else offset + int(effective_limit) - 1
    return read_text_file(
        path=str(target),
        start_line=offset,
        end_line=end_line,
        max_chars=max_chars,
        encoding="utf-8",
    )


def find_files(
    pattern: str = "*",
    path: Optional[str] = None,
    limit: int = FIND_DEFAULT_LIMIT,
    recursive: bool = True,
    show_hidden: bool = False,
    timeout: int = FIND_DEFAULT_TIMEOUT,
) -> str:
    """Find files by glob pattern; returns one path per line."""
    pattern = (pattern or "*").strip() or "*"
    if limit <= 0:
        return "Error: limit must be greater than zero."
    if timeout <= 0:
        return "Error: timeout must be greater than zero."
    limit = min(limit, FIND_MAX_LIMIT)
    timeout = max(1, int(timeout))

    search_root = path or "."
    target, err = _safe_resolve_path(search_root)
    if err:
        return err
    assert target is not None
    denied = _read_denied_error(
        target,
        operation="find_files",
        block_parent_scans=True,
    )
    if denied:
        return denied
    if not target.exists():
        return f"Error: path does not exist: {target}"

    if target.is_file():
        matched = fnmatch.fnmatch(target.name, pattern) or fnmatch.fnmatch(str(target), pattern)
        return str(target) if matched else "No matches found."

    rg_bin = shutil.which("rg")
    if rg_bin:
        command = [rg_bin, "--files", "--no-messages"]
        if show_hidden:
            command.append("--hidden")
        if not recursive:
            command.extend(["--max-depth", "1"])
        if pattern != "*":
            command.extend(["--glob", pattern])
        command.extend(["--", str(target)])
        try:
            proc = subprocess.run(
                command,
                cwd=str(Path.cwd()),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"⏳ find_files timed out after {timeout}s while scanning '{target}'. Narrow the path or pattern and retry."
        except Exception as e:
            return f"Error: failed to run rg: {e}"

        if proc.returncode == 0:
            matched = [line for line in proc.stdout.splitlines() if line.strip()]
            matched = matched[:limit]
            if not matched:
                return "No matches found."
            if len(matched) == limit:
                matched.append(f"... truncated to {limit} entries ...")
            return "\n".join(matched)
        if proc.returncode == 1:
            return "No matches found."
        return f"Error: rg failed: {proc.stderr.strip()}"

    if recursive:
        iterator = target.rglob("*")
    else:
        iterator = target.glob("*")

    matches: list[str] = []
    deadline = time.monotonic() + float(timeout)
    for candidate in iterator:
        if time.monotonic() > deadline:
            if matches:
                matches.append(
                    f"... timed out after {timeout}s; returning {len(matches)} partial match(es) ..."
                )
                return "\n".join(matches)
            return f"⏳ find_files timed out after {timeout}s while scanning '{target}'. Narrow the path or pattern and retry."
        if not candidate.is_file():
            continue
        if not show_hidden and any(part.startswith(".") for part in candidate.relative_to(target).parts):
            continue

        rel = candidate.relative_to(target).as_posix()
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(candidate.name, pattern):
            matches.append(str(candidate))
            if len(matches) >= limit:
                break

    return "\n".join(matches) if matches else "No matches found."


def grep_files(
    pattern: str,
    include: Optional[str] = None,
    path: Optional[str] = None,
    limit: int = GREP_DEFAULT_LIMIT,
    case_sensitive: bool = False,
    timeout: int = GREP_DEFAULT_TIMEOUT,
) -> str:
    """Search file content and return `path:line:snippet` matches."""
    pattern = (pattern or "").strip()
    if not pattern:
        return "Error: pattern must not be empty."
    if limit <= 0:
        return "Error: limit must be greater than zero."
    if timeout <= 0:
        return "Error: timeout must be greater than zero."
    limit = min(limit, GREP_MAX_LIMIT)
    timeout = max(1, int(timeout))

    search_root = path or "."
    target, err = _safe_resolve_path(search_root)
    if err:
        return err
    assert target is not None
    denied = _read_denied_error(
        target,
        operation="grep_files",
        block_parent_scans=True,
    )
    if denied:
        return denied
    if not target.exists():
        return f"Error: path does not exist: {target}"

    rg_bin = shutil.which("rg")
    if rg_bin:
        command = [
            rg_bin,
            "--line-number",
            "--no-heading",
            "--color",
            "never",
            "--max-columns",
            str(READ_FILE_MAX_LINE_LENGTH),
            "--regexp",
            pattern,
            "--no-messages",
        ]
        if not case_sensitive:
            command.append("--ignore-case")
        if include:
            command.extend(["--glob", include])
        command.extend(["--", str(target)])
        try:
            proc = subprocess.run(
                command,
                cwd=str(Path.cwd()),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"⏳ grep_files timed out after {timeout}s while scanning '{target}'. Narrow the path, include pattern, or regex and retry."
        except Exception as e:
            return f"Error: failed to run rg: {e}"

        if proc.returncode == 0:
            matched = [line for line in proc.stdout.splitlines() if line.strip()]
            matched = matched[:limit]
            if not matched:
                return "No matches found."
            if len(proc.stdout.splitlines()) > limit:
                matched.append(f"... truncated to {limit} matches ...")
            return "\n".join(matched)
        if proc.returncode == 1:
            return "No matches found."
        return f"Error: rg failed: {proc.stderr.strip()}"

    # Fallback: Python file scan.
    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags=flags)
    except re.error:
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(re.escape(pattern), flags=flags)

    matched_lines: list[str] = []
    deadline = time.monotonic() + float(timeout)

    def _timeout_message() -> str:
        if matched_lines:
            matched_lines.append(
                f"... timed out after {timeout}s; returning {len(matched_lines)} partial match(es) ..."
            )
            return "\n".join(matched_lines)
        return f"⏳ grep_files timed out after {timeout}s while scanning '{target}'. Narrow the path, include pattern, or regex and retry."

    def _iter_candidate_files():
        if target.is_file():
            yield target
            return
        for root, dirs, files in os.walk(target):
            if time.monotonic() > deadline:
                raise TimeoutError
            dirs.sort()
            files.sort()
            root_path = Path(root)
            for name in files:
                if time.monotonic() > deadline:
                    raise TimeoutError
                yield root_path / name

    try:
        candidates = _iter_candidate_files()
        for p in candidates:
            if time.monotonic() > deadline:
                return _timeout_message()
            if include and not fnmatch.fnmatch(p.name, include):
                continue
            if _is_probably_binary(p):
                continue
            try:
                with p.open("r", encoding="utf-8", errors="replace") as f:
                    for lineno, raw in enumerate(f, start=1):
                        if time.monotonic() > deadline:
                            return _timeout_message()
                        text = raw.rstrip("\r\n")
                        if len(text) > READ_FILE_MAX_LINE_LENGTH:
                            text = text[:READ_FILE_MAX_LINE_LENGTH]
                        if regex.search(text):
                            matched_lines.append(f"{p}:{lineno}:{text}")
                            if len(matched_lines) >= limit:
                                break
            except Exception:
                continue
            if len(matched_lines) >= limit:
                break
    except TimeoutError:
        return _timeout_message()

    return "\n".join(matched_lines) if matched_lines else "No matches found."
