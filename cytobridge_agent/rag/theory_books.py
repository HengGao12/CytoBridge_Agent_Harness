from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import DIRS, RAG_DIR, get_sentence_transformer_model

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1400
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
DEFAULT_THEORY_LITERATURE_FILENAMES = (
    "Unbalanced Optimal Transport Dynamic and Kantorovich Formulation.pdf",
)


def _theory_base_dir() -> Path:
    override = os.getenv("CYTOBRIDGE_THEORY_BOOKS_INDEX_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return RAG_DIR / "theory_books"


def _source_books_dir() -> Path:
    override = os.getenv("CYTOBRIDGE_THEORY_BOOKS_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    bundled_dir = RAG_DIR / "theory_books" / "pdfs"
    if bundled_dir.exists():
        return bundled_dir
    return Path.home() / "books"


def _default_theory_literature_pdf_paths() -> List[Path]:
    literature_dir = DIRS.get("literature")
    if not literature_dir:
        return []
    return [Path(literature_dir) / filename for filename in DEFAULT_THEORY_LITERATURE_FILENAMES]


def _extra_theory_pdf_paths_from_env() -> List[Path]:
    raw = os.getenv("CYTOBRIDGE_THEORY_EXTRA_PDFS", "").strip()
    if not raw:
        return []
    paths: List[Path] = []
    for item in re.split(r"[:;,]", raw):
        candidate = item.strip()
        if candidate:
            paths.append(Path(candidate).expanduser())
    return paths


def _safe_id(value: str) -> str:
    stem = Path(str(value or "book")).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem or "book"


def _clean_text(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_page_text(text: str, *, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [{"text": cleaned, "char_start": 0, "char_end": len(cleaned)}]

    chunks: List[Dict[str, Any]] = []
    start = 0
    step = max(1, chunk_size - max(0, chunk_overlap))
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        if end < len(cleaned):
            boundary = max(cleaned.rfind("\n", start, end), cleaned.rfind(". ", start, end))
            if boundary > start + int(chunk_size * 0.45):
                end = boundary + (1 if cleaned[boundary : boundary + 1] == "\n" else 2)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append({"text": piece, "char_start": start, "char_end": end})
        if end >= len(cleaned):
            break
        start = max(end - chunk_overlap, start + step)
    return chunks


def _extract_pdf_pages(pdf_path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metadata: Dict[str, Any] = {}
    try:
        import pdfplumber

        pages: List[Dict[str, Any]] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            metadata = dict(pdf.metadata or {})
            for idx, page in enumerate(pdf.pages, start=1):
                pages.append({"page": idx, "text": page.extract_text() or ""})
        return pages, metadata
    except Exception as exc:
        logger.debug("pdfplumber extraction failed for %s: %s", pdf_path, exc)

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        metadata = {str(k).lstrip("/"): str(v) for k, v in dict(reader.metadata or {}).items()}
        pages = [
            {"page": idx, "text": page.extract_text() or ""}
            for idx, page in enumerate(reader.pages, start=1)
        ]
        return pages, metadata
    except Exception as exc:
        raise RuntimeError(f"failed to extract PDF text from {pdf_path}: {exc}") from exc


def _guess_section(text: str) -> str:
    for line in str(text or "").splitlines()[:8]:
        candidate = line.strip()
        if not candidate or len(candidate) > 120:
            continue
        if re.match(r"^(chapter|section|appendix)\b", candidate, re.IGNORECASE):
            return candidate
        if re.match(r"^\d+(\.\d+)*\s+\S", candidate):
            return candidate
    return ""


class TheoryBookIndex:
    """Separate dense index for long-form mathematical theory sources."""

    def __init__(
        self,
        *,
        books_dir: Optional[Path] = None,
        base_dir: Optional[Path] = None,
        extra_pdf_paths: Optional[List[Path]] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.books_dir = Path(books_dir).expanduser() if books_dir else _source_books_dir()
        self.base_dir = Path(base_dir).expanduser() if base_dir else _theory_base_dir()
        self.extra_pdf_paths = [
            Path(path).expanduser()
            for path in (
                extra_pdf_paths
                if extra_pdf_paths is not None
                else [*_default_theory_literature_pdf_paths(), *_extra_theory_pdf_paths_from_env()]
            )
        ]
        self.index_file = self.base_dir / "book_index.json"
        self.catalog_file = self.base_dir / "book_catalog.json"
        self.embedding_dir = self.base_dir / "book_embeddings"
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self._index: Optional[Dict[str, Any]] = None
        self._model: Any = None
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

    def _source_pdf_paths(self) -> List[Path]:
        paths: List[Path] = []
        if self.books_dir.exists():
            paths.extend(sorted(self.books_dir.glob("*.pdf"), key=lambda path: path.name.lower()))
        paths.extend(path for path in self.extra_pdf_paths if path.exists() and path.suffix.lower() == ".pdf")

        deduped: Dict[str, Path] = {}
        for path in paths:
            try:
                key = str(path.resolve())
            except Exception:
                key = str(path)
            deduped[key] = path
        return sorted(deduped.values(), key=lambda path: (path.parent.name.lower(), path.name.lower()))

    def _source_pdf_lookup(self) -> Dict[str, Path]:
        lookup: Dict[str, Path] = {}
        for path in self._source_pdf_paths():
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            keys = {
                path.name.lower(),
                path.stem.lower(),
                _safe_id(path.name),
            }
            for key in keys:
                if key:
                    lookup.setdefault(key, resolved)
        return lookup

    def _resolve_index_pdf_path(self, info: Dict[str, Any], source_lookup: Dict[str, Path]) -> Optional[Path]:
        raw_path = str(info.get("pdf_path") or "").strip()
        if raw_path:
            candidate = Path(raw_path).expanduser()
            if candidate.exists():
                try:
                    return candidate.resolve()
                except Exception:
                    return candidate

        keys = [
            str(info.get("pdf_filename") or "").strip(),
            Path(raw_path).name if raw_path else "",
            str(info.get("book_id") or "").strip(),
        ]
        for key in keys:
            if not key:
                continue
            for normalized in {key.lower(), Path(key).stem.lower(), _safe_id(key)}:
                resolved = source_lookup.get(normalized)
                if resolved and resolved.exists():
                    return resolved
        return None

    def _load_index(self) -> Dict[str, Any]:
        if self._index is not None:
            return self._index
        if self.index_file.exists():
            try:
                self._index = json.loads(self.index_file.read_text(encoding="utf-8"))
                return self._index
            except Exception as exc:
                logger.warning("Failed to load theory book index, rebuilding on demand: %s", exc)
        self._index = {"schema_version": 1, "books": {}}
        return self._index

    def _save_index(self, index: Dict[str, Any]) -> None:
        self.index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        catalog = {
            "books_dir": str(self.books_dir),
            "extra_pdf_paths": [str(path) for path in self.extra_pdf_paths],
            "books": [
                {
                    "book_id": book_id,
                    "title": info.get("title", ""),
                    "pdf_path": info.get("pdf_path", ""),
                    "num_pages": info.get("num_pages", 0),
                    "num_chunks": len(info.get("chunks", []) or []),
                }
                for book_id, info in sorted((index.get("books") or {}).items())
            ],
        }
        self.catalog_file.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    def _model_encode(self, texts: List[str]) -> np.ndarray:
        if self._model is None:
            self._model = get_sentence_transformer_model()
        return np.asarray(self._model.encode(texts, show_progress_bar=False), dtype=np.float32)

    def build(self, *, force_rebuild: bool = False) -> Dict[str, Any]:
        index = {"schema_version": 1, "books": {}} if force_rebuild else self._load_index()
        books = dict(index.get("books") or {})
        pdfs = self._source_pdf_paths()
        built = 0
        skipped = 0
        failed = 0

        for pdf_path in pdfs:
            book_id = _safe_id(pdf_path.name)
            existing = books.get(book_id)
            if existing and not force_rebuild:
                emb_path = self.base_dir / str(existing.get("embeddings_file") or "")
                if emb_path.exists() and existing.get("pdf_mtime") == pdf_path.stat().st_mtime:
                    skipped += 1
                    continue

            try:
                pages, metadata = _extract_pdf_pages(pdf_path)
                title = str(metadata.get("Title") or metadata.get("title") or pdf_path.stem).strip()
                chunks: List[Dict[str, Any]] = []
                for page in pages:
                    page_no = int(page.get("page") or 0)
                    for local_idx, chunk in enumerate(
                        _split_page_text(
                            str(page.get("text") or ""),
                            chunk_size=self.chunk_size,
                            chunk_overlap=self.chunk_overlap,
                        ),
                        start=1,
                    ):
                        text = str(chunk.get("text") or "")
                        chunks.append(
                            {
                                "chunk_id": f"{book_id}:p{page_no}:c{local_idx}",
                                "book_id": book_id,
                                "title": title,
                                "pdf_path": str(pdf_path.resolve()),
                                "page": page_no,
                                "page_range": [max(1, page_no - 1), min(len(pages), page_no + 1)],
                                "section": _guess_section(text),
                                "text": text,
                                "char_start": chunk.get("char_start"),
                                "char_end": chunk.get("char_end"),
                            }
                        )
                if not chunks:
                    raise RuntimeError("no text chunks extracted")
                embeddings = self._model_encode([chunk["text"] for chunk in chunks])
                embeddings_file = self.embedding_dir / f"{book_id}_embeddings.npy"
                np.save(embeddings_file, embeddings)
                books[book_id] = {
                    "book_id": book_id,
                    "title": title,
                    "pdf_path": str(pdf_path.resolve()),
                    "pdf_filename": pdf_path.name,
                    "pdf_mtime": pdf_path.stat().st_mtime,
                    "num_pages": len(pages),
                    "chunks": chunks,
                    "embeddings_file": str(embeddings_file.relative_to(self.base_dir)),
                    "embedding_dim": int(embeddings.shape[1]) if len(embeddings.shape) > 1 else 0,
                    "last_updated": datetime.now().isoformat(),
                }
                built += 1
            except Exception as exc:
                failed += 1
                logger.exception("Failed to index theory book %s: %s", pdf_path, exc)

        index = {"schema_version": 1, "books": books}
        self._index = index
        self._save_index(index)
        return {
            "status": "completed",
            "books_dir": str(self.books_dir),
            "extra_pdf_paths": [str(path) for path in self.extra_pdf_paths],
            "indexed_books": len(books),
            "built": built,
            "skipped": skipped,
            "failed": failed,
            "index_file": str(self.index_file),
        }

    def ensure_ready(self) -> Dict[str, Any]:
        index = self._load_index()
        books = dict(index.get("books") or {})
        pdfs = self._source_pdf_paths()
        if not pdfs:
            self._save_index(index)
            return {
                "status": "empty",
                "message": f"No PDF theory sources found in {self.books_dir} or extra_pdf_paths",
                "indexed_books": len(books),
            }
        missing_or_stale = []
        for pdf_path in pdfs:
            book_id = _safe_id(pdf_path.name)
            info = books.get(book_id)
            emb_path = self.base_dir / str((info or {}).get("embeddings_file") or "")
            if not info or not emb_path.exists() or info.get("pdf_mtime") != pdf_path.stat().st_mtime:
                missing_or_stale.append(pdf_path.name)
        if missing_or_stale:
            return self.build(force_rebuild=False)
        return {"status": "ready", "indexed_books": len(books), "index_file": str(self.index_file)}

    def search(self, query: str, *, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        query_text = str(query or "").strip()
        if not query_text:
            return {"query": query_text, "results": [], "error": "query is required"}
        ready = self.ensure_ready()
        index = self._load_index()
        source_lookup = self._source_pdf_lookup()
        candidates: List[Dict[str, Any]] = []
        embedding_blocks: List[np.ndarray] = []
        skipped_missing_pdf_books: List[str] = []
        for info in (index.get("books") or {}).values():
            chunks = list(info.get("chunks") or [])
            emb_path = self.base_dir / str(info.get("embeddings_file") or "")
            if not chunks or not emb_path.exists():
                continue
            resolved_pdf_path = self._resolve_index_pdf_path(dict(info), source_lookup)
            if resolved_pdf_path is None:
                skipped_missing_pdf_books.append(str(info.get("book_id") or info.get("title") or "unknown"))
                continue
            embeddings = np.load(emb_path)
            if len(embeddings) != len(chunks):
                continue
            for chunk in chunks:
                candidate = dict(chunk)
                candidate["pdf_path"] = str(resolved_pdf_path)
                candidate["pdf_filename"] = resolved_pdf_path.name
                candidates.append(candidate)
            embedding_blocks.append(np.asarray(embeddings, dtype=np.float32))
        if not candidates or not embedding_blocks:
            return {
                "query": query_text,
                "results": [],
                "index_status": ready,
                "skipped_missing_pdf_books": skipped_missing_pdf_books,
            }

        matrix = np.vstack(embedding_blocks)
        query_emb = self._model_encode([query_text])[0]
        scores = matrix @ query_emb / ((np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_emb)) + 1e-9)
        top_indices = np.argsort(scores)[::-1][: max(1, int(top_k))]
        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            item = dict(candidates[int(idx)])
            page = int(item.get("page") or 0)
            page_range = item.get("page_range") or [max(1, page - 1), page + 1]
            item.update(
                {
                    "rank": rank,
                    "relevance_score": round(float(scores[int(idx)]), 4),
                    "recommended_read_pages": f"{page_range[0]}-{page_range[1]}",
                    "read_text_command": (
                        f"read_file(file_path=\"{item.get('pdf_path')}\", "
                        f"pdf_mode=\"text\", pages=\"{page_range[0]}-{page_range[1]}\")"
                    ),
                    "read_render_command": (
                        f"read_file(file_path=\"{item.get('pdf_path')}\", "
                        f"pdf_mode=\"render\", pages=\"{page_range[0]}-{page_range[1]}\")"
                    ),
                }
            )
            results.append(item)
        return {
            "query": query_text,
            "index_status": ready,
            "results": results,
            "skipped_missing_pdf_books": skipped_missing_pdf_books,
        }


def format_theory_results(payload: Dict[str, Any]) -> str:
    results = list(payload.get("results") or [])
    if not results:
        message = str((payload.get("index_status") or {}).get("message") or "No relevant theory-book chunks found.")
        return f"Theory search returned no results.\n{message}"

    lines = [
        "=" * 80,
        "Theory Search Results".center(80),
        "=" * 80,
        f"Query: {payload.get('query', '')}",
        "",
    ]
    for item in results:
        text = str(item.get("text") or "").strip()
        if len(text) > 1200:
            text = text[:1200].rstrip() + "\n...[truncated]"
        section = str(item.get("section") or "").strip()
        lines.extend(
            [
                "-" * 80,
                f"[{item.get('rank')}] {item.get('title')}",
                f"book_id: {item.get('book_id')}",
                f"score: {item.get('relevance_score')}",
                f"page: {item.get('page')}  recommended_read_pages: {item.get('recommended_read_pages')}",
                f"section_hint: {section or '(not detected)'}",
                f"pdf_path: {item.get('pdf_path')}",
                f"text_read: {item.get('read_text_command')}",
                f"render_check: {item.get('read_render_command')}",
                "",
                text,
                "",
            ]
        )
    lines.extend(
        [
            "=" * 80,
            "Use the recommended pages with read_file(..., pdf_mode='text') for deeper reading before citing.",
            "=" * 80,
        ]
    )
    return "\n".join(lines)
