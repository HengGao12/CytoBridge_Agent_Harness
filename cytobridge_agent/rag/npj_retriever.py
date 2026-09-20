"""
NPJ Review PDF Retriever for CytoBridge Agent.

Provides offline BM25-based retrieval from npj_review.pdf (or any theory PDF).
No embedding model required - pure lexical matching.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple
import logging

from ..schemas import TheoryChunk, TheoryContext

logger = logging.getLogger(__name__)


def _extract_pdf_text(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Extract text from PDF, returning list of (page_number, text) tuples.
    Uses pypdf for extraction.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed. RAG disabled.")
        return []
    
    pages = []
    try:
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Clean up common PDF extraction artifacts
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 50:  # Skip nearly empty pages
                pages.append((i + 1, text))
    except Exception as e:
        logger.warning(f"Failed to extract PDF text: {e}")
        return []
    
    return pages


def _chunk_text(
    pages: List[Tuple[int, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Tuple[str, int, str]]:
    """
    Split page texts into overlapping chunks.
    Returns list of (chunk_id, page_number, chunk_text).
    """
    chunks = []
    for page_num, text in pages:
        words = text.split()
        if len(words) <= chunk_size:
            chunk_id = f"p{page_num}_c0"
            chunks.append((chunk_id, page_num, text))
        else:
            start = 0
            chunk_idx = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = ' '.join(words[start:end])
                chunk_id = f"p{page_num}_c{chunk_idx}"
                chunks.append((chunk_id, page_num, chunk_text))
                start += chunk_size - chunk_overlap
                chunk_idx += 1
    return chunks


class BM25Index:
    """
    Simple BM25 index for offline retrieval.
    Uses rank_bm25 library.
    """
    
    def __init__(self, chunks: List[Tuple[str, int, str]]):
        """
        Initialize BM25 index from chunks.
        
        Args:
            chunks: List of (chunk_id, page_num, chunk_text)
        """
        self.chunks = chunks
        self.chunk_ids = [c[0] for c in chunks]
        self.page_nums = [c[1] for c in chunks]
        self.texts = [c[2] for c in chunks]
        
        # Tokenize for BM25
        self.tokenized = [self._tokenize(t) for t in self.texts]
        
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.tokenized)
        except ImportError:
            logger.warning("rank_bm25 not installed. Using fallback keyword matching.")
            self.bm25 = None
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, remove punctuation, split."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, int, str, float]]:
        """
        Search for relevant chunks.
        
        Returns:
            List of (chunk_id, page_num, text, score)
        """
        if not self.chunks:
            return []
        
        query_tokens = self._tokenize(query)
        
        if self.bm25 is not None:
            scores = self.bm25.get_scores(query_tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            results = [
                (self.chunk_ids[i], self.page_nums[i], self.texts[i], float(scores[i]))
                for i in top_indices
                if scores[i] > 0
            ]
        else:
            # Fallback: simple keyword overlap
            results = []
            query_set = set(query_tokens)
            for i, tokens in enumerate(self.tokenized):
                overlap = len(query_set & set(tokens))
                if overlap > 0:
                    score = overlap / max(len(query_set), 1)
                    results.append((self.chunk_ids[i], self.page_nums[i], self.texts[i], score))
            results = sorted(results, key=lambda x: x[3], reverse=True)[:top_k]
        
        return results


class NPJRetriever:
    """
    Retriever for npj_review.pdf (CytoBridge theory paper).
    
    Extracts text, chunks it, and provides BM25 search.
    Caches the index for faster subsequent calls.
    """
    
    def __init__(
        self,
        pdf_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        """
        Initialize the retriever.
        
        Args:
            pdf_path: Path to npj_review.pdf. If None, tries to find it in CytoBridge-main.
            cache_dir: Directory to cache the index. If None, no caching.
            chunk_size: Number of words per chunk.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self.pdf_path = pdf_path
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._index: Optional[BM25Index] = None
        self._initialized = False
    
    def _find_pdf(self) -> Optional[Path]:
        """Try to find npj_review.pdf in common locations."""
        if self.pdf_path and Path(self.pdf_path).exists():
            return Path(self.pdf_path)
        
        # Search in common locations
        candidates = [
            Path(__file__).parent.parent.parent / "CytoBridge-main" / "npj_review.pdf",
            Path.cwd() / "CytoBridge-main" / "npj_review.pdf",
            Path.cwd() / "npj_review.pdf",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None
    
    def _get_cache_path(self, pdf_path: Path) -> Optional[Path]:
        """Get cache file path based on PDF hash."""
        if not self.cache_dir:
            return None
        
        # Hash the PDF for cache key
        hasher = hashlib.md5()
        hasher.update(str(pdf_path).encode())
        hasher.update(str(self.chunk_size).encode())
        hasher.update(str(self.chunk_overlap).encode())
        cache_key = hasher.hexdigest()[:16]
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"npj_index_{cache_key}.json"
    
    def _load_cache(self, cache_path: Path) -> Optional[List[Tuple[str, int, str]]]:
        """Load cached chunks from JSON."""
        try:
            with cache_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
                return [(c['id'], c['page'], c['text']) for c in data]
        except Exception:
            return None
    
    def _save_cache(self, cache_path: Path, chunks: List[Tuple[str, int, str]]):
        """Save chunks to JSON cache."""
        try:
            data = [{'id': c[0], 'page': c[1], 'text': c[2]} for c in chunks]
            with cache_path.open('w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def initialize(self) -> bool:
        """
        Initialize the retriever by loading/building the index.
        
        Returns:
            True if successful, False otherwise.
        """
        if self._initialized:
            return self._index is not None
        
        self._initialized = True
        
        pdf_path = self._find_pdf()
        if not pdf_path:
            logger.warning("npj_review.pdf not found. Theory RAG disabled.")
            return False
        
        # Try cache first
        cache_path = self._get_cache_path(pdf_path)
        if cache_path and cache_path.exists():
            chunks = self._load_cache(cache_path)
            if chunks:
                logger.info(f"Loaded {len(chunks)} chunks from cache.")
                self._index = BM25Index(chunks)
                return True
        
        # Extract and chunk
        logger.info(f"Extracting text from {pdf_path}...")
        pages = _extract_pdf_text(pdf_path)
        if not pages:
            logger.warning("No text extracted from PDF.")
            return False
        
        chunks = _chunk_text(pages, self.chunk_size, self.chunk_overlap)
        logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages.")
        
        # Cache
        if cache_path:
            self._save_cache(cache_path, chunks)
        
        self._index = BM25Index(chunks)
        return True
    
    def retrieve(self, query: str, top_k: int = 5) -> TheoryContext:
        """
        Retrieve relevant theory chunks for a query.
        
        Args:
            query: Search query (natural language).
            top_k: Number of chunks to return.
        
        Returns:
            TheoryContext with retrieved chunks.
        """
        if not self._initialized:
            self.initialize()
        
        if not self._index:
            return TheoryContext(query=query, chunks=[])
        
        results = self._index.search(query, top_k=top_k)
        
        chunks = [
            TheoryChunk(
                chunk_id=chunk_id,
                text=text,
                page=page_num,
                relevance_score=score,
            )
            for chunk_id, page_num, text, score in results
        ]
        
        return TheoryContext(query=query, chunks=chunks)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[TheoryChunk]:
        """Retrieve a specific chunk by ID."""
        if not self._index:
            return None
        
        for cid, page, text in self._index.chunks:
            if cid == chunk_id:
                return TheoryChunk(chunk_id=cid, text=text, page=page)
        return None


# Singleton-like accessor
_default_retriever: Optional[NPJRetriever] = None


def get_theory_retriever(
    pdf_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> NPJRetriever:
    """
    Get the default theory retriever instance.
    
    Creates a singleton instance on first call.
    """
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = NPJRetriever(pdf_path=pdf_path, cache_dir=cache_dir)
    return _default_retriever


# Alias for convenience
TheoryRetriever = NPJRetriever

