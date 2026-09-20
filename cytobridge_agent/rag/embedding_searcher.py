"""
Embedding searcher - vector retrieval utilities
"""
import numpy as np
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

from ..schemas import TheoryChunk, TheoryContext
from .config import DIRS, FILES, get_sentence_transformer_model
from .embedding_builder import ParagraphIndexBuilder, DocumentEmbeddingBuilder
from .rag_tools import (
    cosine_similarity,
    is_likely_reference_block,
    extract_structured_pages_up_to_references,
    chunk_by_paragraphs,
)

logger = logging.getLogger(__name__)

class SentenceTransformerIndex:
    """
    Dense retrieval index using Sentence Transformers
    """
    def __init__(self, chunks: List[Dict[str, Any]], pdf_filename: str = None, precomputed_embeddings: np.ndarray = None):
        """
        Initialize the index
        
        Args:
            chunks: Chunk list [{"chunk_id": ..., "page": ..., "text": ...}, ...]
            pdf_filename: PDF filename
            precomputed_embeddings: Precomputed embeddings; skip load and compute when provided
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers not installed.")
        
        self.chunks = [dict(chunk) for chunk in chunks]
        self.chunk_ids = [str(c.get("chunk_id") or "") for c in self.chunks]
        self.page_nums = [int(c.get("page") or 0) for c in self.chunks]
        self.texts = [str(c.get("text") or "") for c in self.chunks]
        self.pdf_filename = pdf_filename
        
        if precomputed_embeddings is not None:
            # Use precomputed embeddings
            self.embeddings = precomputed_embeddings
            logger.info("RAG status message")
        else:
            # Try loading precomputed embeddings locally
            self.embeddings = self._load_cached_embeddings()
            
            if self.embeddings is None:
                logger.info("No cached embeddings found; computing on the fly...")
                self.model = get_sentence_transformer_model()
                logger.info("Encoding chunks for dense retrieval...")
                self.embeddings = self.model.encode(self.texts, show_progress_bar=False)
                logger.info("RAG status message")
    
    def _load_cached_embeddings(self) -> Optional[np.ndarray]:
        """Load cached embeddings from local files"""
        if not self.pdf_filename:
            return None
        
        try:
            index_file = FILES["paragraph_index"]
            if not index_file.exists():
                return None
            
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            if self.pdf_filename not in index_data:
                return None
            
            pdf_info = index_data[self.pdf_filename]
            embeddings_file = pdf_info.get("embeddings_file")
            
            if embeddings_file:
                full_path = DIRS["embedding_base"] / embeddings_file
                if full_path.exists():
                    logger.info(f"Loaded cached embeddings from {full_path}")
                    return np.load(full_path)
                else:
                    logger.warning(f"Embeddings file does not exist: {full_path}")
                    return None
            else:
                logger.warning(f"Index has no embeddings_file information: {self.pdf_filename}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to load cached embeddings: {e}")
            return None
        
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Dense retrieval"""
        if not self.chunks or len(self.embeddings) == 0:
            return []

        if not hasattr(self, 'model'):
            self.model = get_sentence_transformer_model()

        query_emb = self.model.encode([query])[0]
        dot_products = np.dot(self.embeddings, query_emb)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9
        similarities = dot_products / norms

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0:
                chunk = dict(self.chunks[idx])
                chunk["relevance_score"] = score
                results.append(chunk)
        return results


class PDFRetriever:
    """In-PDF document retriever"""
    
    def __init__(self, pdf_path: Path, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize PDFRetriever in on-the-fly mode
        
        Args:
            pdf_path: PDF File paths
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap
        """
        self.pdf_path = Path(pdf_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pdf_filename = self.pdf_path.name
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_lookup: Dict[str, Dict[str, Any]] = {}
        self._index: Optional[SentenceTransformerIndex] = None
        self._initialize()
    
    @classmethod
    def from_prebuilt_index(cls, pdf_path: Path, index_info: Dict):
        """
        Create PDFRetriever from a prebuilt index
        
        Args:
            pdf_path: PDF File paths
            index_info: Index information containing chunks and embeddings_file
            
        Returns:
            PDFRetriever instance
        """
        instance = cls.__new__(cls)
        instance.pdf_path = Path(pdf_path)
        instance.pdf_filename = instance.pdf_path.name
        instance.chunk_size = index_info.get("chunk_size", 500)
        instance.chunk_overlap = index_info.get("chunk_overlap", 100)
        
        # Reconstruct chunks from index information
        chunks = [dict(chunk_info) for chunk_info in index_info.get("chunks", [])]
        
        # Load precomputed embeddings
        embeddings_file = index_info.get("embeddings_file")
        if embeddings_file:
            full_path = DIRS["embedding_base"] / embeddings_file
            if full_path.exists():
                embeddings = np.load(full_path)
                logger.info("RAG status message")
            else:
                logger.warning(f"Embeddings file does not exist: {full_path}")
                embeddings = None
        else:
            embeddings = None
        
        # Create index
        instance.chunks = chunks
        instance.chunk_lookup = {str(chunk.get("chunk_id")): chunk for chunk in chunks}
        instance._index = SentenceTransformerIndex(
            chunks=chunks, 
            pdf_filename=instance.pdf_filename,
            precomputed_embeddings=embeddings
        )
        
        logger.info("RAG status message")
        return instance

    def _initialize(self):
        """Initialize in on-the-fly mode"""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        structured_pages = extract_structured_pages_up_to_references(self.pdf_path)
        if not structured_pages:
            raise ValueError(f"No text extracted from {self.pdf_path}")

        self.chunks = chunk_by_paragraphs(
            structured_pages,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        if not self.chunks:
            raise ValueError(f"No chunks created from {self.pdf_path}")
        self.chunk_lookup = {str(chunk.get("chunk_id")): chunk for chunk in self.chunks}
        self._index = SentenceTransformerIndex(self.chunks, pdf_filename=self.pdf_filename)

    def retrieve(self, query: str, top_k: int = 5, filter_references: bool = True) -> TheoryContext:
        """Retrieve relevant passages"""
        if not self._index:
            return TheoryContext(query=query, chunks=[])

        results = self._index.search(query, top_k=max(top_k * 4, top_k))

        if filter_references:
            results = [r for r in results if not is_likely_reference_block(str(r.get("text") or ""))]
        results = results[:top_k]

        chunks = [
            TheoryChunk(
                chunk_id=str(item.get("chunk_id") or ""),
                text=str(item.get("text") or ""),
                page=int(item.get("page") or 0),
                relevance_score=float(item.get("relevance_score") or 0.0),
                section=item.get("section"),
                block_id=item.get("block_id"),
                char_start=item.get("char_start"),
                char_end=item.get("char_end"),
                chunk_type=item.get("type"),
            )
            for item in results
        ]
        return TheoryContext(query=query, chunks=chunks)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self.chunk_lookup.get(str(chunk_id))

    def expand_with_neighbors(self, chunk_ids: List[str], radius: int = 1) -> List[Dict[str, Any]]:
        if radius <= 0 or not self.chunks:
            return [dict(self.chunk_lookup[cid]) for cid in chunk_ids if cid in self.chunk_lookup]

        order_map = {str(chunk.get("chunk_id")): idx for idx, chunk in enumerate(self.chunks)}
        selected_indices = set()
        primary_ids = [str(cid) for cid in chunk_ids if str(cid) in order_map]
        for cid in primary_ids:
            idx = order_map[cid]
            start = max(0, idx - radius)
            end = min(len(self.chunks), idx + radius + 1)
            for neighbor_idx in range(start, end):
                selected_indices.add(neighbor_idx)

        expanded: List[Dict[str, Any]] = []
        for idx in sorted(selected_indices):
            chunk = dict(self.chunks[idx])
            distance = min(abs(idx - order_map[cid]) for cid in primary_ids) if primary_ids else 0
            chunk["neighbor_rank"] = distance
            expanded.append(chunk)
        return expanded


class DocumentRetriever:
    """Document-level retriever using precomputed document embeddings"""
    
    def __init__(self):
        self.builder = DocumentEmbeddingBuilder(load_model=False)
        self.cached_data = self.builder.load_cached_embeddings()
        self.model = None
    
    def search(self, query: str, knowledge_base: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Run vector retrieval at the document level
        
        Args:
            query: Query string
            knowledge_base: Knowledge-base entry list
            top_k: Number of returned results
            
        Returns:
            List of retrieval results
        """
        if not knowledge_base:
            return []
        
        # Use cached embeddings when available
        if self.cached_data:
            embeddings = self.cached_data["embeddings"]
            metadata = self.cached_data["metadata"]
            
            # Build a knowledge_base-to-metadata mapping
            kb_map = {entry['metadata']['pdf_filename']: entry for entry in knowledge_base}
            
            if self.model is None:
                self.model = get_sentence_transformer_model()
            query_emb = self.model.encode([query])[0]
            
            scores = []
            for idx, doc_emb in enumerate(embeddings):
                sim = cosine_similarity(query_emb.tolist(), doc_emb.tolist())
                if sim > 0:
                    pdf_filename = metadata[idx]["pdf_filename"]
                    if pdf_filename in kb_map:
                        scores.append({
                            "entry": kb_map[pdf_filename],
                            "score": sim,
                            "rank": len(scores) + 1,
                            "method": "vector"
                        })
            
            scores.sort(key=lambda x: x["score"], reverse=True)
            return scores[:top_k]
        
        # Compute on the fly when no cache exists
        else:
            return self._vector_search_realtime(query, knowledge_base, top_k)
    
    def _vector_search_realtime(self, query: str, kb: List[Dict], top_k: int = 10) -> List[Dict]:
        """On-the-fly vector retrieval used when no cache exists"""
        if self.model is None:
            self.model = get_sentence_transformer_model()
        texts = []
        for entry in kb:
            content = entry.get("content", {}) or {}
            text_parts = [
                entry['metadata']['title'],
                ' '.join(entry['analysis'].get('key_methods', [])),
                ' '.join(entry['analysis'].get('key_objects', [])),
                ' '.join(entry['analysis'].get('key_technical_contributions', [])),
                ' '.join(entry['analysis'].get('research_domains', [])),
                entry['analysis'].get('summary', ''),
                str(content.get('retrieval_text') or content.get('full_text') or '')[:12000],
            ]
            texts.append(' '.join(text_parts))
        
        try:
            doc_embeddings = self.model.encode(texts, show_progress_bar=False)
            query_emb = self.model.encode([query])[0]
            
            scores = []
            for idx, doc_emb in enumerate(doc_embeddings):
                sim = cosine_similarity(query_emb.tolist(), doc_emb.tolist())
                if sim > 0:
                    scores.append({
                        "entry": kb[idx],
                        "score": sim,
                        "rank": len(scores) + 1,
                        "method": "vector"
                    })
            
            scores.sort(key=lambda x: x["score"], reverse=True)
            return scores[:top_k]
            
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return []
