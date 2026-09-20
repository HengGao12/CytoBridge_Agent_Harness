"""Core literature retrieval for single-query knowledge search."""
import os
import json
import time
import ast
import re
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import logging
from collections import Counter

from ..schemas import TheoryChunk, TheoryContext
from .config import DIRS, FILES, SEARCH_CONFIG, call_llm, get_sentence_transformer_model
from .knowledge_builder import KnowledgeBaseBuilder, LiteratureProcessor
from .embedding_builder import DocumentEmbeddingBuilder, ParagraphIndexBuilder
from .rag_tools import cosine_similarity

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - graceful fallback
    BM25Okapi = None

logger = logging.getLogger(__name__)

# ==================== Globals ====================
knowledge_base = []
search_cache = {}

__all__ = ['EnhancedKnowledgeSearcher', 'reciprocal_rank_fusion']


def _analysis(entry: Dict[str, Any]) -> Dict[str, Any]:
    return entry.get("analysis", {}) or {}


def _content(entry: Dict[str, Any]) -> Dict[str, Any]:
    return entry.get("content", {}) or {}


def _summary_text(entry: Dict[str, Any]) -> str:
    analysis = _analysis(entry)
    content = _content(entry)
    return str(
        analysis.get("summary")
        or content.get("abstract")
        or content.get("full_text", "")[:2000]
        or ""
    )


def _retrieval_text(entry: Dict[str, Any]) -> str:
    content = _content(entry)
    analysis = _analysis(entry)
    return str(
        content.get("retrieval_text")
        or content.get("full_text")
        or analysis.get("summary")
        or ""
    )


def _document_text(entry: Dict[str, Any]) -> str:
    analysis = _analysis(entry)
    text_parts = [
        str((entry.get("metadata") or {}).get("title") or ""),
        ' '.join(map(str, analysis.get('key_methods', []) or [])),
        ' '.join(map(str, analysis.get('key_objects', []) or [])),
        ' '.join(map(str, analysis.get('key_technical_contributions', []) or [])),
        ' '.join(map(str, analysis.get('research_domains', []) or [])),
        _summary_text(entry),
        _retrieval_text(entry)[:12000],
    ]
    return ' '.join(part.strip() for part in text_parts if str(part or "").strip())


def _tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_./+-]+", str(text or "").lower())


def maximal_marginal_relevance(
    results: List[Dict[str, Any]],
    embeddings: Optional[np.ndarray],
    top_k: int,
    lambda_mult: float = 0.75,
) -> List[Dict[str, Any]]:
    """
    Deduplicate dense-embedding search results with MMR.
    """
    if not results or embeddings is None or len(results) <= top_k:
        return results[:top_k]

    selected: List[Dict[str, Any]] = []
    selected_vecs: List[np.ndarray] = []
    candidate_pool = [dict(item) for item in results]

    while candidate_pool and len(selected) < top_k:
        best_idx = 0
        best_score = None
        for idx, item in enumerate(candidate_pool):
            emb_idx = item.get("_doc_index")
            if emb_idx is None or emb_idx >= len(embeddings):
                mmr_score = float(item.get("score", 0.0))
            else:
                query_sim = float(item.get("score", 0.0))
                doc_vec = embeddings[emb_idx]
                if selected_vecs:
                    redundancy = max(float(cosine_similarity(doc_vec.tolist(), vec.tolist())) for vec in selected_vecs)
                else:
                    redundancy = 0.0
                mmr_score = lambda_mult * query_sim - (1 - lambda_mult) * redundancy
            if best_score is None or mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        chosen = candidate_pool.pop(best_idx)
        selected.append(chosen)
        emb_idx = chosen.get("_doc_index")
        if emb_idx is not None and emb_idx < len(embeddings):
            selected_vecs.append(embeddings[emb_idx])

    return selected


def _innovation_level(entry: Dict[str, Any]) -> str:
    analysis = _analysis(entry)
    innovation = analysis.get("innovation_assessment") or {}
    return str(innovation.get("level") or "unknown")


def _iter_json_candidates(text: str) -> List[Any]:
    """Best-effort JSON/JSON-like object extraction from an LLM response."""
    value = str(text or "")
    candidates: List[Any] = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", value):
        start = match.start()
        try:
            parsed, _ = decoder.raw_decode(value[start:])
            candidates.append(parsed)
            continue
        except Exception:
            pass
        if value[start] == "[":
            end = value.find("]", start + 1)
            if end > start:
                snippet = value[start : end + 1]
                try:
                    candidates.append(ast.literal_eval(snippet))
                except Exception:
                    continue
    return candidates


def _coerce_rerank_index(value: Any, candidate_count: int) -> Optional[int]:
    """Return a 1-based document index if value unambiguously contains one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        idx = value
    elif isinstance(value, float) and value.is_integer():
        idx = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if re.fullmatch(r"\d+", text):
            idx = int(text)
        else:
            match = re.search(r"(?:doc(?:ument)?|paper|index|idx)\s*#?\s*(\d+)", text, flags=re.IGNORECASE)
            if not match:
                return None
            idx = int(match.group(1))
    else:
        return None
    return idx if 1 <= idx <= candidate_count else None


def _extract_indices_from_rerank_object(value: Any, candidate_count: int) -> List[int]:
    if isinstance(value, list):
        indices: List[int] = []
        for item in value:
            if isinstance(item, dict):
                for key in ("index", "idx", "doc", "doc_index", "document", "document_index", "paper_index"):
                    if key in item:
                        idx = _coerce_rerank_index(item.get(key), candidate_count)
                        if idx is not None:
                            indices.append(idx)
                            break
                continue
            idx = _coerce_rerank_index(item, candidate_count)
            if idx is not None:
                indices.append(idx)
        return indices

    if isinstance(value, dict):
        preferred_keys = (
            "ranked_indices",
            "indices",
            "ranking",
            "ranked",
            "order",
            "results",
            "documents",
            "keep",
            "selected",
        )
        for key in preferred_keys:
            if key in value:
                indices = _extract_indices_from_rerank_object(value.get(key), candidate_count)
                if indices:
                    return indices
        return []

    idx = _coerce_rerank_index(value, candidate_count)
    return [idx] if idx is not None else []


def parse_llm_rerank_indices(response: str, candidate_count: int) -> List[int]:
    """
    Robustly parse a reranker response into 1-based document indices.

    The reranker prompt asks for a JSON list, but small formatting drift should
    not make retrieval fail. We accept arrays, common object wrappers, lists of
    objects, and document-labelled plain text. Invalid or duplicate indices are
    ignored while preserving order.
    """
    if candidate_count <= 0:
        return []

    raw_indices: List[int] = []
    for parsed in _iter_json_candidates(response):
        raw_indices = _extract_indices_from_rerank_object(parsed, candidate_count)
        if raw_indices:
            break

    if not raw_indices:
        text = str(response or "")
        labelled = re.findall(r"(?:doc(?:ument)?|paper)\s*#?\s*(\d+)", text, flags=re.IGNORECASE)
        if labelled:
            raw_indices = [int(item) for item in labelled]
        else:
            raw_indices = [int(item) for item in re.findall(r"\b\d+\b", text)]

    ordered: List[int] = []
    seen = set()
    for idx in raw_indices:
        if 1 <= idx <= candidate_count and idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    return ordered


# ==================== Knowledge-base loading ====================
def load_knowledge_base() -> List[Dict[str, Any]]:
    """Load the structured knowledge base."""
    global knowledge_base
    
    if knowledge_base:
        return knowledge_base
    
    try:
        kb_file = FILES["structured_knowledge_base"]
        logger.info(f"Loading knowledge base: {kb_file}")
        
        if not kb_file.exists():
            logger.error(f"Knowledge-base file does not exist: {kb_file}")
            return []
        
        with open(kb_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            knowledge_base = data.get("knowledge_base", [])
            
            # Normalize APA citation metadata.
            for entry in knowledge_base:
                metadata = entry.get("metadata", {})
                entry["citation_matched"] = bool(metadata.get("apa_citation"))
                entry["apa_citation"] = metadata.get("apa_citation", "")
                
                # Ensure pdf_filename is present.
                if "pdf_filename" not in metadata and "title" in metadata:
                    metadata["pdf_filename"] = metadata["title"].replace(" ", "_") + ".pdf"
            
            logger.info(f"Loaded {len(knowledge_base)} knowledge-base entries")
            return knowledge_base
            
    except Exception as e:
        logger.error(f"Failed to load knowledge base: {str(e)}")
        return []


# ==================== Vector search ====================
def vector_search(query: str, kb: List[Dict], top_k: int = 10) -> List[Dict]:
    """Vector search with the configured embedding model."""
    if not kb:
        return []
    
    logger.debug(f"Running vector search: query='{query[:50]}...'")
    
    texts = [_document_text(entry) for entry in kb]
    
    try:
        model = get_sentence_transformer_model()
        # Encode all documents.
        doc_embeddings = model.encode(texts, show_progress_bar=False)
        
        # Encode the query.
        query_emb = model.encode([query])[0]
        
        # Compute similarities.
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
        logger.error(f"Vector search failed: {e}")
        return []


# ==================== RRF fusion ====================
def reciprocal_rank_fusion(
    result_lists: List[List[Dict]], 
    k: int = 60,
    min_score_threshold: float = 0.1
) -> List[Dict]:
    """
    Reciprocal-rank fusion for multiple result lists.
    
    Args:
        result_lists: Multiple search result lists.
        k: RRF constant.
        min_score_threshold: Minimum score threshold.

    Returns:
        Fused result list.
    """
    if not result_lists:
        return []
    
    # If there is only one retrieval method, return it unchanged.
    if len(result_lists) == 1:
        logger.info("Only one result list; skipping RRF fusion")
        return result_lists[0]
    
    fusion_scores = {}
    all_doc_ids = set()  # Track all observed documents.
    
    for results in result_lists:
        for rank, result in enumerate(results, 1):
            entry = result["entry"]
            metadata = entry.get("metadata", {})
            pdf_filename = metadata.get("pdf_filename", "")
            
            # Use a stable document identifier.
            doc_id = pdf_filename or entry.get("title", f"doc_{rank}")
            all_doc_ids.add(doc_id)
            
            method = result.get("method", "unknown")
            
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = {
                    "entry": entry,
                    "score": 0,
                    "methods_used": [],
                    "original_scores": {},
                    "appearances": 0,
                }
            
            # RRF contribution.
            contribution = 1 / (k + rank)
            fusion_scores[doc_id]["score"] += contribution
            fusion_scores[doc_id]["methods_used"].append(method)
            fusion_scores[doc_id]["original_scores"][method] = result["score"]
            fusion_scores[doc_id]["appearances"] += 1
    
    # Convert to list and sort.
    fused_results = list(fusion_scores.values())
    
    # Add a normalized score that accounts for recurrence.
    max_appearances = max(r["appearances"] for r in fused_results) if fused_results else 1
    for result in fused_results:
        # Combine RRF score with occurrence frequency.
        frequency_boost = result["appearances"] / max_appearances
        result["combined_score"] = result["score"] * (1 + 0.3 * frequency_boost)
    
    # Sort by combined score.
    fused_results.sort(key=lambda x: x.get("combined_score", x["score"]), reverse=True)
    
    for rank, result in enumerate(fused_results, 1):
        result["fusion_rank"] = rank
    
    logger.info(f"RRF fusion: fused {len(result_lists)} result lists into {len(fused_results)} results")
    return fused_results


# ==================== LLM reranking ====================
def llm_rerank(llm_client, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
    """Rerank candidates with an LLM."""
    if not SEARCH_CONFIG["rerank_enabled"] or not candidates or not llm_client:
        return candidates[:top_k]
    
    try:
        doc_texts = []
        for i, cand in enumerate(candidates):
            entry = cand["entry"]
            analysis = _analysis(entry)
            evidence_lines = []
            for chunk in list(cand.get("fine_grained_chunks") or [])[:2]:
                section = str(chunk.get("section") or "unknown")
                page = chunk.get("page", "?")
                evidence = str(chunk.get("text") or "").strip().replace("\n", " ")
                evidence_lines.append(f"- p.{page} [{section}] {evidence[:260]}")
            evidence_block = "\n".join(evidence_lines) if evidence_lines else "- (no chunk evidence attached)"
            doc_texts.append(f"""
Document {i+1}:
Title: {entry['metadata']['title']}
Methods: {', '.join(analysis.get('key_methods', [])[:3])}
Objects: {', '.join(analysis.get('key_objects', [])[:3])}
Summary: {_summary_text(entry)[:200]}
Innovation level: {_innovation_level(entry)}
Chunk evidence:
{evidence_block}
""")
        
        prompt = f"""You are an expert judge of research-paper relevance.

Query: "{query}"

Rank the following documents by relevance to the query. Keep only documents
that are clearly relevant and worth preserving in the final literature context.
Consider:
- direct relevance to the query topic
- methodological alignment
- technical contribution match
- whether chunk evidence directly supports the document-query relationship
- whether the document helps later theory selection, downstream analysis, or manuscript reasoning

Do not keep weakly related papers just to fill the list.
If only three papers are strongly relevant, return only three.
If eight to ten papers are clearly relevant, you may return more.

Documents:
{''.join(doc_texts)}

Return only one JSON object in this exact format: {{"ranked_indices": [2, 1, 4]}}
`ranked_indices` must contain only the retained document indices, ordered from
most to least relevant. Do not return Markdown, explanations, or any text
outside the JSON object.

Ranking result:"""

        response = call_llm(
            llm_client=llm_client,
            user_prompt=prompt,
            temperature=0.1
        )
        
        ranked_indices = parse_llm_rerank_indices(response, len(candidates))
        parse_status = "parsed"
        if not ranked_indices:
            repair_prompt = f"""Your reranking output did not satisfy the structured contract, so no valid document indices could be parsed.

Original query: "{query}"
Candidate document count: {len(candidates)}
Invalid output:
{str(response or '')[:4000]}

Please output again. Return only one JSON object in this format:
{{"ranked_indices": [2, 1, 4]}}

Requirements:
- Use only integer indices from 1 to {len(candidates)}.
- Do not repeat indices.
- Do not keep weakly related papers just to fill the list.
- Do not output Markdown, explanations, or text outside the JSON object.
"""
            repair_response = call_llm(
                llm_client=llm_client,
                user_prompt=repair_prompt,
                temperature=0.0,
            )
            ranked_indices = parse_llm_rerank_indices(repair_response, len(candidates))
            parse_status = "parsed_after_retry" if ranked_indices else "fallback_no_valid_indices"
        if not ranked_indices:
            logger.warning(
                "LLM reranking produced no valid indices; falling back to fused ranking. response_prefix=%r",
                str(response or "")[:240],
            )
            fallback = [dict(item) for item in candidates[:top_k]]
            for pos, item in enumerate(fallback, 1):
                item["_llm_rerank_parse_status"] = parse_status
                item["_llm_rerank_position"] = pos
            return fallback

        reranked = []
        for pos, idx in enumerate(ranked_indices, 1):
            item = dict(candidates[idx - 1])
            item["_llm_rerank_parse_status"] = parse_status
            item["_llm_rerank_position"] = pos
            reranked.append(item)

        logger.info(f"LLM reranking complete: {len(reranked)} results")
        return reranked[:top_k]
            
    except Exception as e:
        logger.error(f"LLM reranking failed: {e}")
        return candidates[:top_k]


# ==================== Searcher class ====================
class EnhancedKnowledgeSearcher:
    """
    Knowledge searcher using the configured retrieval models.
    Handles one query at a time and does not perform HyDE expansion.
    """
    
    def __init__(self, llm_client=None):
        self.knowledge_base = None
        self.initialized = False
        self.llm_client = llm_client
        self.knowledge_builder = None
        self.paragraph_index_builder = None
        self.document_embedding_builder = None
        self.model = None
        self.document_embeddings = None
        self.document_texts = None
        self.document_tokens: List[List[str]] = []
        self.doc_index_map: Dict[str, int] = {}
        self.bm25 = None
        
        if llm_client:
            # Initialize builder instances.
            self.knowledge_builder = KnowledgeBaseBuilder(llm_client=llm_client)
            self.paragraph_index_builder = ParagraphIndexBuilder()
            self.document_embedding_builder = DocumentEmbeddingBuilder(load_model=False)
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the searcher."""
        try:
            logger.info("Initializing EnhancedKnowledgeSearcher...")
            
            kb_file = FILES["structured_knowledge_base"]
            # Build the knowledge base when the cached artifact is missing or empty.
            need_build = False
            if not kb_file.exists() or kb_file.stat().st_size < 100:
                logger.warning("Knowledge-base file is missing or empty; building it now...")
                need_build = True
            
            if need_build and self.knowledge_builder:
                logger.info("Building knowledge base with KnowledgeBaseBuilder...")
                result = self.knowledge_builder.build(verbose=False)
            
            # Load the knowledge base.
            self.knowledge_base = load_knowledge_base()
            
            # Load or build document embeddings.
            if self.knowledge_base:
                logger.info("Loading or building document embeddings...")
                self._precompute_document_embeddings()
                
            # Check and build the paragraph index.
            paragraph_index_file = FILES["paragraph_index"]
            if not paragraph_index_file.exists():
                logger.info("Paragraph index file is missing; building it now...")
                if self.paragraph_index_builder:
                    self.paragraph_index_builder.build(force_rebuild=False, max_pdfs=None)
                else:
                    from .embedding_builder import build_paragraph_index
                    build_paragraph_index(force_rebuild=False, max_pdfs=None)
            else:
                logger.info("Paragraph index exists; checking completeness and filling missing entries.")
                if self.paragraph_index_builder:
                    self.paragraph_index_builder.build(force_rebuild=False, max_pdfs=None)
            
            if self.knowledge_base:
                self.initialized = True
                logger.info(f"Initialization complete with {len(self.knowledge_base)} entries")
                
                with_apa = sum(1 for e in self.knowledge_base if e.get("citation_matched"))
                logger.info(f"APA citations: {with_apa}/{len(self.knowledge_base)}")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
    
    def _precompute_document_embeddings(self):
        """Precompute and cache embeddings for all documents."""
        if not self.knowledge_base:
            return

        texts = [_document_text(entry) for entry in self.knowledge_base]
        self.document_texts = texts
        self.document_tokens = [_tokenize_for_bm25(text) for text in texts]
        self.doc_index_map = {
            str((entry.get("metadata") or {}).get("pdf_filename") or f"doc_{idx}"): idx
            for idx, entry in enumerate(self.knowledge_base)
        }

        builder = self.document_embedding_builder or DocumentEmbeddingBuilder(load_model=False)
        self.document_embedding_builder = builder
        cached = builder.load_cached_embeddings()
        if cached:
            embeddings = cached.get("embeddings")
            metadata = cached.get("metadata") or []
            expected_ids = [
                str((entry.get("metadata") or {}).get("pdf_filename") or "")
                for entry in self.knowledge_base
            ]
            cached_ids = [str(item.get("pdf_filename") or "") for item in metadata if isinstance(item, dict)]
            if (
                isinstance(embeddings, np.ndarray)
                and embeddings.shape[0] == len(expected_ids)
                and cached_ids == expected_ids
            ):
                self.document_embeddings = embeddings
                logger.info(f"Loaded cached document embeddings for {len(self.document_embeddings)} documents")
            else:
                logger.warning(
                    "Document embedding cache does not match the current knowledge base; re-encoding. cache_shape=%s cache_meta=%s kb=%s",
                    getattr(embeddings, "shape", None),
                    len(cached_ids),
                    len(expected_ids),
                )

        if self.document_embeddings is None:
            if self.model is None:
                self.model = get_sentence_transformer_model()
            self.document_embeddings = self.model.encode(texts, show_progress_bar=False)
            builder.save_embeddings_for_knowledge_base(self.knowledge_base, self.document_embeddings)

        if BM25Okapi and self.document_tokens:
            self.bm25 = BM25Okapi(self.document_tokens)
        else:
            self.bm25 = None
        logger.info(f"Document embeddings ready for {len(self.document_embeddings)} documents")

    def dense_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Run vector search with cached document embeddings.
        """
        if not self.knowledge_base or self.document_embeddings is None:
            logger.warning("Document embeddings are not cached; falling back to on-the-fly computation")
            return vector_search(query, self.knowledge_base, top_k)

        # Compute the query embedding.
        if self.model is None:
            self.model = get_sentence_transformer_model()
        query_emb = self.model.encode([query])[0]

        # Compute similarities.
        scores = []
        for idx, doc_emb in enumerate(self.document_embeddings):
            sim = cosine_similarity(query_emb.tolist(), doc_emb.tolist())
            if sim > 0:
                scores.append({
                    "entry": self.knowledge_base[idx],
                    "score": sim,
                    "rank": len(scores) + 1,
                    "method": "dense",
                    "_doc_index": idx,
                })

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    def vector_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Backward-compatible interface; defaults to dense retrieval."""
        return self.dense_search(query, top_k=top_k)

    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self.knowledge_base or not self.document_texts:
            return []

        query_tokens = _tokenize_for_bm25(query)
        if not query_tokens:
            return []

        if self.bm25 is not None:
            raw_scores = self.bm25.get_scores(query_tokens)
        else:
            raw_scores = []
            query_counter = Counter(query_tokens)
            for doc_tokens in self.document_tokens:
                doc_counter = Counter(doc_tokens)
                overlap = sum(min(doc_counter[token], count) for token, count in query_counter.items())
                raw_scores.append(float(overlap))

        results = []
        for idx, score in enumerate(raw_scores):
            if score <= 0:
                continue
            results.append(
                {
                    "entry": self.knowledge_base[idx],
                    "score": float(score),
                    "rank": len(results) + 1,
                    "method": "bm25" if self.bm25 is not None else "keyword",
                    "_doc_index": idx,
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        dense_results = self.dense_search(query, top_k=max(top_k * 2, top_k + 4))
        keyword_results = self.keyword_search(query, top_k=max(top_k * 2, top_k + 4))
        result_lists = [dense_results]
        if keyword_results:
            result_lists.append(keyword_results)
        fused = reciprocal_rank_fusion(result_lists, k=SEARCH_CONFIG.get("rrf_k_constant", 60))
        reranked = maximal_marginal_relevance(
            fused,
            self.document_embeddings,
            top_k=min(len(fused), max(top_k, SEARCH_CONFIG.get("max_search_results", top_k))),
        )
        for rank, item in enumerate(reranked, 1):
            item["rank"] = rank
            item["method"] = "hybrid"
        return reranked[:top_k]
    
    def rebuild_knowledge_base(self, verbose: bool = True) -> Dict[str, Any]:
        """Rebuild the knowledge base."""
        if not self.knowledge_builder:
            logger.error("Knowledge builder is not initialized")
            return {"status": "error", "message": "Knowledge builder not initialized"}
        
        logger.info("Rebuilding knowledge base...")
        result = self.knowledge_builder.build(force_rebuild=True, verbose=verbose)
        
        # Reload the knowledge base.
        global knowledge_base
        knowledge_base = []
        self.knowledge_base = load_knowledge_base()
        if self.knowledge_base:
            self.document_embeddings = None
            self._precompute_document_embeddings()
        
        return result
    
    def rebuild_paragraph_index(self, force_rebuild: bool = True, max_pdfs: int = None) -> Dict[str, Any]:
        """Rebuild the paragraph index."""
        if not self.paragraph_index_builder:
            logger.error("Paragraph index builder is not initialized")
            return {"status": "error", "message": "Paragraph index builder not initialized"}
        
        logger.info("Rebuilding paragraph index...")
        return self.paragraph_index_builder.build(force_rebuild=force_rebuild, max_pdfs=max_pdfs)
    
    def process_single_pdf(self, pdf_path: Path, apa_citations: Dict[str, str] = None) -> Optional[Dict]:
        """Process a single PDF paper."""
        processor = LiteratureProcessor(pdf_path, self.llm_client)
        return processor.process(apa_citations)
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        use_hyde: bool = False,
        use_rerank: bool = True
    ) -> List[Dict]:
        """
        Run a single-query vector retrieval.
        
        Args:
            query: Query string.
            top_k: Number of returned results.
            use_hyde: Whether to use HyDE. Deprecated and retained for API compatibility.
            use_rerank: Whether to use LLM reranking.
            
        Returns:
            List of retrieval results.
        """
        if use_hyde:
            logger.warning("use_hyde is deprecated; HyDE expansion should be handled externally")
        
        logger.info(f"Running single-query retrieval: query='{query[:50]}...', top_k={top_k}")
        
        if not self.initialized:
            logger.warning("Searcher is not initialized")
            return []
        
        # Run hybrid retrieval.
        vector_results = self.hybrid_search(query, top_k=top_k * 2)
        
        # Optional LLM reranking.
        if use_rerank and vector_results and self.llm_client:
            final_results = llm_rerank(self.llm_client, query, vector_results, top_k)
        else:
            final_results = vector_results[:top_k]
        
        # Format output.
        formatted_results = []
        for result in final_results:
            entry = result["entry"]
            analysis = _analysis(entry)
            formatted = {
                "title": entry["metadata"]["title"],
                "relevance_score": round(result["score"], 3),
                "methods_used": result.get("methods_used", [result.get("method", "hybrid")]),
                "apa_citation": entry["metadata"].get("apa_citation", ""),
                "summary": _summary_text(entry),
                "key_methods": list(analysis.get("key_methods", [])[:3]),
                "key_contributions": list(analysis.get("key_technical_contributions", [])[:2]),
                "research_domains": list(analysis.get("research_domains", [])),
                "pdf_filename": entry["metadata"].get("pdf_filename", ""),
                "rerank_position": result.get("_llm_rerank_position"),
                "rerank_parse_status": result.get("_llm_rerank_parse_status"),
            }
            formatted_results.append(formatted)
        
        return formatted_results
