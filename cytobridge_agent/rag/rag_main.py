"""
RAG helper documentation.
"""
import os
import json
import logging
import time
import uuid
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import concurrent.futures

from .config import DIRS, FILES, SEARCH_CONFIG, ensure_dir, call_llm
from ..display import DisplayManager
from .knowledge_builder import KnowledgeBaseBuilder
from .embedding_builder import ParagraphIndexBuilder, DocumentEmbeddingBuilder
from .knowledge_searcher import EnhancedKnowledgeSearcher, reciprocal_rank_fusion, llm_rerank
from .embedding_searcher import DocumentRetriever, PDFRetriever
from .rag_tools import load_apa_citations
from .theory_books import TheoryBookIndex, format_theory_results
from ..utils.llm_runtime import derive_bounded_stateful_session_id

logger = logging.getLogger(__name__)


def _build_isolated_rag_llm(llm_client: Any) -> Any:
    """Return an isolated LLM instance for internal RAG calls when possible.

    RAG reranking/summarization has a strict JSON/text contract and must not
    share stateful provider conversation state with the planner loop.
    """
    if llm_client is None:
        return None
    clone = llm_client
    try:
        if hasattr(llm_client, "model_copy"):
            clone = llm_client.model_copy(deep=False)
        elif hasattr(llm_client, "copy"):
            clone = llm_client.copy(deep=False)
    except Exception:
        clone = llm_client

    if clone is llm_client and hasattr(llm_client, "session_id"):
        logger.warning(
            "Stateful LLM client could not be cloned for RAG isolation; disabling internal RAG LLM calls."
        )
        return None
    if hasattr(clone, "session_id"):
        base_session = str(getattr(llm_client, "session_id", "") or "cytobridge-rag")
        try:
            setattr(
                clone,
                "session_id",
                derive_bounded_stateful_session_id(base_session, f"-rag-{uuid.uuid4().hex[:8]}"),
            )
        except Exception:
            logger.debug("Failed to assign isolated RAG session id", exc_info=True)
    return clone

# RAG helper comment.
HYDE_TEMPLATES = [
    {
        "name": "research_paper_methods",
        "system": "You are a researcher writing about computational methods.",
        "prompt": """Based on the question: "{query}"

Write a paragraph from the Methods section of a research paper that would answer this question.
Focus on technical implementation, algorithms, and procedures.
Include specific technical terms and methodological details.

Methods section excerpt:"""
    },
    {
        "name": "abstract_summary",
        "system": "You are writing a conference paper abstract.",
        "prompt": """Based on: "{query}"

Write a structured abstract that addresses this topic.
Format: Background, Methods, Results, Conclusion.
Keep it concise but technically accurate.

Abstract:"""
    },
    {
        "name": "review_article",
        "system": "You are writing a literature review.",
        "prompt": """Based on: "{query}"

Write a paragraph from a review article that summarizes key developments.
Compare different approaches and highlight major findings.
Include citations to hypothetical papers.

Review excerpt:"""
    },
    {
        "name": "technical_report",
        "system": "You are writing a technical report.",
        "prompt": """Based on: "{query}"

Write a technical summary that includes:
- Problem statement
- Proposed solution
- Key innovations
- Experimental results

Technical summary:"""
    }
]

class RAGManager:
    def __init__(self, llm_client=None):
        """
        RAG helper documentation.
        
        Args:
            RAG helper documentation.
        """
        self.llm_client = _build_isolated_rag_llm(llm_client)
        self.initialized = False
        self.knowledge_base = []
        self.apa_citations = {}
        
        # RAG helper comment.
        self.knowledge_builder = KnowledgeBaseBuilder(llm_client=self.llm_client)
        self.paragraph_index_builder = ParagraphIndexBuilder()
        self.document_embedding_builder = DocumentEmbeddingBuilder(load_model=False)
        self.searcher = None
        self.document_retriever = None
        self.theory_book_index = None
        self.last_search_result: Optional[Dict[str, Any]] = None
        self.last_search_record_paths: Dict[str, str] = {}
        self.last_theory_search_result: Optional[Dict[str, Any]] = None
        
        # RAG helper comment.
        self.paragraph_index_cache = None
        self.pdf_retrievers = {}
        
        # RAG helper comment.
        self.apa_citations = load_apa_citations()
        
        # RAG helper comment.
        if self.llm_client:
            self.initialize()
        
    def initialize(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        RAG helper documentation.
        
        Args:
            RAG helper documentation.
            
        Returns:
            RAG helper documentation.
        """
        # RAG helper comment.
        if self.initialized:
            logger.debug("RAG status message")
            return {"status": "already_initialized"}
        
        logger.info("RAG status message")
        
        # RAG helper comment.
        if not self.llm_client:
            error_msg = "RAG error"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        
        # RAG helper comment.
        if force_rebuild or not FILES["structured_knowledge_base"].exists():
            logger.info("Loaded paragraph index with %s indexed PDFs", len(self.paragraph_index_cache))
            build_result = self.knowledge_builder.build(force_rebuild=force_rebuild)
            if build_result.get("status") == "error":
                return build_result
        else:
            logger.info("RAG status message")
        
        # RAG helper comment.
        if force_rebuild or not FILES["paragraph_index"].exists():
            logger.info("RAG status message")
            self.paragraph_index_builder.build(force_rebuild=force_rebuild, max_pdfs=None)
        else:
            logger.info("RAG status message")
        
        # RAG helper comment.
        self.searcher = EnhancedKnowledgeSearcher(llm_client=self.llm_client)
        self.document_retriever = DocumentRetriever()
        
        # RAG helper comment.
        self.knowledge_base = self._load_knowledge_base()
        
        # RAG helper comment.
        self._load_paragraph_index()
        
        self.initialized = True
        logger.info("RAG status message")
        
        return {
            "status": "success",
            "knowledge_base_size": len(self.knowledge_base),
            "apa_citations": len(self.apa_citations)
        }
    
    def _load_knowledge_base(self) -> List[Dict]:
        """RAG helper."""
        try:
            if FILES["structured_knowledge_base"].exists():
                with open(FILES["structured_knowledge_base"], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("knowledge_base", [])
        except Exception as e:
            logger.error("RAG status message")
        return []
    
    def _load_paragraph_index(self):
        """RAG helper."""
        if not FILES["paragraph_index"].exists():
            logger.warning("RAG status message")
            return
        
        try:
            with open(FILES["paragraph_index"], 'r', encoding='utf-8') as f:
                self.paragraph_index_cache = json.load(f)
            logger.info("RAG status message")

            literature_files = sorted(p.name for p in Path(DIRS["literature"]).glob("*.pdf"))
            indexed_files = set(self.paragraph_index_cache.keys() if isinstance(self.paragraph_index_cache, dict) else [])
            missing = [name for name in literature_files if name not in indexed_files]
            if missing:
                logger.warning(
                    "Paragraph index coverage is incomplete: indexed %s / literature %s; missing %s PDFs. "
                    "Missing papers will trigger slower on-the-fly indexing.",
                    len(indexed_files),
                    len(literature_files),
                    len(missing),
                )
                logger.warning(
                    "Consider a full rebuild with ParagraphIndexBuilder.build(force_rebuild=True, max_pdfs=None)."
                )
        except Exception as e:
            logger.error("RAG status message")
    
    def _get_pdf_retriever(self, pdf_filename: str) -> Optional[PDFRetriever]:
        """
        RAG helper documentation.
        
        Args:
            pdf_filename: PDF filename
            
        Returns:
            RAG helper documentation.
        """
        # RAG helper comment.
        if pdf_filename in self.pdf_retrievers:
            return self.pdf_retrievers[pdf_filename]
        
        # RAG helper comment.
        pdf_path = DIRS["literature"] / pdf_filename
        if not pdf_path.exists():
            logger.warning("RAG status message")
            return None
        
        # RAG helper comment.
        if self.paragraph_index_cache and pdf_filename in self.paragraph_index_cache:
            try:
                # RAG helper comment.
                retriever = PDFRetriever.from_prebuilt_index(
                    pdf_path=pdf_path,
                    index_info=self.paragraph_index_cache[pdf_filename]
                )
                self.pdf_retrievers[pdf_filename] = retriever
                logger.debug("RAG status message")
                return retriever
            except Exception as e:
                logger.error("RAG status message")
        
        # RAG helper comment.
        logger.warning("RAG status message")
        try:
            retriever = PDFRetriever(pdf_path)
            self.pdf_retrievers[pdf_filename] = retriever
            return retriever
        except Exception as e:
            logger.error("RAG status message")
            return None
    
    def _generate_hyde_expansions(self, query: str, num_expansions: int = 2) -> List[str]:
        """
        RAG helper documentation.
        
        Args:
            RAG helper documentation.
            RAG helper documentation.
            
        Returns:
            RAG helper documentation.
        """
        if not self.llm_client:
            logger.warning("RAG status message")
            return [query]
        
        if num_expansions <= 0:
            return [query]
        
        expansions = [query]
        num_to_generate = min(num_expansions, len(HYDE_TEMPLATES))
        display = DisplayManager()
        
        logger.info("RAG status message")
        display.print_status(f"Generating HyDE query expansions ({num_to_generate})...")
        display.print_tool_output(f"HyDE original query:\n{query}")
        
        for i in range(num_to_generate):
            template = HYDE_TEMPLATES[i]
            try:
                user_prompt = template["prompt"].format(query=query)
                response = call_llm(
                    llm_client=self.llm_client,
                    user_prompt=user_prompt,
                    system_prompt=template["system"],
                    temperature=0.7
                )
                
                if response and len(response.strip()) > 50:
                    generated = response.strip().replace('\n', ' ')
                    expansions.append(generated)
                    logger.info("RAG status message")
                    display.print_tool_output(
                        f"HyDE expansion {i+1} [{template['name']}]:\n{generated}"
                    )
                else:
                    logger.warning("RAG status message")
                    if not response:
                        logger.warning("RAG status message")
                    
            except Exception as e:
                logger.error("RAG status message")
        
        logger.info("RAG status message")
        return expansions
    
    def _search_in_document(
        self,
        pdf_filename: str,
        query: str,
        top_k: int = 3,
        neighbor_radius: int = 0,
    ) -> List[Dict]:
        """
        RAG helper documentation.
        
        Args:
            pdf_filename: PDF filename
            query: Query string
            top_k: Number of returned results
            
        Returns:
            RAG helper documentation.
        """
        logger.debug("RAG status message")
        
        retriever = self._get_pdf_retriever(pdf_filename)
        if not retriever:
            logger.warning("RAG status message")
            return []
        
        try:
            ctx = retriever.retrieve(
                query=query,
                top_k=top_k,
                filter_references=True
            )

            if neighbor_radius > 0:
                expanded_chunks = retriever.expand_with_neighbors(
                    [chunk.chunk_id for chunk in ctx.chunks],
                    radius=neighbor_radius,
                )
                score_map = {chunk.chunk_id: chunk.relevance_score for chunk in ctx.chunks}
                chunk_payloads = []
                for item in expanded_chunks:
                    item = dict(item)
                    item["relevance_score"] = float(score_map.get(str(item.get("chunk_id")), 0.0))
                    chunk_payloads.append(item)
                chunk_payloads.sort(
                    key=lambda item: (
                        -float(item.get("relevance_score") or 0.0),
                        int(item.get("neighbor_rank") or 0),
                        int(item.get("page") or 0),
                        int(item.get("char_start") or 0),
                    )
                )
            else:
                chunk_payloads = [
                    {
                        "chunk_id": chunk.chunk_id,
                        "page": chunk.page,
                        "text": chunk.text,
                        "relevance_score": chunk.relevance_score,
                        "section": chunk.section,
                        "block_id": chunk.block_id,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "type": chunk.chunk_type,
                        "neighbor_rank": chunk.neighbor_rank,
                    }
                    for chunk in ctx.chunks
                ]

            formatted = []
            for chunk in chunk_payloads:
                formatted.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "page": chunk.get("page"),
                    "text": chunk.get("text"),
                    "relevance_score": chunk.get("relevance_score", 0.0),
                    "section": chunk.get("section"),
                    "block_id": chunk.get("block_id"),
                    "char_start": chunk.get("char_start"),
                    "char_end": chunk.get("char_end"),
                    "type": chunk.get("type"),
                    "neighbor_rank": chunk.get("neighbor_rank"),
                })

            logger.debug("RAG status message")
            return formatted
            
        except Exception as e:
            logger.error("RAG status message")
            return []
    
    def _resolve_log_dir(self, output_dir: Optional[str] = None) -> Path:
        base = Path(output_dir).expanduser().resolve() if output_dir else Path(DIRS["output"]).resolve()
        log_dir = base / "rag_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _save_search_record(self, query: str, content: str, result: Dict = None, output_dir: Optional[str] = None):
        """
        RAG helper documentation.
        
        Args:
            query: Query string
            RAG helper documentation.
            RAG helper documentation.
        """
        log_dir = self._resolve_log_dir(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # RAG helper comment.
        txt_filename = f"search_{timestamp}.txt"
        txt_path = log_dir / txt_filename

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("RAG status message")
        
        # RAG helper comment.
        if result:
            json_filename = f"search_{timestamp}.json"
            json_path = log_dir / json_filename
            
            record = {
                "query": query,
                "timestamp": timestamp,
                "expanded_queries": result.get("expanded_queries", []),
                "results": result.get("results", [])
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            logger.info("RAG status message")
            self.last_search_record_paths = {
                "text": str(txt_path),
                "json": str(json_path),
            }
        else:
            self.last_search_record_paths = {"text": str(txt_path)}
    
    def search(
        self,
        query: str,
        top_k: int = SEARCH_CONFIG["top_k_default"],
        use_hyde: bool = True,
        num_hyde_expansions: int = 2,
        use_rerank: bool = True,
        fine_grained: bool = False,
        fine_grained_top_k: int = 3,
        return_details: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        RAG helper documentation.
        """
        start_time = time.time()
        
        if not self.initialized:
            self.initialize()
        
        if not self.searcher:
            return {"error": "Searcher is not initialized", "results": []}

        # RAG helper comment.
        expanded_queries = self._generate_hyde_expansions(query, num_hyde_expansions)
        logger.info("RAG status message")

        # RAG helper comment.
        all_raw_results = []
        for i, q in enumerate(expanded_queries):
            logger.info("RAG status message")
            raw_results = self.searcher.hybrid_search(
                q,
                top_k=min(max(top_k * 3, top_k + 4), SEARCH_CONFIG["max_search_results"]),
            )
            formatted_for_fusion = []
            for rank, item in enumerate(raw_results, 1):
                formatted_for_fusion.append({
                    "entry": item["entry"],
                    "score": item["score"],
                    "rank": rank,
                    "source_query": q,
                    "source_query_idx": i
                })
            all_raw_results.append(formatted_for_fusion)

        # RAG helper comment.
        logger.info("RAG status message")
        fused_docs = reciprocal_rank_fusion(all_raw_results, k=60)
        candidate_pool_size = min(
            len(fused_docs),
            max(top_k * 3, top_k + 4, 10),
            SEARCH_CONFIG["max_search_results"],
        )
        fused_docs_top = fused_docs[:candidate_pool_size]
        
        # RAG helper comment.
        if fine_grained:
            logger.info("RAG status message")
            
            # RAG helper comment.
            pdf_items = []
            for idx, doc_item in enumerate(fused_docs_top):
                pdf_filename = doc_item["entry"]["metadata"].get("pdf_filename")
                if pdf_filename:
                    pdf_items.append((idx, pdf_filename))
            
            # RAG helper comment.
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_item = {}
                for idx, pdf_filename in pdf_items:
                    future = executor.submit(
                        self._search_single_pdf_paragraphs,
                        pdf_filename,
                        expanded_queries,
                        fine_grained_top_k
                    )
                    future_to_item[future] = (idx, pdf_filename)
                
                # RAG helper comment.
                for future in concurrent.futures.as_completed(future_to_item):
                    idx, pdf_filename = future_to_item[future]
                    try:
                        chunks = future.result()
                        fused_docs_top[idx]["fine_grained_chunks"] = chunks
                    except Exception as e:
                        logger.error("RAG status message")
                        fused_docs_top[idx]["fine_grained_chunks"] = []
            
            # RAG helper comment.
            for doc_item in fused_docs_top:
                if "fine_grained_chunks" not in doc_item:
                    doc_item["fine_grained_chunks"] = []
        
        # RAG helper comment.
        if use_rerank and self.llm_client and fused_docs_top:
            logger.info("RAG status message")
            rerank_keep = min(
                max(top_k * 2, top_k + 2),
                len(fused_docs_top),
                SEARCH_CONFIG["max_search_results"],
            )
            reranked = llm_rerank(self.llm_client, query, fused_docs_top, top_k=rerank_keep)
            final_docs = reranked
        else:
            final_docs = fused_docs_top[:top_k]
        
        # RAG helper comment.
        doc_results = []
        for item in final_docs:
            entry = item["entry"]
            analysis = entry.get("analysis", {}) or {}
            
            # RAG helper comment.
            pdf_filename = entry["metadata"].get("pdf_filename", "")
            file_path = ""
            if pdf_filename:
                file_path = str(DIRS["literature"] / pdf_filename)
            
            formatted = {
                "title": entry["metadata"]["title"],
                "relevance_score": round(item.get("score", 0), 3),
                "fusion_rank": item.get("fusion_rank", 0),
                "pdf_path": file_path,
                "file_path": file_path,
                "apa_citation": entry["metadata"].get("apa_citation", ""),
                "summary": analysis.get("summary", ""),
                "key_methods": list(analysis.get("key_methods", [])[:3]),
                "key_contributions": list(analysis.get("key_technical_contributions", [])[:2]),
                "innovation_score": ((analysis.get("innovation_assessment") or {}).get("score", 0)),
                "research_domains": list(analysis.get("research_domains", [])),
                "pdf_filename": pdf_filename,
                "rerank_position": item.get("_llm_rerank_position"),
                "rerank_parse_status": item.get("_llm_rerank_parse_status"),
            }
            if fine_grained:
                formatted["fine_grained_chunks"] = item.get("fine_grained_chunks", [])
            doc_results.append(formatted)
        
        result_summary = {
            "query": query,
            "expanded_queries": expanded_queries if use_hyde else None,
            "results": doc_results,
        }
        return result_summary
    
    def _search_single_pdf_paragraphs(self, pdf_filename: str, queries: List[str], top_k: int) -> List[Dict]:
        """
        RAG helper documentation.
        
        Args:
            RAG helper documentation.
            RAG helper documentation.
            RAG helper documentation.
        
        Returns:
            RAG helper documentation.
        """
        retriever = self._get_pdf_retriever(pdf_filename)
        if not retriever:
            return []
        
        all_chunks = []
        seen_ids = set()
        
        for q in queries:
            ctx = retriever.retrieve(q, top_k=top_k, filter_references=True)
            for chunk in ctx.chunks:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    all_chunks.append({
                        "chunk_id": chunk.chunk_id,
                        "page": chunk.page,
                        "text": chunk.text,
                        "relevance_score": chunk.relevance_score,
                        "section": chunk.section,
                        "block_id": chunk.block_id,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "type": chunk.chunk_type,
                        "neighbor_rank": chunk.neighbor_rank,
                    })
        
        # RAG helper comment.
        all_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        return all_chunks[:top_k]

    def search_in_document(
        self,
        pdf_filename: str,
        query: str,
        top_k: int = 3,
        neighbor_radius: int = 0,
    ) -> List[Dict]:
        """
        RAG helper documentation.
        
        Args:
            pdf_filename: PDF filename
            query: Query string
            top_k: Number of returned results
            
        Returns:
            RAG helper documentation.
        """
        return self._search_in_document(pdf_filename, query, top_k, neighbor_radius=neighbor_radius)
    
    def format_literature_context(
        self,
        results: List[Dict],
        max_excerpts_per_paper: int = 3
    ) -> str:
        """
        Format retrieval results as structured literature context.
        
        Args:
            results: List of retrieval results
            max_excerpts_per_paper: Maximum excerpts per paper.
            
        Returns:
            Formatted literature context as structured text.
        """
        if not results:
            return "No relevant literature found."
        
        lines = []
        lines.append("=" * 80)
        lines.append("Literature Retrieval Results".center(76))
        lines.append("=" * 80)
        lines.append(f"Found {len(results)} relevant papers\n")
        
        max_display = int(SEARCH_CONFIG.get("max_display_results", 12))
        for i, result in enumerate(results[:max_display], 1):
            title = result.get("title", "Unknown title")
            file_path = result.get("pdf_path", "") or result.get("file_path", "")
            pdf_filename = result.get("pdf_filename", "")
            apa = result.get("apa_citation", "")
            summary = result.get("summary", "")
            key_methods = result.get("key_methods", [])
            key_contributions = result.get("key_contributions", [])
            innovation_score = result.get("innovation_score", 0)
            research_domains = result.get("research_domains", [])
            relevance_score = result.get("relevance_score", 0)
            chunks = result.get("fine_grained_chunks", [])
            
            # Paper separator.
            lines.append("-" * 80)
            lines.append(f"Paper [{i}]")
            lines.append("-" * 80)
            
            # Basic information.
            lines.append(f"Title: {title}")
            lines.append(f"Relevance: {relevance_score:.3f}")
            if file_path:
                lines.append(f"File path: {file_path}")
                lines.append(f"Close text reading: read_file(file_path=\"{file_path}\", pdf_mode=\"text\", pages=\"1-3\")")
                lines.append(f"Layout check: read_file(file_path=\"{file_path}\", pdf_mode=\"render\", pages=\"1-3\")")
            elif pdf_filename:
                lines.append(f"Close text reading: read_file(file_path=\"{pdf_filename}\", pdf_mode=\"text\", pages=\"1-3\")")
                lines.append(f"Layout check: read_file(file_path=\"{pdf_filename}\", pdf_mode=\"render\", pages=\"1-3\")")
            if apa:
                lines.append(f"APA citation: {apa}")
            if research_domains:
                lines.append(f"Research domains: {', '.join(research_domains)}")
            lines.append(f"Innovation score: {innovation_score}/10")

            # Summary.
            lines.append("\nSummary:")
            lines.append(f"   {summary}")
            
            # Key methods.
            if key_methods:
                lines.append("\nKey methods:")
                for method in key_methods[:5]:
                    lines.append(f"   • {method}")
            
            # Key contributions.
            if key_contributions:
                lines.append("\nKey contributions:")
                for contribution in key_contributions[:3]:
                    lines.append(f"   • {contribution}")
            
            # Relevant excerpts.
            if chunks:
                lines.append(f"\nRelevant excerpts ({len(chunks)} chunks):")
                for j, chunk in enumerate(chunks[:max_excerpts_per_paper], 1):
                    page = chunk.get('page', '?')
                    text = chunk.get('text', '')
                    score = chunk.get('relevance_score', 0)
                    
                    # Format text with indentation and line-length limits.
                    words = text.split()
                    lines_per_chunk = []
                    current_line = []
                    current_length = 0
                    
                    for word in words:
                        if current_length + len(word) + 1 <= 80:
                            current_line.append(word)
                            current_length += len(word) + 1
                        else:
                            lines_per_chunk.append(' '.join(current_line))
                            current_line = [word]
                            current_length = len(word)
                    
                    if current_line:
                        lines_per_chunk.append(' '.join(current_line))
                    
                    lines.append(f"\n   [{i}.{j}] (page {page}, relevance: {score:.3f}):")
                    for line in lines_per_chunk:
                        lines.append(f"      {line}")
            
            lines.append("")  # Blank line separator.
        
        lines.append("=" * 80)
        lines.append("Retrieval Complete".center(76))
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def rebuild_indexes(self, rebuild_kb: bool = False, rebuild_paragraph: bool = True) -> Dict[str, Any]:
        """
        RAG helper documentation.
        
        Args:
            RAG helper documentation.
            RAG helper documentation.
            
        Returns:
            RAG helper documentation.
        """
        results = {}
        
        if rebuild_kb:
            logger.info("RAG status message")
            results["knowledge_base"] = self.knowledge_builder.build(force_rebuild=True)
        
        if rebuild_paragraph:
            logger.info("RAG status message")
            results["paragraph_index"] = self.paragraph_index_builder.build(force_rebuild=True, max_pdfs=None)
            self.pdf_retrievers = {}
        
        # RAG helper comment.
        self.knowledge_base = self._load_knowledge_base()
        self._load_paragraph_index()
        self.searcher = EnhancedKnowledgeSearcher(llm_client=self.llm_client)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """RAG helper."""
        stats = {
            "initialized": self.initialized,
            "knowledge_base_size": len(self.knowledge_base),
            "apa_citations": len(self.apa_citations),
            "knowledge_base_file": str(FILES["structured_knowledge_base"]),
            "paragraph_index_file": str(FILES["paragraph_index"])
        }
        
        # RAG helper comment.
        if FILES["paragraph_index"].exists():
            try:
                with open(FILES["paragraph_index"], 'r') as f:
                    index_data = json.load(f)
                stats["indexed_pdfs"] = len(index_data)
                total_chunks = sum(data.get("num_chunks", 0) for data in index_data.values())
                stats["total_chunks"] = total_chunks
                
                # RAG helper comment.
                if self.paragraph_index_cache:
                    stats["loaded_indexes"] = len(self.paragraph_index_cache)
            except:
                stats["indexed_pdfs"] = 0
                stats["total_chunks"] = 0
        
        return stats
    
    # RAG helper comment.
    def search_literature(
        self,
        query: str,
        top_k: int = SEARCH_CONFIG["top_k_default"],
        use_hyde: bool = True,
        num_hyde_expansions: int = 2,
        use_rerank: bool = True,
        include_excerpts: bool = True,
        fine_grained_top_k: int = 3,
        llm_client=None,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        RAG helper documentation.
        
        Returns:
            RAG helper documentation.
        """
        result = self.search(
            query=query,
            top_k=top_k,
            use_hyde=use_hyde,
            num_hyde_expansions=num_hyde_expansions,
            use_rerank=use_rerank,
            fine_grained=include_excerpts,
            fine_grained_top_k=fine_grained_top_k,
            output_dir=output_dir,
        )

        # RAG helper comment.
        content = self.format_literature_context(
            result.get("results", []),
            max_excerpts_per_paper=fine_grained_top_k
        )
        
        # RAG helper comment.
        self._save_search_record(query, content, result, output_dir=output_dir)
        self.last_search_result = result
        
        return content

    def search_theory(
        self,
        query: str,
        top_k: int = 5,
        output_dir: Optional[str] = None,
    ) -> str:
        """Search long-form mathematical theory books and return page-localized excerpts."""
        if self.theory_book_index is None:
            self.theory_book_index = TheoryBookIndex()
        result = self.theory_book_index.search(query, top_k=top_k)
        content = format_theory_results(result)
        self.last_theory_search_result = result
        self._save_search_record(
            f"[theory_books] {query}",
            content,
            {
                "expanded_queries": [],
                "results": result.get("results", []),
                "index_status": result.get("index_status", {}),
            },
            output_dir=output_dir,
        )
        return content
