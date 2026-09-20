"""
RAG helper documentation.
"""
import logging
from typing import List, Dict, Optional, Any

from .knowledge_searcher import EnhancedKnowledgeSearcher
from .config import SEARCH_CONFIG

logger = logging.getLogger(__name__)

class GitHubSearcher:
    """
    RAG helper documentation.
    """
    
    def __init__(self, llm_client=None):
        self.base_searcher = EnhancedKnowledgeSearcher(llm_client)
    
    def search_code(
        self,
        query: str,
        top_k: int = 5,
        content_type: Optional[str] = None,  # RAG helper comment.
        language: Optional[str] = None,
        use_hyde: bool = False,
        use_rerank: bool = True
    ) -> List[Dict]:
        """
        RAG helper documentation.
        
        Args:
            query: Query string
            top_k: Number of returned results
            RAG helper documentation.
            RAG helper documentation.
            RAG helper documentation.
            RAG helper documentation.
            
        Returns:
            List of retrieval results
        """
        # RAG helper comment.
        results = self.base_searcher.search(
            query=query,
            top_k=top_k * 2,  # RAG helper comment.
            use_hyde=use_hyde,
            use_rerank=use_rerank
        )
        
        # RAG helper comment.
        code_results = []
        for res in results:
            metadata = res.get("metadata", {})
            # RAG helper comment.
            if metadata.get("repo_name"):  # RAG helper comment.
                if content_type and metadata.get("content_type") != content_type:
                    continue
                if language and metadata.get("language") != language:
                    continue
                code_results.append(res)
        
        # RAG helper comment.
        code_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return code_results[:top_k]
    
    def get_repo_summary(self, repo_url: str) -> List[Dict]:
        """
        RAG helper documentation.
        """
        # RAG helper comment.
        entries = []
        for entry in self.base_searcher.knowledge_base:
            if entry.get("metadata", {}).get("repo_url") == repo_url:
                entries.append(entry)
        return entries
    
    def rebuild_index(self):
        """RAG helper."""
        return self.base_searcher.rebuild_knowledge_base()