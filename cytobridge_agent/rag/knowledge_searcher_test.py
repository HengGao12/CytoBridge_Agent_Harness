from __future__ import annotations

import numpy as np
from langchain_core.messages import AIMessage

from cytobridge_agent.rag import knowledge_searcher
from cytobridge_agent.rag.config import call_llm
from cytobridge_agent.rag.embedding_builder import DocumentEmbeddingBuilder
from cytobridge_agent.rag.knowledge_searcher import EnhancedKnowledgeSearcher, llm_rerank, parse_llm_rerank_indices
from cytobridge_agent.rag.rag_main import RAGManager


class _FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def chat(self, messages, temperature=0.1, max_tokens=2000):
        del messages, temperature, max_tokens
        return self.response


class _SequenceFakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, temperature=0.1, max_tokens=2000):
        del messages, temperature, max_tokens
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return ""


class _StatefulInvokeLLM:
    def __init__(self):
        self.session_id = "planner-session"
        self.seen_session_ids = []

    def invoke(self, messages):
        del messages
        self.seen_session_ids.append(self.session_id)
        return AIMessage(content="[1, 2, 3]")


class _CloneableStatefulInvokeLLM(_StatefulInvokeLLM):
    def model_copy(self, deep=False):  # noqa: ANN001
        del deep
        clone = _CloneableStatefulInvokeLLM()
        clone.session_id = self.session_id
        return clone


def _candidate(idx: int) -> dict:
    return {
        "entry": {
            "metadata": {"title": f"Paper {idx}", "pdf_filename": f"paper_{idx}.pdf"},
            "analysis": {
                "summary": f"Summary {idx}",
                "key_methods": [f"method_{idx}"],
                "key_objects": ["cell"],
                "innovation_assessment": {"level": "medium"},
            },
            "content": {},
        },
        "score": 1.0 / idx,
        "fine_grained_chunks": [{"page": idx, "section": "Methods", "text": f"evidence {idx}"}],
    }


def _knowledge_base() -> list[dict]:
    return [
        {
            "metadata": {"title": "Paper 1", "pdf_filename": "paper_1.pdf", "year": "2024"},
            "analysis": {
                "summary": "Flow matching for single-cell dynamics.",
                "key_methods": ["flow matching"],
                "key_objects": ["cell dynamics"],
                "key_technical_contributions": ["transport coupling"],
                "research_domains": ["single-cell"],
            },
            "content": {"retrieval_text": "Document one retrieval text."},
        },
        {
            "metadata": {"title": "Paper 2", "pdf_filename": "paper_2.pdf", "year": "2025"},
            "analysis": {
                "summary": "Unbalanced transport for mass changes.",
                "key_methods": ["unbalanced OT"],
                "key_objects": ["mass"],
                "key_technical_contributions": ["growth modeling"],
                "research_domains": ["optimal transport"],
            },
            "content": {"retrieval_text": "Document two retrieval text."},
        },
    ]


def _cache_builder(tmp_path):
    builder = DocumentEmbeddingBuilder(load_model=False)
    builder.cache_file = tmp_path / "document_embeddings.npy"
    builder.metadata_file = tmp_path / "document_embeddings_meta.json"
    return builder


def test_document_embedding_builder_saves_and_loads_cache_without_model(tmp_path) -> None:
    embeddings = np.arange(6, dtype=np.float32).reshape(2, 3)
    builder = _cache_builder(tmp_path)

    result = builder.save_embeddings_for_knowledge_base(_knowledge_base(), embeddings)

    assert result["status"] == "completed"
    assert result["num_documents"] == 2

    reloaded = _cache_builder(tmp_path).load_cached_embeddings()
    assert reloaded is not None
    assert np.array_equal(reloaded["embeddings"], embeddings)
    assert [item["pdf_filename"] for item in reloaded["metadata"]] == ["paper_1.pdf", "paper_2.pdf"]


def test_searcher_precompute_uses_cached_document_embeddings_without_model(monkeypatch, tmp_path) -> None:
    embeddings = np.arange(6, dtype=np.float32).reshape(2, 3)
    builder = _cache_builder(tmp_path)
    builder.save_embeddings_for_knowledge_base(_knowledge_base(), embeddings)

    def _fail_model_load():
        raise AssertionError("document cache hit should not load the sentence transformer")

    monkeypatch.setattr(knowledge_searcher, "get_sentence_transformer_model", _fail_model_load)

    searcher = EnhancedKnowledgeSearcher.__new__(EnhancedKnowledgeSearcher)
    searcher.knowledge_base = _knowledge_base()
    searcher.model = None
    searcher.document_embeddings = None
    searcher.document_texts = None
    searcher.document_tokens = None
    searcher.doc_index_map = {}
    searcher.bm25 = None
    searcher.document_embedding_builder = _cache_builder(tmp_path)

    searcher._precompute_document_embeddings()

    assert np.array_equal(searcher.document_embeddings, embeddings)
    assert searcher.model is None
    assert searcher.document_embedding_builder is not None


def test_parse_llm_rerank_indices_accepts_common_structured_outputs() -> None:
    assert parse_llm_rerank_indices("[2, 1, 4]", 5) == [2, 1, 4]
    assert parse_llm_rerank_indices('{"ranking": [3, 1, 2]}', 5) == [3, 1, 2]
    assert parse_llm_rerank_indices('{"results": [{"index": 4}, {"document": "doc 2"}]}', 5) == [4, 2]
    assert parse_llm_rerank_indices("Ranking: document 2 > document 1 > document 3", 5) == [2, 1, 3]


def test_parse_llm_rerank_indices_filters_invalid_and_duplicates() -> None:
    assert parse_llm_rerank_indices("[2, 99, 2, 1, 0, -1]", 3) == [2, 1]


def test_llm_rerank_degrades_to_original_order_on_unparseable_response(monkeypatch) -> None:
    monkeypatch.setitem(knowledge_searcher.SEARCH_CONFIG, "rerank_enabled", True)
    candidates = [_candidate(1), _candidate(2), _candidate(3)]

    reranked = llm_rerank(_FakeLLM("I cannot rank these documents."), "query", candidates, top_k=2)

    assert [item["entry"]["metadata"]["title"] for item in reranked] == ["Paper 1", "Paper 2"]
    assert all(item.get("_llm_rerank_parse_status") == "fallback_no_valid_indices" for item in reranked)


def test_llm_rerank_accepts_json_object_ranking(monkeypatch) -> None:
    monkeypatch.setitem(knowledge_searcher.SEARCH_CONFIG, "rerank_enabled", True)
    candidates = [_candidate(1), _candidate(2), _candidate(3)]

    reranked = llm_rerank(_FakeLLM('{"ranked_indices": [3, 1]}'), "query", candidates, top_k=3)

    assert [item["entry"]["metadata"]["title"] for item in reranked] == ["Paper 3", "Paper 1"]
    assert [item.get("_llm_rerank_position") for item in reranked] == [1, 2]
    assert all(item.get("_llm_rerank_parse_status") == "parsed" for item in reranked)


def test_llm_rerank_retries_once_on_unstructured_response(monkeypatch) -> None:
    monkeypatch.setitem(knowledge_searcher.SEARCH_CONFIG, "rerank_enabled", True)
    candidates = [_candidate(1), _candidate(2), _candidate(3)]
    llm = _SequenceFakeLLM(["These are relevant but I forgot JSON.", '{"ranked_indices": [2, 1]}'])

    reranked = llm_rerank(llm, "query", candidates, top_k=3)

    assert llm.calls == 2
    assert [item["entry"]["metadata"]["title"] for item in reranked] == ["Paper 2", "Paper 1"]
    assert all(item.get("_llm_rerank_parse_status") == "parsed_after_retry" for item in reranked)


def test_call_llm_isolates_stateful_internal_sessions() -> None:
    llm = _StatefulInvokeLLM()

    response = call_llm(llm, "rank these documents", agent="rag")

    assert response == "[1, 2, 3]"
    assert llm.session_id == "planner-session"
    assert llm.seen_session_ids
    assert llm.seen_session_ids[0] != "planner-session"
    assert "-internal-rag-" in llm.seen_session_ids[0]


def test_rag_manager_clones_stateful_llm_for_internal_calls() -> None:
    llm = _CloneableStatefulInvokeLLM()

    manager = RAGManager(llm_client=llm)

    assert manager.llm_client is not llm
    assert llm.session_id == "planner-session"
    assert manager.llm_client.session_id != "planner-session"
    assert "-rag-" in manager.llm_client.session_id
