"""
RAG (Retrieval-Augmented Generation) module for CytoBridge Agent.

Provides offline BM25-based retrieval from npj_review.pdf for theory support.
"""
from .npj_retriever import NPJRetriever, TheoryRetriever, get_theory_retriever

__all__ = ["NPJRetriever", "TheoryRetriever", "get_theory_retriever"]

