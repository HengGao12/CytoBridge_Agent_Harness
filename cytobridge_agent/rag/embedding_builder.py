"""Embedding builders for document and paragraph retrieval indexes."""
import json
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import logging

from .config import DIRS, FILES, KB_CONFIG, MODEL_CONFIG, ensure_dir, get_sentence_transformer_model
from .rag_tools import (
    extract_pages_up_to_references,
    extract_structured_pages_up_to_references,
    extract_text_up_to_references,  # Backward-compatible legacy helper.
    chunk_by_paragraphs,
    load_apa_citations,
    get_text_stats,
)

logger = logging.getLogger(__name__)


class ParagraphIndexBuilder:
    """Build and manage the offline paragraph index."""

    def __init__(self, model_name: str = None):
        """
        Initialize the paragraph index builder.

        Args:
            model_name: Sentence Transformer model name.
        """
        self.model_name = model_name or MODEL_CONFIG["sentence_transformer"]["model_name"]
        self.model = None
        self.index_file = FILES["paragraph_index"]

        # Per-PDF embedding cache directory.
        self.pdf_embedding_dir = DIRS["embedding_base"] / "pdf_embeddings"
        ensure_dir(self.pdf_embedding_dir)

    def build(self, force_rebuild: bool = False, max_pdfs: Optional[int] = None) -> Dict[str, Any]:
        """
        Build the paragraph index.

        Args:
            force_rebuild: Rebuild all indexed PDFs.
            max_pdfs: Maximum number of PDFs to process. None means all PDFs.

        Returns:
            Build summary.
        """
        literature_dir = DIRS["literature"]
        all_pdf_files = sorted(literature_dir.glob("*.pdf"), key=lambda p: p.name.lower())
        pdf_files = all_pdf_files
        limited_build = max_pdfs is not None and len(all_pdf_files) > max_pdfs
        existing_index_data: Dict[str, Any] = {}

        if self.index_file.exists() and not force_rebuild:
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    existing_index_data = json.load(f)
                logger.info(f"Found existing paragraph index with {len(existing_index_data)} PDFs")
            except Exception as e:
                logger.warning(f"Failed to read existing paragraph index; rebuilding: {e}")
                existing_index_data = {}

        if max_pdfs is not None and len(all_pdf_files) > max_pdfs:
            logger.info(f"Processing the first {max_pdfs} PDFs out of {len(all_pdf_files)} for testing")
            pdf_files = all_pdf_files[:max_pdfs]
        else:
            logger.info(f"Processing all {len(pdf_files)} PDFs")

        pdfs_to_process: List[Path] = []
        for pdf_path in pdf_files:
            existing = existing_index_data.get(pdf_path.name) if existing_index_data else None
            if force_rebuild or not existing:
                pdfs_to_process.append(pdf_path)
                continue

            schema_version = int(existing.get("schema_version", 1))
            embeddings_file = str(existing.get("embeddings_file") or "").strip()
            embeddings_ok = False
            if embeddings_file:
                embeddings_path = DIRS["embedding_base"] / embeddings_file
                embeddings_ok = embeddings_path.exists()
            if schema_version < 2 or not embeddings_ok:
                pdfs_to_process.append(pdf_path)

        if not force_rebuild and not pdfs_to_process:
            logger.info("Paragraph index is already complete; skipping build.")
            return {
                "status": "skipped",
                "message": "Index already up-to-date",
                "index_file": str(self.index_file),
                "indexed_pdfs": len(existing_index_data),
            }

        logger.info(
            f"Building paragraph index: target PDFs={len(pdf_files)}, pending PDFs={len(pdfs_to_process)} "
            f"(force_rebuild={force_rebuild})"
        )

        self._load_model()

        index_data = {} if force_rebuild else dict(existing_index_data)
        successful = 0
        failed = 0
        total_chunks = 0
        total_pages = 0

        for pdf_path in pdfs_to_process:
            try:
                logger.info(f"Processing: {pdf_path.name}")

                # Use structured pages to preserve block and section metadata.
                pages = extract_structured_pages_up_to_references(pdf_path)
                if not pages:
                    logger.warning(f"  Failed to extract text: {pdf_path.name}")
                    failed += 1
                    continue

                total_pages += len(pages)
                
                # Split structured pages into retrieval chunks.
                chunks = chunk_by_paragraphs(
                    pages,
                    chunk_size=KB_CONFIG["chunk_size"],
                    chunk_overlap=KB_CONFIG["chunk_overlap"]
                )
                if not chunks:
                    logger.warning(f"  Failed to chunk text: {pdf_path.name}")
                    failed += 1
                    continue

                chunk_texts = [str(c.get("text") or "") for c in chunks]
                embeddings = self.model.encode(chunk_texts, show_progress_bar=True)

                pdf_index = []
                for chunk, emb in zip(chunks, embeddings):
                    chunk_id = str(chunk.get("chunk_id") or "")
                    page = int(chunk.get("page") or 0)
                    text = str(chunk.get("text") or "")
                    # Store lightweight text statistics for diagnostics.
                    stats = get_text_stats(text)

                    pdf_index.append({
                        "chunk_id": chunk_id,
                        "page": page,
                        "text": text,
                        "type": chunk.get("type", "text"),
                        "section": chunk.get("section"),
                        "block_id": chunk.get("block_id"),
                        "char_start": chunk.get("char_start"),
                        "char_end": chunk.get("char_end"),
                        "char_count": stats["char_count"],
                        "word_count": stats["word_count"],
                        "sentence_count": stats["sentence_count"]
                    })

                pdf_embeddings_file = self.pdf_embedding_dir / f"{pdf_path.stem}_embeddings.npy"
                np.save(pdf_embeddings_file, embeddings)

                index_data[pdf_path.name] = {
                    "chunks": pdf_index,
                    "num_chunks": len(pdf_index),
                    "num_pages": len(pages),
                    "schema_version": 2,
                    "embeddings_file": str(pdf_embeddings_file.relative_to(DIRS["embedding_base"])),
                    "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
                    "last_updated": datetime.now().isoformat(),
                    "total_chars": sum(c["char_count"] for c in pdf_index),
                    "total_words": sum(c["word_count"] for c in pdf_index)
                }
                successful += 1
                total_chunks += len(pdf_index)
                logger.info(f"  Success: {len(pdf_index)} chunks, {len(pages)} pages, embeddings saved to {pdf_embeddings_file}")

            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                failed += 1

        if limited_build and not force_rebuild:
            logger.info("max_pdfs was set: incrementally updating the selected subset without overwriting the full index.")

        self._save_index(index_data)

        logger.info(f"Paragraph index build complete: successful={successful}, failed={failed}")
        logger.info(f"Processed {total_pages} pages and generated {total_chunks} text chunks")

        if successful > 0:
            logger.info(f"\nEmbedding files saved under: {self.pdf_embedding_dir}")
            for file in self.pdf_embedding_dir.glob("*_embeddings.npy"):
                size_mb = file.stat().st_size / (1024 * 1024)
                logger.info(f"   - {file.name} ({size_mb:.2f} MB)")

        return {
            "status": "completed",
            "successful": successful,
            "failed": failed,
            "total_chunks": total_chunks,
            "total_pages": total_pages,
            "index_file": str(self.index_file),
            "embedding_dir": str(self.pdf_embedding_dir)
        }

    def load_embeddings(self, pdf_filename: str) -> Optional[np.ndarray]:
        """
        Load embeddings for one PDF.

        Args:
            pdf_filename: PDF filename.

        Returns:
            Embedding array, or None if unavailable.
        """
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            if pdf_filename not in index_data:
                logger.warning(f"PDF {pdf_filename} is not in the index")
                return None

            pdf_info = index_data[pdf_filename]
            embeddings_file = pdf_info.get("embeddings_file")

            if embeddings_file:
                full_path = DIRS["embedding_base"] / embeddings_file
                if full_path.exists():
                    return np.load(full_path)
                else:
                    logger.warning(f"Embeddings file does not exist: {full_path}")
                    return None
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return None

    def get_pdf_info(self, pdf_filename: str) -> Optional[Dict]:
        """
        Get index metadata for one PDF.

        Args:
            pdf_filename: PDF filename.

        Returns:
            Index metadata dictionary, or None.
        """
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            return index_data.get(pdf_filename)
        except Exception as e:
            logger.error(f"Failed to get PDF info: {e}")
            return None

    def get_chunk_by_id(self, pdf_filename: str, chunk_id: str) -> Optional[Dict]:
        """
        Get a specific chunk by chunk id.

        Args:
            pdf_filename: PDF filename.
            chunk_id: Chunk id.

        Returns:
            Chunk metadata dictionary, or None.
        """
        pdf_info = self.get_pdf_info(pdf_filename)
        if not pdf_info:
            return None

        for chunk in pdf_info.get("chunks", []):
            if chunk["chunk_id"] == chunk_id:
                return chunk
        return None

    def _load_model(self):
        """Load the Sentence Transformer model."""
        self.model = get_sentence_transformer_model()

    def _save_index(self, index_data: Dict):
        """Save the index to disk."""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Paragraph index metadata saved to {self.index_file}")


class DocumentEmbeddingBuilder:
    """Build document-level embeddings."""

    def __init__(self, load_model: bool = True):
        self.model = get_sentence_transformer_model() if load_model else None
        self.embeddings_cache = {}
        self.cache_file = DIRS["embedding_base"] / "document_embeddings.npy"
        self.metadata_file = DIRS["embedding_base"] / "document_embeddings_meta.json"

    @staticmethod
    def _metadata_from_knowledge_base(knowledge_base: List[Dict]) -> List[Dict[str, Any]]:
        metadata = []
        for entry in knowledge_base:
            entry_metadata = entry.get("metadata", {}) or {}
            metadata.append({
                "pdf_filename": entry_metadata.get("pdf_filename", ""),
                "title": entry_metadata.get("title", ""),
                "year": entry_metadata.get("year", ""),
                "authors": entry_metadata.get("authors", []),
            })
        return metadata

    def build_from_knowledge_base(self, knowledge_base: List[Dict]) -> Dict[str, Any]:
        """
        Build document embeddings from the structured knowledge base.

        Args:
            knowledge_base: Structured knowledge-base entries.

        Returns:
            Build summary.
        """
        texts = []

        for entry in knowledge_base:
            # Build a compact text representation for each document.
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
            # Drop empty parts.
            text_parts = [p for p in text_parts if p.strip()]
            texts.append(' '.join(text_parts))

        if not texts:
            logger.warning("No valid text available for document embeddings")
            return {
                "status": "error",
                "message": "No valid texts"
            }

        logger.info(f"Encoding {len(texts)} documents...")
        if self.model is None:
            self.model = get_sentence_transformer_model()
        embeddings = self.model.encode(texts, show_progress_bar=True)

        return self.save_embeddings_for_knowledge_base(knowledge_base, embeddings)

    def save_embeddings_for_knowledge_base(self, knowledge_base: List[Dict], embeddings: np.ndarray) -> Dict[str, Any]:
        """Save precomputed document embeddings and metadata."""
        metadata = self._metadata_from_knowledge_base(knowledge_base)
        np.save(self.cache_file, embeddings)
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self.embeddings_cache = {
            "embeddings": embeddings,
            "metadata": metadata
        }

        logger.info(f"Document embeddings saved to {self.cache_file}")

        return {
            "status": "completed",
            "num_documents": len(metadata),
            "embedding_dim": embeddings.shape[1],
            "cache_file": str(self.cache_file),
            "metadata_file": str(self.metadata_file)
        }

    def load_cached_embeddings(self) -> Optional[Dict]:
        """Load cached embeddings."""
        if self.embeddings_cache:
            return self.embeddings_cache

        if self.cache_file.exists() and self.metadata_file.exists():
            try:
                embeddings = np.load(self.cache_file)
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                self.embeddings_cache = {
                    "embeddings": embeddings,
                    "metadata": metadata
                }
                logger.info(f"Loaded embeddings for {len(metadata)} documents")
                return self.embeddings_cache
            except Exception as e:
                logger.error(f"Failed to load cached embeddings: {e}")

        return None

    def clear_cache(self):
        """Clear the embedding cache."""
        self.embeddings_cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        logger.info("Document embedding cache cleared")
