"""
High-level RAG Retriever for Baseline LLM Mapping Framework.
Manages GRI disclosure chunk indexing, FAISS persistence, query vector embedding,
configurable top-K retrieval, and score/latency logging.
"""

from dataclasses import dataclass, field
from pathlib import Path

import time
from typing import Any, Dict, List, Optional

from llm_mapping.config.settings import settings
from llm_mapping.data.brsr_loader import BRSRDisclosure
from llm_mapping.data.gri_loader import GRIDisclosure
from llm_mapping.rag.chunker import GRIIntelligentChunker, TextChunk
from llm_mapping.rag.embeddings import EmbeddingGenerator, SentenceTransformerEmbeddingGenerator
from llm_mapping.rag.vector_store import VectorStore, FAISSVectorStore, SearchResult
from llm_mapping.utils.logging_config import setup_logger

logger = setup_logger("retriever")


@dataclass
class RetrievedCandidate:
    """Retrieved candidate disclosure match item."""
    rank: int
    score: float
    chunk: TextChunk
    gri_id: str
    title: str
    standard: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "gri_id": self.gri_id,
            "title": self.title,
            "standard": self.standard,
            "chunk_id": self.chunk.chunk_id,
            "requirement": self.chunk.requirement,
            "description": self.chunk.description,
        }


@dataclass
class RetrievalResult:
    """Structured container for single BRSR query retrieval result."""
    brsr_id: str
    brsr_label: str
    top_k: int
    candidates: List[RetrievedCandidate]
    retrieval_latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brsr_id": self.brsr_id,
            "brsr_label": self.brsr_label,
            "top_k": self.top_k,
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "candidates": [c.to_dict() for c in self.candidates],
            "metadata": self.metadata,
        }


class RAGRetriever:
    """
    RAG Retriever orchestrating Intelligent Chunking, Embedding Generation,
    Persistent FAISS Indexing, and Top-K Candidate Retrieval.
    """

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None,
        chunker: Optional[GRIIntelligentChunker] = None,
        persist_dir: Optional[Path] = None,
    ):
        self.chunker = chunker or GRIIntelligentChunker()
        self.embedding_generator = (
            embedding_generator
            or SentenceTransformerEmbeddingGenerator(model_name=settings.rag.embedding_model_name)
        )
        self.vector_store = vector_store or FAISSVectorStore(
            dimension=self.embedding_generator.embedding_dim
        )
        self.persist_dir = Path(persist_dir or settings.paths.vector_store_dir)
        self._gri_lookup: Dict[str, GRIDisclosure] = {}

    def index_gri_disclosures(self, gri_disclosures: List[GRIDisclosure], force_reindex: bool = False):
        """
        Loads persistent index if available on disk, or indexes GRI disclosures from scratch.

        Args:
            gri_disclosures: List of raw GRI disclosures.
            force_reindex: If True, forces re-chunking and re-embedding.
        """
        # Maintain lookup map for full objects
        for gri in gri_disclosures:
            self._gri_lookup[gri.id] = gri

        # Try loading index from disk if not forcing reindex
        if not force_reindex and settings.rag.persist_index:
            loaded = self.vector_store.load_index(self.persist_dir)
            if loaded:
                logger.info(
                    f"Successfully loaded persistent vector index from {self.persist_dir}. "
                    "Skipping embedding re-computation."
                )
                return

        logger.info("Initializing vector indexing pipeline for GRI disclosures...")
        start_time = time.time()

        # 1. Intelligent Chunking
        chunks = self.chunker.chunk_all_gri_disclosures(gri_disclosures)
        logger.info(f"Intelligent Chunker generated {len(chunks)} text chunks from {len(gri_disclosures)} disclosures.")

        # 2. Batch Embedding Generation
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedding_generator.embed_batch(chunk_texts, batch_size=16)

        # 3. Add to Vector Store
        self.vector_store.clear()
        self.vector_store.add_chunks(chunks, embeddings)

        # 4. Persist to Disk
        if settings.rag.persist_index:
            self.vector_store.save_index(self.persist_dir)

        elapsed = time.time() - start_time
        logger.info(f"GRI Disclosure Indexing Pipeline finished in {elapsed:.2f}s.")

    def retrieve(self, brsr_disclosure: BRSRDisclosure, top_k: int = 5) -> RetrievalResult:
        """
        Retrieves Top-K candidate GRI disclosures for a given BRSR disclosure query.

        Args:
            brsr_disclosure: Target BRSR disclosure query object.
            top_k: Number of candidates to retrieve (e.g. 3, 5, 10).

        Returns:
            RetrievalResult: Complete retrieval result containing matches, scores, and latency.
        """
        start_time = time.perf_counter()

        # Build query embedding
        query_text = brsr_disclosure.get_full_text()
        query_vec = self.embedding_generator.embed_text(query_text)

        # Perform k-NN search in vector store
        search_results = self.vector_store.search(query_vec, top_k=top_k)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        candidates: List[RetrievedCandidate] = []
        retrieved_ids = []
        retrieved_scores = []

        for sr in search_results:
            c_chunk = sr.chunk
            cand = RetrievedCandidate(
                rank=sr.rank,
                score=sr.score,
                chunk=c_chunk,
                gri_id=c_chunk.disclosure_id,
                title=c_chunk.title,
                standard=c_chunk.section,
            )
            candidates.append(cand)
            retrieved_ids.append(c_chunk.disclosure_id)
            retrieved_scores.append(round(sr.score, 4))

        # Log detailed metrics as required
        logger.info(
            f"Retrieval Query BRSR ID [{brsr_disclosure.id}] | Top-{top_k} | Latency: {latency_ms:.2f}ms | "
            f"Retrieved IDs: {retrieved_ids} | Scores: {retrieved_scores}"
        )

        return RetrievalResult(
            brsr_id=brsr_disclosure.id,
            brsr_label=brsr_disclosure.label,
            top_k=top_k,
            candidates=candidates,
            retrieval_latency_ms=latency_ms,
            metadata={"query_text_length": len(query_text)},
        )
