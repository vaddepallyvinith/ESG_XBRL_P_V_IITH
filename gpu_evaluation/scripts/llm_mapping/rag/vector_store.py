"""
Vector Database Interface and FAISS Implementation for RAG Pipeline.
Provides persistent vector indexing and k-NN retrieval to avoid recomputing embeddings.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np

from llm_mapping.rag.chunker import TextChunk
from llm_mapping.utils.logging_config import setup_logger

logger = setup_logger("vector_store")


@dataclass
class SearchResult:
    """SearchResult containing retrieved TextChunk, rank, and cosine similarity score."""
    chunk: TextChunk
    score: float
    rank: int


class VectorStore(ABC):
    """Abstract interface for Vector Store databases."""

    @abstractmethod
    def add_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]):
        """Indexes text chunks with vector embeddings."""
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 10) -> List[SearchResult]:
        """Performs vector similarity search."""
        pass

    @abstractmethod
    def save_index(self, persist_dir: Path):
        """Persists index and metadata to disk."""
        pass

    @abstractmethod
    def load_index(self, persist_dir: Path) -> bool:
        """Loads index and metadata from disk if available."""
        pass

    @abstractmethod
    def clear(self):
        """Clears index from memory."""
        pass


class FAISSVectorStore(VectorStore):
    """
    FAISS-based persistent vector database using Inner Product (Cosine Similarity on normalized vectors).
    """

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.chunks: List[TextChunk] = []

    def _init_index(self, dim: int):
        self.dimension = dim
        # IndexFlatIP calculates inner product (equals cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(dim)

    def add_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]):
        """
        Indexes text chunks with embeddings in FAISS.

        Args:
            chunks: List of TextChunk objects.
            embeddings: List of embedding vectors.
        """
        if not chunks or not embeddings:
            logger.warning("Empty chunks or embeddings passed to FAISSVectorStore.")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)}).")

        dim = len(embeddings[0])
        if self.index is None or self.dimension != dim:
            self._init_index(dim)

        matrix = np.array(embeddings, dtype=np.float32)

        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        matrix_norm = matrix / norms

        self.index.add(matrix_norm)
        self.chunks.extend(chunks)
        logger.info(f"Successfully added {len(chunks)} vectors to FAISS index. Total indexed: {self.index.ntotal}")

    def search(self, query_vector: List[float], top_k: int = 10) -> List[SearchResult]:
        """
        Performs k-NN search in FAISS index.

        Args:
            query_vector: Dense query embedding vector.
            top_k: Top K nearest neighbor count.

        Returns:
            List[SearchResult]: Search results ordered by similarity score.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty. Search returned 0 results.")
            return []

        q_arr = np.array([query_vector], dtype=np.float32)
        norm = np.linalg.norm(q_arr)
        if norm > 0:
            q_arr = q_arr / norm

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_arr, actual_k)

        results: List[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(score),
                    rank=rank,
                )
            )

        return results

    def save_index(self, persist_dir: Path):
        """
        Persists FAISS index binary and metadata JSON to disk.

        Args:
            persist_dir: Target directory path.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty. Skipping save.")
            return

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        index_file = persist_dir / "faiss.index"
        meta_file = persist_dir / "chunks_metadata.json"

        logger.info(f"Saving FAISS index ({self.index.ntotal} vectors) to {index_file}...")
        faiss.write_index(self.index, str(index_file))

        chunks_data = [c.to_dict() for c in self.chunks]
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"dimension": self.dimension, "chunks": chunks_data}, f, indent=2)

        logger.info(f"Persisted FAISS index and {len(chunks_data)} chunk records to {persist_dir}.")

    def load_index(self, persist_dir: Path) -> bool:
        """
        Loads FAISS binary index and chunk metadata from disk if present and dimension matches.

        Args:
            persist_dir: Directory containing index and metadata.

        Returns:
            bool: True if loaded successfully, False if missing, corrupted, or dimension mismatch.
        """
        persist_dir = Path(persist_dir)
        index_file = persist_dir / "faiss.index"
        meta_file = persist_dir / "chunks_metadata.json"

        if not index_file.exists() or not meta_file.exists():
            logger.info(f"No existing FAISS index found at {persist_dir}.")
            return False

        try:
            logger.info(f"Loading persistent FAISS index from {index_file}...")
            loaded_index = faiss.read_index(str(index_file))

            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            loaded_dim = meta.get("dimension", loaded_index.d)

            # Dimension mismatch check
            if loaded_dim != self.dimension:
                logger.warning(
                    f"Persistent FAISS index dimension ({loaded_dim}) does not match current model dimension ({self.dimension}). Re-indexing required."
                )
                return False

            self.index = loaded_index
            self.dimension = loaded_dim
            chunks_data = meta.get("chunks", [])

            self.chunks = []
            for cd in chunks_data:
                chunk = TextChunk(
                    chunk_id=cd["chunk_id"],
                    disclosure_id=cd["disclosure_id"],
                    title=cd["title"],
                    requirement=cd["requirement"],
                    description=cd["description"],
                    metadata=cd["metadata"],
                    framework=cd.get("framework", "GRI"),
                    section=cd.get("section", ""),
                    text=cd.get("text", ""),
                )
                self.chunks.append(chunk)

            logger.info(
                f"Successfully loaded FAISS index with {self.index.ntotal} vectors "
                f"and {len(self.chunks)} chunks from disk (Bypassed embedding recomputation)."
            )
            return True
        except Exception as e:
            logger.error(f"Error loading FAISS persistent index from {persist_dir}: {e}")
            return False

    def clear(self):
        """Clears FAISS index and chunk list."""
        self.index = None
        self.chunks.clear()
        logger.info("Cleared FAISSVectorStore.")
