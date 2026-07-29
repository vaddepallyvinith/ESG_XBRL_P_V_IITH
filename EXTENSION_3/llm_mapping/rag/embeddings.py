"""
Embedding Generator Interface and Implementations for RAG Pipeline.
Supports configurable sentence-transformers embedding models with default BAAI/bge-large-en-v1.5.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import time
import numpy as np

from llm_mapping.utils.logging_config import setup_logger

logger = setup_logger("embeddings")


class EmbeddingGenerator(ABC):
    """Abstract Base Class for dense vector embedding generators."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the embedding model name."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Returns the dimension of output embedding vectors."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embeds a single text string into a float vector."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embeds a list of text strings into float vectors."""
        pass


class SentenceTransformerEmbeddingGenerator(EmbeddingGenerator):
    """
    Embedding generator powered by SentenceTransformers.
    Defaults to BAAI/bge-large-en-v1.5.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: Optional[str] = None):
        self._model_name = model_name
        self.device = device
        self._model = None
        self._dim: Optional[int] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._load_model()
            if self._model:
                self._dim = self._model.get_sentence_embedding_dimension()
            else:
                self._dim = 1024  # Default dimension for BAAI/bge-large-en-v1.5
        return self._dim

    def _load_model(self):
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            try:
                import os
                import torch
                from sentence_transformers import SentenceTransformer

                # Optimize CPU multi-threading for PyTorch
                num_cores = os.cpu_count() or 8
                torch.set_num_threads(min(16, num_cores))

                device_to_use = self.device
                if not device_to_use:
                    device_to_use = "cuda" if torch.cuda.is_available() else "cpu"

                logger.info(f"Loading SentenceTransformer model '{self._model_name}' on device '{device_to_use}' (threads={torch.get_num_threads()})...")
                start_time = time.time()
                self._model = SentenceTransformer(self._model_name, device=device_to_use)
                elapsed = time.time() - start_time
                self._dim = self._model.get_embedding_dimension() if hasattr(self._model, "get_embedding_dimension") else self._model.get_sentence_embedding_dimension()
                logger.info(
                    f"Successfully loaded '{self._model_name}' (dim={self._dim}) in {elapsed:.2f}s."
                )
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model '{self._model_name}': {e}")
                raise RuntimeError(f"Embedding model loading failed: {e}")

    def embed_text(self, text: str) -> List[float]:
        self._load_model()
        if not text or not text.strip():
            return [0.0] * self.embedding_dim

        import torch
        with torch.inference_mode():
            vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        self._load_model()
        if not texts:
            return []

        import sys
        import torch

        logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})...")
        sys.stdout.flush()
        start_time = time.time()

        all_vecs = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                b_start = time.time()

                vecs = self._model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                all_vecs.extend(vecs.tolist())
                b_elapsed = time.time() - b_start
                msg = f"  Processed batch [{batch_num}/{total_batches}] ({len(batch_texts)} items) in {b_elapsed:.2f}s."
                logger.info(msg)
                print(msg, flush=True)

        elapsed = time.time() - start_time
        comp_msg = f"Batch embedding completed in {elapsed:.2f}s ({len(texts)/elapsed:.1f} texts/sec)."
        logger.info(comp_msg)
        print(comp_msg, flush=True)
        return all_vecs


class DummyEmbeddingGenerator(EmbeddingGenerator):
    """Fallback dummy generator for testing without heavy model loads."""

    def __init__(self, model_name: str = "dummy-1024", dimension: int = 1024):
        self._model_name = model_name
        self._dim = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.randn(self._dim)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]
