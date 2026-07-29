"""
Configuration settings for the Baseline LLM Mapping Framework.
Defines paths, model settings, RAG parameters, and baseline evaluation options.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathConfig:
    """Paths configuration for processed datasets and outputs."""
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_processed_dir: Path = base_dir.parent / "data" / "processed"
    brsr_filename: str = "raw_Business responsibility and sustainability reporting by listed entitiesAnnexure1_p.json"
    output_dir: Path = base_dir / "output"
    vector_store_dir: Path = output_dir / "vector_store"

    @property
    def brsr_file_path(self) -> Path:
        return self.data_processed_dir / self.brsr_filename


@dataclass
class RAGConfig:
    """RAG & Embedding configuration parameters."""
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    similarity_threshold: float = 0.3
    persist_index: bool = True


@dataclass
class LLMConfig:
    """Ollama LLM client configuration parameters."""
    host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model_name: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 1024
    request_timeout: int = 120


@dataclass
class SystemSettings:
    """Central configuration container for the baseline mapping framework."""
    paths: PathConfig = field(default_factory=PathConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def __post_init__(self):
        # Ensure directories exist
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.vector_store_dir.mkdir(parents=True, exist_ok=True)


# Default global settings instance
settings = SystemSettings()
