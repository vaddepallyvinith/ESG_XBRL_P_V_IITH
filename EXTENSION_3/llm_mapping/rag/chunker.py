"""
RAG Text Chunker for Baseline LLM Mapping Framework.
Implements intelligent disclosure chunking containing Disclosure ID, Title, Requirement,
Description, Metadata, Framework, and Section attributes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from llm_mapping.data.gri_loader import GRIDisclosure


@dataclass
class TextChunk:
    """
    Structured representation of a single document text chunk for vector indexing.

    Attributes:
        chunk_id: Unique chunk identifier.
        disclosure_id: Parent disclosure ID (e.g. 'Disclosure 201-1').
        title: Disclosure title/label.
        requirement: Reporting requirement text/list.
        description: Detailed content/description.
        metadata: Metadata dictionary (file source, page numbers, etc.).
        framework: Framework name (e.g. 'GRI').
        section: Standard title / section name (e.g. 'GRI 201: Economic Performance 2016').
        text: Formatted string representation for embedding generation.
    """
    chunk_id: str
    disclosure_id: str
    title: str
    requirement: str
    description: str
    metadata: Dict[str, Any]
    framework: str = "GRI"
    section: str = ""
    text: str = ""

    def __post_init__(self):
        if not self.text:
            self.text = self.to_embedding_text()

    def to_embedding_text(self) -> str:
        """Constructs rich structured text for dense embedding model input."""
        parts = []
        if self.framework:
            parts.append(f"Framework: {self.framework}")
        if self.section:
            parts.append(f"Section/Standard: {self.section}")
        if self.disclosure_id:
            parts.append(f"Disclosure ID: {self.disclosure_id}")
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.requirement:
            parts.append(f"Requirement: {self.requirement}")
        if self.description:
            parts.append(f"Description: {self.description}")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk object to clean dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "disclosure_id": self.disclosure_id,
            "title": self.title,
            "requirement": self.requirement,
            "description": self.description,
            "metadata": self.metadata,
            "framework": self.framework,
            "section": self.section,
            "text": self.text,
        }


class GRIIntelligentChunker:
    """Intelligent chunker for GRI disclosure objects."""

    def __init__(self, max_words_per_chunk: int = 500, overlap_words: int = 50):
        self.max_words_per_chunk = max_words_per_chunk
        self.overlap_words = overlap_words

    def create_chunks_from_gri(self, gri: GRIDisclosure) -> List[TextChunk]:
        """
        Processes a single GRIDisclosure object into one or more structured TextChunks.

        Args:
            gri: Target GRIDisclosure object.

        Returns:
            List[TextChunk]: Generated intelligent text chunks.
        """
        # Format requirements
        req_texts = []
        for req in gri.requirements:
            if isinstance(req, dict):
                req_text = req.get("text") or req.get("label") or str(req)
            else:
                req_text = str(req)
            if req_text.strip():
                req_texts.append(req_text.strip())
        formatted_requirement = " ".join(req_texts)

        # Format content/description
        desc_texts = [c.strip() for c in gri.content if isinstance(c, str) and c.strip()]
        formatted_description = " ".join(desc_texts)

        # Metadata dictionary
        metadata = {
            "source_file": gri.source_file,
            "standard_id": gri.standard_id or "",
            "page_start": gri.page_start,
            "page_end": gri.page_end,
            "tables_count": len(gri.tables),
        }

        section = gri.standard_title or gri.standard_id or "GRI Standard"
        title = gri.label or gri.text or gri.id

        full_desc_words = formatted_description.split()

        # If description is short or empty, create single comprehensive chunk
        if len(full_desc_words) <= self.max_words_per_chunk:
            return [
                TextChunk(
                    chunk_id=f"{gri.id}_chunk_0",
                    disclosure_id=gri.id,
                    title=title,
                    requirement=formatted_requirement,
                    description=formatted_description,
                    metadata=metadata,
                    framework="GRI",
                    section=section,
                )
            ]

        # Multi-chunk splitting for long descriptions
        chunks: List[TextChunk] = []
        step = self.max_words_per_chunk - self.overlap_words
        idx = 0
        chunk_count = 0

        while idx < len(full_desc_words):
            chunk_desc_words = full_desc_words[idx : idx + self.max_words_per_chunk]
            chunk_desc = " ".join(chunk_desc_words)
            chunks.append(
                TextChunk(
                    chunk_id=f"{gri.id}_chunk_{chunk_count}",
                    disclosure_id=gri.id,
                    title=title,
                    requirement=formatted_requirement,
                    description=chunk_desc,
                    metadata=metadata,
                    framework="GRI",
                    section=section,
                )
            )
            idx += step
            chunk_count += 1

        return chunks

    def chunk_all_gri_disclosures(self, gri_disclosures: List[GRIDisclosure]) -> List[TextChunk]:
        """
        Batch process a list of GRI disclosures into intelligent TextChunks.

        Args:
            gri_disclosures: List of GRI disclosure objects.

        Returns:
            List[TextChunk]: Complete list of intelligent text chunks.
        """
        all_chunks: List[TextChunk] = []
        for gri in gri_disclosures:
            chunks = self.create_chunks_from_gri(gri)
            all_chunks.extend(chunks)
        return all_chunks
