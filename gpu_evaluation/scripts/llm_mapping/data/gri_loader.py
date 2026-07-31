"""
GRI Datasets Loader for Baseline LLM Mapping Framework.
Extracts disclosures from all processed GRI JSON files into structured Python objects.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_mapping.utils.logging_config import setup_logger

logger = setup_logger("gri_loader")


@dataclass
class GRIDisclosure:
    """Structured representation of a GRI Disclosure requirement."""
    id: str
    label: str
    text: str
    source_file: str
    standard_id: Optional[str] = None
    standard_title: Optional[str] = None
    requirements: List[Any] = field(default_factory=list)
    content: List[str] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert object to clean dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "text": self.text,
            "source_file": self.source_file,
            "standard_id": self.standard_id,
            "standard_title": self.standard_title,
            "requirements": self.requirements,
            "content": self.content,
            "tables": self.tables,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }

    def get_full_text(self) -> str:
        """Construct full text representation for embedding or retrieval."""
        parts = []
        if self.standard_title:
            parts.append(f"Standard: {self.standard_title}")
        if self.label:
            parts.append(f"Disclosure: {self.label}")
        if self.text and self.text != self.label:
            parts.append(f"Text: {self.text}")
        if self.requirements:
            req_texts = []
            for r in self.requirements:
                if isinstance(r, dict):
                    req_texts.append(r.get("text") or r.get("label") or str(r))
                else:
                    req_texts.append(str(r))
            parts.append("Requirements: " + " ".join(req_texts))
        if self.content:
            parts.append("Content: " + " ".join(self.content))
        return " | ".join(parts)


class GRILoader:
    """Loader for processed GRI dataset JSON files."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def load_disclosures(self) -> List[GRIDisclosure]:
        """
        Loads all GRI JSON files in data_dir (excluding BRSR files) and extracts disclosures.

        Returns:
            List[GRIDisclosure]: Structured GRI disclosure objects.
        """
        if not self.data_dir.exists():
            logger.error(f"GRI data directory not found at: {self.data_dir}")
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        gri_files = sorted([
            f for f in self.data_dir.glob("*.json")
            if "raw_Business" not in f.name and "BRSR_" not in f.name and f.name != "manifest.json"
        ])

        logger.info(f"Found {len(gri_files)} GRI dataset JSON files in {self.data_dir.name}.")

        disclosures: List[GRIDisclosure] = []

        for file_path in gri_files:
            try:
                file_disclosures = self._load_single_gri_file(file_path)
                disclosures.extend(file_disclosures)
            except Exception as e:
                logger.warning(f"Error parsing GRI file {file_path.name}: {e}")

        logger.info(f"Successfully extracted {len(disclosures)} GRI disclosures from {len(gri_files)} files.")
        return disclosures

    def _load_single_gri_file(self, file_path: Path) -> List[GRIDisclosure]:
        """Parses a single GRI JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        source_file = file_path.name
        standards = data.get("standards", [])
        extracted = []

        for std in standards:
            std_id = std.get("standard_id")
            std_title = std.get("title")

            for d in std.get("disclosures", []):
                disc_id = d.get("id") or d.get("disclosure_id")
                label = d.get("label") or d.get("title") or ""
                text = d.get("text") or label

                if not disc_id and not label:
                    continue

                unique_id = disc_id or f"GRI_{hash(label)}"

                extracted.append(
                    GRIDisclosure(
                        id=unique_id,
                        label=label,
                        text=text,
                        source_file=source_file,
                        standard_id=std_id,
                        standard_title=std_title,
                        requirements=d.get("requirements", []),
                        content=d.get("content", []),
                        tables=d.get("tables", []),
                        page_start=d.get("page_start"),
                        page_end=d.get("page_end"),
                    )
                )

        return extracted
