"""
BRSR Dataset Loader for Baseline LLM Mapping Framework.
Extracts disclosures from raw BRSR JSON datasets into structured Python objects.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_mapping.utils.logging_config import setup_logger

logger = setup_logger("brsr_loader")


@dataclass
class BRSRDisclosure:
    """Structured representation of a BRSR Disclosure requirement."""
    id: str
    label: str
    text: str
    section_id: Optional[str] = None
    section_label: Optional[str] = None
    principle_num: Optional[str] = None
    principle_label: Optional[str] = None
    indicator_group: Optional[str] = None
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
            "section_id": self.section_id,
            "section_label": self.section_label,
            "principle_num": self.principle_num,
            "principle_label": self.principle_label,
            "indicator_group": self.indicator_group,
            "content": self.content,
            "tables": self.tables,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }

    def get_full_text(self) -> str:
        """Construct full text representation for embedding or retrieval."""
        parts = []
        if self.section_label:
            parts.append(f"Section: {self.section_label}")
        if self.principle_label:
            parts.append(f"Principle: {self.principle_label}")
        if self.indicator_group:
            parts.append(f"Group: {self.indicator_group}")
        if self.label:
            parts.append(f"Label: {self.label}")
        if self.text and self.text != self.label:
            parts.append(f"Text: {self.text}")
        if self.content:
            parts.append("Content: " + " ".join(self.content))
        return " | ".join(parts)


class BRSRLoader:
    """Loader for BRSR dataset JSON file."""

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)

    def load_disclosures(self) -> List[BRSRDisclosure]:
        """
        Loads the BRSR JSON file and extracts all disclosures recursively.

        Returns:
            List[BRSRDisclosure]: Structured BRSR disclosure objects.
        """
        if not self.file_path.exists():
            logger.error(f"BRSR file not found at: {self.file_path}")
            raise FileNotFoundError(f"BRSR file not found: {self.file_path}")

        logger.info(f"Loading BRSR dataset from {self.file_path.name}...")
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        disclosures: List[BRSRDisclosure] = []
        seen_ids = set()

        sections = data.get("sections", [])
        logger.info(f"Found {len(sections)} main sections in BRSR JSON.")

        for sec in sections:
            sec_id = sec.get("section_id")
            sec_label = sec.get("label")

            # 1. Direct section disclosures
            for d in sec.get("disclosures", []):
                disc_obj = self._parse_disclosure_dict(d, sec_id=sec_id, sec_label=sec_label)
                if disc_obj:
                    disclosures.append(disc_obj)

            # 2. Section principles
            for p in sec.get("principles", []):
                p_num = p.get("principle_num")
                p_label = p.get("label")

                # Direct principle disclosures
                for d in p.get("disclosures", []):
                    disc_obj = self._parse_disclosure_dict(
                        d, sec_id=sec_id, sec_label=sec_label, p_num=p_num, p_label=p_label
                    )
                    if disc_obj:
                        disclosures.append(disc_obj)

                # Principle indicator groups disclosures
                for ig in p.get("indicator_groups", []):
                    ig_label = ig.get("label")
                    for d in ig.get("disclosures", []):
                        disc_obj = self._parse_disclosure_dict(
                            d,
                            sec_id=sec_id,
                            sec_label=sec_label,
                            p_num=p_num,
                            p_label=p_label,
                            ig_label=ig_label,
                        )
                        if disc_obj:
                            disclosures.append(disc_obj)

        logger.info(f"Successfully extracted {len(disclosures)} BRSR disclosures.")
        return disclosures

    def _parse_disclosure_dict(
        self,
        d: Dict[str, Any],
        sec_id: Optional[str] = None,
        sec_label: Optional[str] = None,
        p_num: Optional[str] = None,
        p_label: Optional[str] = None,
        ig_label: Optional[str] = None,
    ) -> Optional[BRSRDisclosure]:
        """Helper to create BRSRDisclosure object from raw dictionary."""
        disc_id = d.get("id") or d.get("disclosure_id")
        label = d.get("label") or d.get("title") or ""
        text = d.get("text") or label

        if not disc_id and not label:
            return None

        # Build clean compound ID if generic
        unique_id = disc_id or f"BRSR_{hash(label)}"
        if p_num and not unique_id.startswith("P"):
            unique_id = f"P{p_num}_{unique_id}"

        return BRSRDisclosure(
            id=unique_id,
            label=label,
            text=text,
            section_id=sec_id,
            section_label=sec_label,
            principle_num=p_num,
            principle_label=p_label,
            indicator_group=ig_label,
            content=d.get("content", []),
            tables=d.get("tables", []),
            page_start=d.get("page_start"),
            page_end=d.get("page_end"),
        )
