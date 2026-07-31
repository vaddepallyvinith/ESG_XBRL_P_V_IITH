"""
heading_detector.py – Detect document headings from font metadata
=================================================================

Uses font size, boldness, and BRSR-specific patterns to classify
text blocks into heading levels and detect BRSR structural markers.
"""

import re
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass
from parser.pdf_parser import TextBlock, PageContent, DocumentContent

logger = logging.getLogger(__name__)


# ─── Heading Level Constants ─────────────────────────────────────────────────

HEADING_LEVELS = {
    "h1": 1,  # "SECTION A", "SECTION B", "SECTION C"
    "h2": 2,  # "PRINCIPLE 1", "Principle 6"
    "h3": 3,  # "Essential Indicators", "Leadership Indicators"
    "h4": 4,  # Individual question/disclosure headings
    "body": 0,
}

# BRSR structural patterns
BRSR_PATTERNS = {
    "section": re.compile(r"(?i)^SECTION\s+[A-C]\b"),
    "principle": re.compile(r"(?i)^PRINCIPLE\s+(\d)\b"),
    "essential": re.compile(r"(?i)Essential\s+Indicators?"),
    "leadership": re.compile(r"(?i)Leadership\s+Indicators?"),
    "question_num": re.compile(r"^(\d{1,2})\.\s+"),
    "sub_question": re.compile(r"^[a-z]\)\s+|^\([a-z]\)\s+"),
    "section_a_fields": re.compile(
        r"(?i)(Details of the listed entity|"
        r"Products/Services|"
        r"Operations|"
        r"Employees|"
        r"Holding.*subsidiary|"
        r"CSR Details|"
        r"Transparency)"
    ),
}

# GRI structural patterns
GRI_PATTERNS = {
    "standard_title": re.compile(r"^GRI\s+\d{1,3}:\s+.*"),
    "disclosure": re.compile(r"^Disclosure\s+\d{1,3}-\d{1,2}.*"),
    "subsection": re.compile(r"^(REQUIREMENTS|RECOMMENDATIONS|GUIDANCE)\b"),
    "topic": re.compile(r"^\d+\.\s+Topic\s+management\s+disclosures|^\d+\.\s+Topic\s+disclosures|^Topic\s+\d{1,2}\.\d{1,2}.*"),
}


# ─── Heading Detection Result ────────────────────────────────────────────────

@dataclass
class HeadingInfo:
    """Classification result for a text block."""
    level: int = 0           # 0=body, 1=h1, 2=h2, 3=h3, 4=h4
    heading_type: str = ""   # "section", "principle", "essential", "leadership", "question", ""
    section_label: str = ""  # "SECTION A", "PRINCIPLE 6", etc.
    text: str = ""
    page_num: int = 0
    is_heading: bool = False


# ─── Heading Detector ────────────────────────────────────────────────────────

class HeadingDetector:
    """Detect and classify headings in BRSR documents."""

    def __init__(self, font_thresholds=None):
        self.thresholds = font_thresholds or {"h1": 16.0, "h2": 13.0, "h3": 11.0, "body": 9.0}

    def classify_block(self, block: TextBlock, is_gri: bool = False) -> HeadingInfo:
        """Classify a single text block as heading or body text."""
        text = block.text.strip()
        if not text:
            return HeadingInfo(text=text, page_num=block.page_num)

        font_level = self._classify_by_font(block)

        # 1. Pattern-based detection (highest priority)
        if is_gri:
            pattern_result = self._match_gri_pattern(text, block.page_num, font_level)
        else:
            pattern_result = self._match_brsr_pattern(text, block.page_num)
            
        if pattern_result and pattern_result.is_heading:
            return pattern_result

        # 2. Font-size based detection
        if font_level > 0:
            return HeadingInfo(
                level=font_level,
                heading_type="font_based",
                section_label=text[:80],
                text=text,
                page_num=block.page_num,
                is_heading=True,
            )

        # 3. Body text
        return HeadingInfo(text=text, page_num=block.page_num)

    def _match_gri_pattern(self, text: str, page_num: int, font_level: int) -> Optional[HeadingInfo]:
        """Check for GRI-specific structural patterns."""
        if GRI_PATTERNS["standard_title"].match(text):
            # To avoid headers/footers, it must be on page 1-3 OR have a large font size
            if page_num <= 3 or font_level in (1, 2):
                return HeadingInfo(level=1, heading_type="gri_standard", section_label=text[:50].strip(), text=text, page_num=page_num, is_heading=True)
        if GRI_PATTERNS["disclosure"].match(text):
            return HeadingInfo(level=2, heading_type="gri_disclosure", section_label=text[:50].strip(), text=text, page_num=page_num, is_heading=True)
        m = GRI_PATTERNS["subsection"].match(text)
        if m:
            return HeadingInfo(level=3, heading_type="gri_subsection", section_label=m.group(1), text=text, page_num=page_num, is_heading=True)
        if GRI_PATTERNS["topic"].match(text):
            return HeadingInfo(level=2, heading_type="gri_topic", section_label=text[:50].strip(), text=text, page_num=page_num, is_heading=True)
        return None

    def _match_brsr_pattern(self, text: str, page_num: int) -> Optional[HeadingInfo]:
        """Check for BRSR-specific structural patterns."""
        # Section A/B/C
        m = BRSR_PATTERNS["section"].match(text)
        if m:
            return HeadingInfo(
                level=1, heading_type="section",
                section_label=text[:30].strip(), text=text,
                page_num=page_num, is_heading=True,
            )

        # Principle N
        m = BRSR_PATTERNS["principle"].match(text)
        if m:
            return HeadingInfo(
                level=2, heading_type="principle",
                section_label=f"Principle{m.group(1)}", text=text,
                page_num=page_num, is_heading=True,
            )

        # Essential/Leadership Indicators
        if BRSR_PATTERNS["essential"].search(text):
            return HeadingInfo(
                level=3, heading_type="essential",
                section_label="Essential Indicators", text=text,
                page_num=page_num, is_heading=True,
            )
        if BRSR_PATTERNS["leadership"].search(text):
            return HeadingInfo(
                level=3, heading_type="leadership",
                section_label="Leadership Indicators", text=text,
                page_num=page_num, is_heading=True,
            )

        # Section A fields
        if BRSR_PATTERNS["section_a_fields"].search(text) and len(text) < 100:
            return HeadingInfo(
                level=3, heading_type="section_field",
                section_label=text[:60].strip(), text=text,
                page_num=page_num, is_heading=True,
            )

        # Numbered questions (e.g., "1. Does the entity have...")
        m = BRSR_PATTERNS["question_num"].match(text)
        if m and len(text) > 20:
            return HeadingInfo(
                level=4, heading_type="question",
                section_label=f"Q{m.group(1)}", text=text,
                page_num=page_num, is_heading=True,
            )

        return None

    def _classify_by_font(self, block: TextBlock) -> int:
        """Classify heading level based on font size and boldness."""
        size = block.avg_font_size

        if size >= self.thresholds["h1"] and block.is_bold:
            return 1
        if size >= self.thresholds["h2"] and block.is_bold:
            return 2
        if size >= self.thresholds["h3"] and block.is_bold:
            return 3
        if block.is_bold and block.line_count <= 2 and len(block.text) < 120:
            return 4

        return 0  # body text

    def detect_all(self, document: DocumentContent, is_gri: bool = False) -> List[HeadingInfo]:
        """Detect all headings in a complete document."""
        headings = []
        for page in document.pages:
            for block in page.blocks:
                if block.block_type == "text":
                    info = self.classify_block(block, is_gri=is_gri)
                    if info.is_heading:
                        headings.append(info)
        logger.info(f"Detected {len(headings)} headings in {document.file_name}")
        return headings
