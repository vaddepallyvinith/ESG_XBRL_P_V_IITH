"""
section_segmenter.py – BRSR Document Section Segmentation
==========================================================

Takes parsed document content + detected headings and builds a
hierarchical tree:

  Document
   └─ Section (A/B/C)
       └─ Principle (1-9)  [Section C only]
           └─ IndicatorGroup (Essential/Leadership)
               └─ Disclosure (individual questions/metrics)
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from parser.pdf_parser import DocumentContent, TextBlock, TableBlock
from parser.heading_detector import HeadingDetector, HeadingInfo
from parser.table_extractor import normalize_table, detect_metric_table

logger = logging.getLogger(__name__)


@dataclass
class DisclosureNode:
    """A single disclosure/question with its content."""
    id: str = ""
    label: str = ""
    text: str = ""
    tables: List[Dict[str, Any]] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    content_blocks: List[str] = field(default_factory=list)
    metric: Optional[str] = None
    unit: Optional[str] = None
    datatype: Optional[str] = None
    applicability: Optional[str] = None


@dataclass
class IndicatorGroup:
    """Essential or Leadership indicator group."""
    group_type: str = ""  # "Essential" or "Leadership"
    disclosures: List[DisclosureNode] = field(default_factory=list)
    content_blocks: List[str] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0


@dataclass
class PrincipleNode:
    """A BRSR Principle (1-9)."""
    principle_num: int = 0
    label: str = ""
    text: str = ""
    indicator_groups: List[IndicatorGroup] = field(default_factory=list)
    disclosures: List[DisclosureNode] = field(default_factory=list)
    content_blocks: List[str] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0


@dataclass
class SectionNode:
    """A major BRSR section (A, B, or C)."""
    section_id: str = ""
    label: str = ""
    page_start: int = 0
    page_end: int = 0
    principles: List[PrincipleNode] = field(default_factory=list)
    disclosures: List[DisclosureNode] = field(default_factory=list)
    content_blocks: List[str] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DocumentTree:
    """Complete hierarchical structure of a parsed BRSR document."""
    framework: str = "BRSR"
    framework_full_name: str = "Business Responsibility and Sustainability Report"
    issuer: str = "SEBI"
    company: str = ""
    fiscal_year: str = ""
    source_file: str = ""
    total_pages: int = 0
    sections: List[SectionNode] = field(default_factory=list)
    content_blocks: List[str] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)


class SectionSegmenter:
    """Segment a parsed BRSR document into hierarchical sections."""

    def __init__(self):
        self.heading_detector = HeadingDetector()

    def segment(self, document: DocumentContent) -> DocumentTree:
        """Build the document tree from parsed content."""
        tree = DocumentTree(
            company=document.company,
            fiscal_year=document.fiscal_year,
            source_file=document.file_name,
            total_pages=document.total_pages,
        )

        # Flatten all blocks with their heading classifications
        classified_blocks = self._classify_all_blocks(document)

        # Build hierarchy
        current_section: Optional[SectionNode] = None
        current_principle: Optional[PrincipleNode] = None
        current_indicator_group: Optional[IndicatorGroup] = None
        current_disclosure: Optional[DisclosureNode] = None

        skip_blocks = False

        for block_info in classified_blocks:
            block = block_info["block"]
            heading = block_info["heading"]
            page_num = block_info["page_num"]

            if heading and heading.is_heading:
                if heading.heading_type == "section":
                    skip_blocks = False
                    # New section
                    section_id = re.search(r"[A-C]", heading.section_label)
                    current_section = SectionNode(
                        section_id=section_id.group() if section_id else "",
                        label=heading.text[:100],
                        page_start=page_num,
                        page_end=page_num,
                    )
                    tree.sections.append(current_section)
                    current_principle = None
                    current_indicator_group = None
                    current_disclosure = None

                elif heading.heading_type == "principle":
                    num_match = re.search(r"(\d)", heading.section_label)
                    p_num = int(num_match.group(1)) if num_match else 0
                    
                    # Environmental Filtering: Only parse Principle 6
                    if p_num != 6:
                        skip_blocks = True
                        current_principle = None
                        current_indicator_group = None
                        current_disclosure = None
                        continue
                        
                    skip_blocks = False
                    current_principle = PrincipleNode(
                        principle_num=int(num_match.group(1)) if num_match else 0,
                        label=heading.section_label,
                        text=heading.text,
                        page_start=page_num,
                        page_end=page_num,
                    )
                    if current_section:
                        current_section.principles.append(current_principle)
                    current_indicator_group = None
                    current_disclosure = None

                elif heading.heading_type in ("essential", "leadership"):
                    current_indicator_group = IndicatorGroup(
                        group_type="Essential" if heading.heading_type == "essential" else "Leadership",
                        page_start=page_num,
                        page_end=page_num,
                    )
                    if current_principle:
                        current_principle.indicator_groups.append(current_indicator_group)
                    current_disclosure = None

                elif heading.heading_type == "question" or heading.level == 4:
                    current_disclosure = DisclosureNode(
                        id=heading.section_label,
                        label=heading.text[:120],
                        text=heading.text,
                        page_start=page_num,
                        page_end=page_num,
                    )
                    self._attach_disclosure(
                        current_disclosure, current_indicator_group,
                        current_principle, current_section
                    )
                elif skip_blocks:
                    continue

            else:
                if skip_blocks:
                    continue
                # Content block — attach to deepest active node
                target = current_disclosure or current_indicator_group or current_principle or current_section or tree
                if target:
                    if block.block_type == "text":
                        if hasattr(target, 'content_blocks'):
                            target.content_blocks.append(block.text)
                    elif block.block_type == "table":
                        table_data = normalize_table(block)
                        if table_data and hasattr(target, 'tables'):
                            target.tables.append(table_data)
                    
                    # Update page_end on target and parent nodes
                    if hasattr(target, 'page_end'):
                        target.page_end = max(target.page_end, page_num)
                    if current_disclosure:
                        current_disclosure.page_end = max(current_disclosure.page_end, page_num)
                    if current_indicator_group:
                        current_indicator_group.page_end = max(current_indicator_group.page_end, page_num)
                    if current_principle:
                        current_principle.page_end = max(current_principle.page_end, page_num)
                    if current_section:
                        current_section.page_end = max(current_section.page_end, page_num)

        # Stats
        total_disclosures = sum(
            len(s.disclosures) + sum(
                len(p.disclosures) + sum(
                    len(ig.disclosures) for ig in p.indicator_groups
                ) for p in s.principles
            ) for s in tree.sections
        )
        logger.info(
            f"Segmented {document.file_name}: "
            f"{len(tree.sections)} sections, "
            f"{total_disclosures} disclosures"
        )
        return tree

    def _classify_all_blocks(self, document: DocumentContent) -> List[Dict]:
        """Classify every block in the document with heading info."""
        result = []
        for page in document.pages:
            for block in page.blocks:
                heading = None
                if block.block_type == "text":
                    heading = self.heading_detector.classify_block(block)

                result.append({
                    "block": block,
                    "heading": heading,
                    "page_num": page.page_num,
                })
        return result

    def _attach_disclosure(
        self,
        disclosure: DisclosureNode,
        indicator_group: Optional[IndicatorGroup],
        principle: Optional[PrincipleNode],
        section: Optional[SectionNode],
    ):
        """Attach a disclosure to the deepest available parent."""
        if indicator_group:
            indicator_group.disclosures.append(disclosure)
        elif principle:
            principle.disclosures.append(disclosure)
        elif section:
            section.disclosures.append(disclosure)


@dataclass
class GRIRequirementNode:
    """A requirement, recommendation or guidance block in GRI."""
    type: str = "" # "REQUIREMENTS", "RECOMMENDATIONS", "GUIDANCE"
    text: str = ""
    content_blocks: List[str] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    metric: Optional[str] = None
    unit: Optional[str] = None
    datatype: Optional[str] = None
    applicability: Optional[str] = None

@dataclass
class GRIDisclosureNode:
    """A GRI Disclosure (e.g. 302-1)"""
    id: str = ""
    label: str = ""
    text: str = ""
    requirements: List[GRIRequirementNode] = field(default_factory=list)
    content_blocks: List[str] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    metric: Optional[str] = None
    unit: Optional[str] = None
    datatype: Optional[str] = None
    applicability: Optional[str] = None

@dataclass
class GRIStandardNode:
    """A GRI Standard (e.g. GRI 302: Energy 2016)"""
    standard_id: str = ""
    title: str = ""
    disclosures: List[GRIDisclosureNode] = field(default_factory=list)
    content_blocks: List[str] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0

@dataclass
class GRIDocumentTree:
    """Complete hierarchical structure of a GRI standard."""
    source_file: str = ""
    total_pages: int = 0
    standards: List[GRIStandardNode] = field(default_factory=list)
    content_blocks: List[str] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)

class GRISegmenter:
    """Segment a parsed GRI document into hierarchical sections."""

    def __init__(self):
        self.heading_detector = HeadingDetector()

    def segment(self, document: DocumentContent) -> GRIDocumentTree:
        """Build the document tree from parsed content."""
        tree = GRIDocumentTree(
            source_file=document.file_name,
            total_pages=document.total_pages,
        )

        classified_blocks = self._classify_all_blocks(document)

        current_standard: Optional[GRIStandardNode] = None
        current_disclosure: Optional[GRIDisclosureNode] = None
        current_req: Optional[GRIRequirementNode] = None

        for block_info in classified_blocks:
            block = block_info["block"]
            heading = block_info["heading"]
            page_num = block_info["page_num"]

            if heading and heading.is_heading:
                if heading.heading_type == "gri_standard":
                    current_standard = GRIStandardNode(
                        standard_id=heading.section_label,
                        title=heading.text,
                        page_start=page_num,
                        page_end=page_num,
                    )
                    tree.standards.append(current_standard)
                    current_disclosure = None
                    current_req = None

                elif heading.heading_type in ("gri_disclosure", "gri_topic"):
                    current_disclosure = GRIDisclosureNode(
                        id=heading.section_label,
                        label=heading.text[:120],
                        text=heading.text,
                        page_start=page_num,
                        page_end=page_num,
                    )
                    if current_standard:
                        current_standard.disclosures.append(current_disclosure)
                    current_req = None

                elif heading.heading_type == "gri_subsection":
                    current_req = GRIRequirementNode(
                        type=heading.section_label,
                        text=heading.text,
                        page_start=page_num,
                        page_end=page_num,
                    )
                    if current_disclosure:
                        current_disclosure.requirements.append(current_req)
            else:
                target = current_req or current_disclosure or current_standard or tree
                if target:
                    if block.block_type == "text":
                        if hasattr(target, 'content_blocks'):
                            target.content_blocks.append(block.text)
                    elif block.block_type == "table":
                        table_data = normalize_table(block)
                        if table_data and hasattr(target, 'tables'):
                            target.tables.append(table_data)
                    
                    # Update page_end on target and recursively on all active parent nodes
                    if hasattr(target, 'page_end'):
                        target.page_end = max(target.page_end, page_num)
                    if current_req:
                        current_req.page_end = max(current_req.page_end, page_num)
                    if current_disclosure:
                        current_disclosure.page_end = max(current_disclosure.page_end, page_num)
                    if current_standard:
                        current_standard.page_end = max(current_standard.page_end, page_num)

        logger.info(
            f"Segmented {document.file_name}: "
            f"{len(tree.standards)} standards, "
            f"{sum(len(s.disclosures) for s in tree.standards)} disclosures"
        )
        return tree

    def _classify_all_blocks(self, document: DocumentContent) -> List[Dict]:
        """Classify every block in the document with heading info using GRI mode."""
        result = []
        for page in document.pages:
            for block in page.blocks:
                heading = None
                if block.block_type == "text":
                    heading = self.heading_detector.classify_block(block, is_gri=True)

                result.append({
                    "block": block,
                    "heading": heading,
                    "page_num": page.page_num,
                })
        return result
