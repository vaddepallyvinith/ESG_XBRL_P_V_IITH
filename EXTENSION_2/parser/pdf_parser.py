"""
pdf_parser.py – Phase 1: BRSR PDF Document Parser
===================================================

Parses BRSR report PDFs using PyMuPDF (fitz) to extract:
  - Text blocks with font metadata (size, bold, italic, color)
  - Tables with cell-level structure
  - Page-level document structure
  - Reading order preservation

Produces a list of PageContent objects containing typed blocks
(TextBlock, TableBlock) that downstream modules consume.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class TextSpan:
    """A single text span with font metadata."""
    text: str
    font_name: str = ""
    font_size: float = 0.0
    is_bold: bool = False
    is_italic: bool = False
    color: int = 0


@dataclass
class TextBlock:
    """A block of text from a PDF page."""
    block_type: str = "text"
    text: str = ""
    spans: List[TextSpan] = field(default_factory=list)
    bbox: tuple = (0, 0, 0, 0)
    page_num: int = 0
    avg_font_size: float = 0.0
    is_bold: bool = False
    line_count: int = 1


@dataclass
class TableCell:
    """A single cell in a table."""
    text: str = ""
    row: int = 0
    col: int = 0


@dataclass
class TableBlock:
    """A table extracted from a PDF page."""
    block_type: str = "table"
    cells: List[TableCell] = field(default_factory=list)
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    bbox: tuple = (0, 0, 0, 0)
    page_num: int = 0
    row_count: int = 0
    col_count: int = 0


@dataclass
class PageContent:
    """All content from a single PDF page."""
    page_num: int
    width: float = 0.0
    height: float = 0.0
    blocks: List = field(default_factory=list)  # TextBlock | TableBlock


@dataclass
class DocumentContent:
    """Complete parsed document."""
    file_path: str = ""
    file_name: str = ""
    company: str = ""
    report_type: str = ""
    fiscal_year: str = ""
    total_pages: int = 0
    pages: List[PageContent] = field(default_factory=list)


# ─── PDF Parser ──────────────────────────────────────────────────────────────

class BRSRPdfParser:
    """Parse BRSR report PDFs using PyMuPDF."""

    def __init__(self, heading_sizes=None):
        self.heading_sizes = heading_sizes or {"h1": 16.0, "h2": 13.0, "h3": 11.0, "body": 9.0}

    def parse(self, pdf_path: str) -> DocumentContent:
        """Parse a single PDF file and return structured content."""
        pdf_path = str(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        path_obj = Path(pdf_path)

        # Detect metadata from path: .../Company/BRSR/FYxxxx.pdf
        parts = path_obj.parts
        company = ""
        report_type = ""
        fiscal_year = path_obj.stem  # e.g., "FY2023-24"

        for i, part in enumerate(parts):
            if part in ("BRSR", "Annual"):
                report_type = part
                if i > 0:
                    company = parts[i - 1]
                break

        content = DocumentContent(
            file_path=pdf_path,
            file_name=path_obj.name,
            company=company,
            report_type=report_type,
            fiscal_year=fiscal_year,
            total_pages=len(doc),
        )

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_content = self._parse_page(page, page_num + 1)
            content.pages.append(page_content)

        doc.close()
        logger.info(
            f"Parsed {path_obj.name}: {content.total_pages} pages, "
            f"{sum(len(p.blocks) for p in content.pages)} blocks"
        )
        return content

    def _parse_page(self, page: fitz.Page, page_num: int) -> PageContent:
        """Parse a single page into text and table blocks."""
        pc = PageContent(
            page_num=page_num,
            width=page.rect.width,
            height=page.rect.height,
        )

        # Extract tables first (so we can exclude table regions from text)
        tables = self._extract_tables(page, page_num)
        table_bboxes = [t.bbox for t in tables]

        # Extract text blocks (excluding table regions)
        text_blocks = self._extract_text_blocks(page, page_num, table_bboxes)

        # Merge in reading order (top to bottom)
        all_blocks = text_blocks + tables
        all_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))  # Sort by y, then x
        pc.blocks = all_blocks

        return pc

    def _extract_text_blocks(
        self, page: fitz.Page, page_num: int, exclude_bboxes: List[tuple]
    ) -> List[TextBlock]:
        """Extract text blocks with font metadata, excluding table regions."""
        blocks = []
        raw_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in raw_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 = text block
                continue

            bbox = tuple(block["bbox"])

            # Skip if overlaps with a table
            if self._overlaps_any(bbox, exclude_bboxes):
                continue

            spans_data = []
            full_text_parts = []
            font_sizes = []
            bold_count = 0
            total_spans = 0

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    font_name = span.get("font", "")
                    font_size = span.get("size", 0.0)
                    is_bold = "Bold" in font_name or "bold" in font_name
                    is_italic = "Italic" in font_name or "italic" in font_name

                    spans_data.append(TextSpan(
                        text=text,
                        font_name=font_name,
                        font_size=font_size,
                        is_bold=is_bold,
                        is_italic=is_italic,
                        color=span.get("color", 0),
                    ))
                    full_text_parts.append(text)
                    font_sizes.append(font_size)
                    if is_bold:
                        bold_count += 1
                    total_spans += 1

            if not full_text_parts:
                continue

            full_text = " ".join(full_text_parts)
            avg_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0.0
            is_bold = bold_count > total_spans / 2 if total_spans > 0 else False

            blocks.append(TextBlock(
                text=full_text,
                spans=spans_data,
                bbox=bbox,
                page_num=page_num,
                avg_font_size=round(avg_size, 1),
                is_bold=is_bold,
                line_count=len(block.get("lines", [])),
            ))

        return blocks

    def _extract_tables(self, page: fitz.Page, page_num: int) -> List[TableBlock]:
        """Extract tables from a page using PyMuPDF's table detection."""
        tables = []
        try:
            found = page.find_tables()
            for table in found:
                extracted = table.extract()
                if not extracted or len(extracted) < 2:
                    continue

                # First row as headers
                headers = [str(c).strip() if c else "" for c in extracted[0]]
                rows = []
                cells = []

                for r_idx, row in enumerate(extracted):
                    row_data = [str(c).strip() if c else "" for c in row]
                    if r_idx > 0:
                        rows.append(row_data)
                    for c_idx, cell_text in enumerate(row_data):
                        cells.append(TableCell(text=cell_text, row=r_idx, col=c_idx))

                tables.append(TableBlock(
                    cells=cells,
                    headers=headers,
                    rows=rows,
                    bbox=tuple(table.bbox),
                    page_num=page_num,
                    row_count=len(extracted),
                    col_count=len(headers),
                ))
        except Exception as e:
            logger.debug(f"Table extraction failed on page {page_num}: {e}")

        return tables

    def _overlaps_any(self, bbox: tuple, exclusions: List[tuple], threshold: float = 0.5) -> bool:
        """Check if bbox overlaps significantly with any exclusion region."""
        x0, y0, x1, y1 = bbox
        area = max((x1 - x0) * (y1 - y0), 1)

        for ex in exclusions:
            ex0, ey0, ex1, ey1 = ex
            ix0 = max(x0, ex0)
            iy0 = max(y0, ey0)
            ix1 = min(x1, ex1)
            iy1 = min(y1, ey1)
            if ix0 < ix1 and iy0 < iy1:
                overlap_area = (ix1 - ix0) * (iy1 - iy0)
                if overlap_area / area > threshold:
                    return True
        return False


# ─── Batch Parser ────────────────────────────────────────────────────────────

def parse_brsr_directory(base_dir: str, companies: List[str] = None) -> List[DocumentContent]:
    """Parse all BRSR PDFs in the financial_dataset directory structure."""
    # Resolve relative paths against the project root
    if not Path(base_dir).is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        base = project_root / base_dir
    else:
        base = Path(base_dir)
    parser = BRSRPdfParser()
    documents = []

    # Pattern: base_dir/Company/BRSR/*.pdf
    brsr_pdfs = sorted(base.glob("*/BRSR/*.pdf"))

    if companies:
        brsr_pdfs = [p for p in brsr_pdfs if any(c in str(p) for c in companies)]

    if not brsr_pdfs:
        raise FileNotFoundError(f"No BRSR PDFs found under {base_dir}")

    logger.info(f"Found {len(brsr_pdfs)} BRSR PDFs")

    for pdf_path in brsr_pdfs:
        try:
            doc = parser.parse(str(pdf_path))
            documents.append(doc)
        except Exception as e:
            logger.error(f"Failed to parse {pdf_path}: {e}")

    return documents


def parse_raw_directory(raw_dir: str, file_filter: Optional[str] = None) -> List[DocumentContent]:
    """Parse standard PDFs directly in the raw directory. If file_filter is provided, only process matching filenames."""
    if not Path(raw_dir).is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        base = project_root / raw_dir
    else:
        base = Path(raw_dir)
        
    parser = BRSRPdfParser()
    documents = []

    # Get all PDFs recursively from the base directory
    pdfs = sorted(base.rglob("*.pdf"))
    
    if file_filter:
        pdfs = [p for p in pdfs if file_filter.lower() in p.name.lower()]

    # Environmental Filtering: Strictly process GRI 300 series, BRSR, and Universals
    filtered_pdfs = []
    for p in pdfs:
        name = p.name
        if "BRSR" in name:
            filtered_pdfs.append(p)
        elif "GRI " in name:
            # Exclude Economic (200), Social (400), Sector (11-14), and Non-Env Topic Standards
            if "GRI 2" in name and "GRI 2_" not in name:  # Exclude 200 series but keep GRI 2
                continue
            if "GRI 4" in name: # Exclude 400 series
                continue
            if re.search(r"GRI 1[1-9]_", name): # Exclude Sector
                continue
            filtered_pdfs.append(p)
        else:
            filtered_pdfs.append(p)
    pdfs = filtered_pdfs

    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under {base} matching filter: {file_filter}")

    logger.info(f"Found {len(pdfs)} standard PDFs to parse.")

    for pdf_path in pdfs:
        try:
            doc = parser.parse(str(pdf_path))
            documents.append(doc)
        except Exception as e:
            logger.error(f"Failed to parse {pdf_path}: {e}")

    return documents
