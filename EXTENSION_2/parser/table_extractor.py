"""
table_extractor.py – Enhanced table extraction for BRSR reports
================================================================

Wraps PyMuPDF table detection and normalizes output into
structured rows with header detection and cleanup.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from parser.pdf_parser import TableBlock

logger = logging.getLogger(__name__)


def normalize_table(table: TableBlock) -> Dict[str, Any]:
    """Normalize a TableBlock into a clean dictionary representation."""
    if not table.headers and not table.rows:
        return {}

    # Clean headers
    clean_headers = [_clean_cell(h) for h in table.headers]

    # Clean rows
    clean_rows = []
    for row in table.rows:
        clean_row = [_clean_cell(c) for c in row]
        # Skip completely empty rows
        if any(c for c in clean_row):
            clean_rows.append(clean_row)

    if not clean_rows:
        return {}

    # Try to detect if first row is actually a header
    if not any(clean_headers) and clean_rows:
        clean_headers = clean_rows.pop(0)

    return {
        "headers": clean_headers,
        "rows": clean_rows,
        "row_count": len(clean_rows),
        "col_count": len(clean_headers),
        "page_num": table.page_num,
        "bbox": list(table.bbox),
    }


def table_to_records(table: TableBlock) -> List[Dict[str, str]]:
    """Convert a table to a list of key-value records (one per row)."""
    norm = normalize_table(table)
    if not norm:
        return []

    headers = norm["headers"]
    records = []
    for row in norm["rows"]:
        record = {}
        for i, header in enumerate(headers):
            key = header if header else f"col_{i}"
            val = row[i] if i < len(row) else ""
            record[key] = val
        records.append(record)

    return records


def detect_metric_table(table: TableBlock) -> bool:
    """Heuristic: check if table likely contains BRSR metrics/indicators."""
    all_text = " ".join(table.headers).lower()
    for row in table.rows:
        all_text += " " + " ".join(row).lower()

    metric_keywords = [
        "fy", "current", "previous", "unit", "total",
        "male", "female", "employee", "worker",
        "energy", "water", "waste", "emission",
        "scope", "ghg", "percentage", "number",
    ]
    hits = sum(1 for kw in metric_keywords if kw in all_text)
    return hits >= 3


def _clean_cell(text: str) -> str:
    """Clean a table cell value."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Remove common artifacts
    text = text.replace("\n", " ").replace("\r", "")
    return text
