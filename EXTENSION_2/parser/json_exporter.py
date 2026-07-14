"""
json_exporter.py – Export segmented documents to structured JSON
=================================================================

Converts DocumentTree hierarchies into JSON preserving:
  Framework, Topic, Disclosure, Metric, Definition,
  Units, Page Number, Section Number, Hierarchy
"""

import os
import json
import logging
from typing import Dict, List, Any
from parser.section_segmenter import DocumentTree, SectionNode, PrincipleNode, IndicatorGroup, DisclosureNode
from parser.section_segmenter import GRIDocumentTree, GRIStandardNode, GRIDisclosureNode, GRIRequirementNode

logger = logging.getLogger(__name__)


def gri_requirement_to_dict(req: GRIRequirementNode) -> Dict[str, Any]:
    return {
        "type": req.type,
        "text": req.text,
        "content": req.content_blocks,
        "tables": req.tables,
        "page_start": req.page_start,
        "page_end": req.page_end,
    }

def gri_disclosure_to_dict(d: GRIDisclosureNode) -> Dict[str, Any]:
    return {
        "id": d.id,
        "label": d.label,
        "text": d.text,
        "requirements": [gri_requirement_to_dict(r) for r in d.requirements],
        "content": d.content_blocks,
        "tables": d.tables,
        "page_start": d.page_start,
        "page_end": d.page_end,
    }

def gri_standard_to_dict(s: GRIStandardNode) -> Dict[str, Any]:
    return {
        "standard_id": s.standard_id,
        "title": s.title,
        "page_start": s.page_start,
        "page_end": s.page_end,
        "content": s.content_blocks,
        "tables": s.tables,
        "disclosures": [gri_disclosure_to_dict(d) for d in s.disclosures],
    }

def gri_document_tree_to_dict(tree: GRIDocumentTree) -> Dict[str, Any]:
    return {
        "framework": "GRI",
        "source_file": tree.source_file,
        "total_pages": tree.total_pages,
        "content": tree.content_blocks,
        "tables": tree.tables,
        "standards": [gri_standard_to_dict(s) for s in tree.standards],
    }

def disclosure_to_dict(d: DisclosureNode) -> Dict[str, Any]:
    """Convert a DisclosureNode to a serializable dict."""
    return {
        "id": d.id,
        "label": d.label,
        "text": d.text,
        "content": d.content_blocks,
        "tables": d.tables,
        "page_start": d.page_start,
        "page_end": d.page_end,
    }


def indicator_group_to_dict(ig: IndicatorGroup) -> Dict[str, Any]:
    return {
        "group_type": ig.group_type,
        "page_start": ig.page_start,
        "page_end": ig.page_end,
        "content": ig.content_blocks,
        "tables": ig.tables,
        "disclosures": [disclosure_to_dict(d) for d in ig.disclosures],
    }


def principle_to_dict(p: PrincipleNode) -> Dict[str, Any]:
    return {
        "principle_num": p.principle_num,
        "label": p.label,
        "text": p.text,
        "page_start": p.page_start,
        "page_end": p.page_end,
        "content": p.content_blocks,
        "tables": p.tables,
        "indicator_groups": [indicator_group_to_dict(ig) for ig in p.indicator_groups],
        "disclosures": [disclosure_to_dict(d) for d in p.disclosures],
    }


def section_to_dict(s: SectionNode) -> Dict[str, Any]:
    return {
        "section_id": s.section_id,
        "label": s.label,
        "page_start": s.page_start,
        "page_end": s.page_end,
        "content": s.content_blocks,
        "tables": s.tables,
        "principles": [principle_to_dict(p) for p in s.principles],
        "disclosures": [disclosure_to_dict(d) for d in s.disclosures],
    }


def document_tree_to_dict(tree: DocumentTree) -> Dict[str, Any]:
    """Convert a complete DocumentTree to a JSON-serializable dict."""
    return {
        "framework": "BRSR",
        "framework_full_name": "Business Responsibility and Sustainability Report",
        "issuer": "SEBI",
        "company": tree.company,
        "fiscal_year": tree.fiscal_year,
        "source_file": tree.source_file,
        "total_pages": tree.total_pages,
        "content": tree.content_blocks,
        "tables": tree.tables,
        "sections": [section_to_dict(s) for s in tree.sections],
    }


def export_structured_json(tree: Any, output_path: str):
    """Export a single document tree to JSON."""
    if isinstance(tree, GRIDocumentTree):
        data = gri_document_tree_to_dict(tree)
    else:
        data = document_tree_to_dict(tree)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Exported: {output_path}")


def export_batch(trees: List[Any], output_dir: str):
    """Export multiple document trees."""
    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    for tree in trees:
        if isinstance(tree, GRIDocumentTree):
            filename = f"GRI_{tree.source_file.replace('.pdf', '')}.json"
            filepath = os.path.join(output_dir, filename)
            export_structured_json(tree, filepath)
            manifest.append({
                "framework": "GRI",
                "source_file": tree.source_file,
                "output_file": filename,
                "total_pages": tree.total_pages,
                "standards": len(tree.standards),
            })
        else:
            filename = f"{tree.company}_{tree.fiscal_year}.json" if tree.company else f"BRSR_{tree.source_file.replace('.pdf', '')}.json"
            filepath = os.path.join(output_dir, filename)
            export_structured_json(tree, filepath)
            manifest.append({
                "framework": "BRSR",
                "company": tree.company,
                "fiscal_year": tree.fiscal_year,
                "source_file": tree.source_file,
                "output_file": filename,
                "total_pages": tree.total_pages,
                "sections": len(tree.sections),
            })

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"documents": manifest}, f, indent=2)
    logger.info(f"✅ Manifest: {manifest_path} ({len(manifest)} documents)")
