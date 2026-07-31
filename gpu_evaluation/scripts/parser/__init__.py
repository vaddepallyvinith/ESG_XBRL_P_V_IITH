"""BRSR-GRI Ontology-Guided Semantic Mapping – Parser Module"""
from parser.pdf_parser import BRSRPdfParser
from parser.json_exporter import export_structured_json

__all__ = ["BRSRPdfParser", "export_structured_json"]
