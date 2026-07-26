# BRSR-GRI Ontology-Guided Semantic Mapping Framework

**Research-grade AI framework for automated ontology-guided semantic mapping between BRSR Environmental Reporting Standard (SEBI) and GRI Environmental Standards.**

## Research Problem

> Can an ontology-guided semantic mapping system automatically align BRSR Environmental disclosures with GRI Environmental Standards using multi-source semantic evidence while generating explainable and confidence-aware mappings?

## Architecture

```
Phase 1: PDF Parsing          → Structured JSON (hierarchy-preserving)
Phase 2: Ontology Construction → RDF/OWL + Neo4j (BRSR + GRI ontologies)
Phase 3: Semantic Mapping      → Multi-evidence candidate mapping
Phase 4: Evaluation            → Confidence scoring + explainability
Phase 5: Agentic Workflow      → LangGraph orchestration (7 agents)
```

## Quick Start

```bash
# Setup
cd EXTENSION_2
pip install -r requirements.txt

# Phase 1: Parse BRSR PDFs
python main.py --phase 1

# Parse single company
python main.py --phase 1 --company TCS

# Run all phases
python main.py --all
```

## Project Structure

```
EXTENSION_2/
├── config/settings.yaml       # Global configuration
├── data/raw/                  # Source PDFs
├── data/processed/            # Parsed JSON outputs
├── parser/                    # Phase 1: PDF parsing
├── ontology/                  # Phase 2: Ontology construction
├── mapping/                   # Phase 3: Semantic mapping
├── evaluation/                # Phase 4: Evaluation
├── agents/                    # Phase 5: LangGraph workflow
├── tests/                     # Unit tests
├── logs/                      # Pipeline logs
├── main.py                    # CLI entry point
└── requirements.txt           # Dependencies
```

## Data Sources

- **BRSR PDFs**: 30 reports (10 companies × 3 fiscal years) from `financial_dataset/`
- **GRI Standards**: GRI 300-series Environmental Standards

## Key Technologies

| Component | Technology |
|-----------|-----------|
| PDF Parsing | PyMuPDF (fitz) |
| Data Models | Pydantic |
| Ontology | RDFLib + OWL |
| Embeddings | Sentence-Transformers |
| LLM | Gemini 2.0 Flash |
| Orchestration | LangGraph |
