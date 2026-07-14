"""
main.py – BRSR-GRI Ontology Mapping Pipeline CLI
==================================================

Orchestrates all 5 phases:
  Phase 1: PDF Parsing           → Structured JSON
  Phase 2: Ontology Construction → RDF/OWL + Neo4j CSV
  Phase 3: Semantic Mapping      → Mapping Repository
  Phase 4: Evaluation            → Reports + Metrics
  Phase 5: Agentic Workflow      → LangGraph orchestration

Usage:
  python main.py --phase 1                    # Parse PDFs
  python main.py --phase 1 --company TCS      # Parse specific company
  python main.py --phase 2                    # Build ontologies
  python main.py --all                        # Run full pipeline
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path

import yaml

# Try to load .env from parent directory
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip("'\"")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load pipeline configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def banner():
    print(r"""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║   BRSR-GRI Ontology-Guided Semantic Mapping Framework          ║
    ║   ────────────────────────────────────────────────────          ║
    ║   Phase 1 ─ PDF Parsing & Structured JSON                     ║
    ║   Phase 2 ─ Ontology Construction (BRSR + GRI)                ║
    ║   Phase 3 ─ Semantic Mapping Engine                            ║
    ║   Phase 4 ─ Evaluation & Explainability                        ║
    ║   Phase 5 ─ LangGraph Agentic Workflow                         ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)


def run_phase1(config: dict, company: str = None):
    """Phase 1: Parse PDFs → Structured JSON."""
    logger.info("═══ PHASE 1: PDF Parsing & Structured JSON ═══")
    t0 = time.time()

    from parser.pdf_parser import parse_raw_directory
    from parser.section_segmenter import SectionSegmenter, GRISegmenter
    from parser.json_exporter import export_batch

    raw_dir = config["data"]["raw_dir"]
    output_dir = config["data"]["output_dir"]

    # Parse PDFs
    documents = parse_raw_directory(raw_dir, company)
    logger.info(f"Parsed {len(documents)} documents")

    # Segment into hierarchical trees
    brsr_segmenter = SectionSegmenter()
    gri_segmenter = GRISegmenter()
    trees = []
    
    for doc in documents:
        is_gri = "GRI" in doc.file_name
        logger.info(f"📄 Segmenting: {doc.file_name} (is_gri={is_gri})")
        if is_gri:
            tree = gri_segmenter.segment(doc)
        else:
            tree = brsr_segmenter.segment(doc)
        trees.append(tree)

    # NLP Enrichment
    from parser.nlp_extractor import EnrichmentEngine
    enricher = EnrichmentEngine()
    enricher.enrich_trees(trees)

    # Export to JSON
    export_batch(trees, output_dir)

    elapsed = time.time() - t0
    logger.info(f"Phase 1 complete in {elapsed:.1f}s — {len(trees)} documents processed")
    return trees


def run_phase2(config: dict):
    """Phase 2: Build ontologies from structured JSON."""
    # Add a dedicated phase 2 log file
    fh = logging.FileHandler("logs/phase2.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    
    logger.info("═══ PHASE 2: Ontology Construction ═══")
    t0 = time.time()
    
    from ontology.builder import OntologyBuilder
    from ontology.visualizer import generate_mermaid_diagram
    
    processed_dir = config["data"]["output_dir"]
    ontology_output_dir = os.path.join(processed_dir, "ontology")
    graph_output_dir = os.path.join(processed_dir, "graph")
    
    os.makedirs(ontology_output_dir, exist_ok=True)
    os.makedirs(graph_output_dir, exist_ok=True)
    
    # 1. Build RDF Graph
    builder = OntologyBuilder(processed_dir)
    graph = builder.build()
    
    # Save TTL
    ttl_path = os.path.join(ontology_output_dir, "esg_ontology.ttl")
    graph.serialize(destination=ttl_path, format="turtle")
    logger.info(f"✅ Exported RDF ontology to {ttl_path} ({len(graph)} triples)")
    
    # 2. Export to Neo4j CSV
    builder.export_neo4j_csv(graph_output_dir)
    
    # 3. Generate Mermaid visualization
    visualizer_path = os.path.join(ontology_output_dir, "ontology_diagram.md")
    try:
        generate_mermaid_diagram(builder.nodes_data, builder.edges_data, visualizer_path)
    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}")
    
    elapsed = time.time() - t0
    logger.info(f"Phase 2 complete in {elapsed:.1f}s")





def run_phase3(config: dict):
    """Phase 3: Semantic Mapping Engine"""
    logger.info("═══ PHASE 3: Semantic Mapping Engine ═══")
    start_time = time.time()
    
    output_dir = Path(config["data"]["output_dir"])
    ontology_path = output_dir / "ontology" / "esg_ontology.ttl"
    mapping_out_dir = output_dir / "mapping"
    mapping_out_dir.mkdir(parents=True, exist_ok=True)
    
    if not ontology_path.exists():
        logger.error(f"Ontology not found at {ontology_path}. Run Phase 2 first.")
        return
        
    from mapping.engine import SemanticMappingEngine
    engine = SemanticMappingEngine(config)
    engine.run(str(ontology_path), str(mapping_out_dir))
    
    logger.info(f"Phase 3 complete in {time.time() - start_time:.1f}s")


def run_phase4(config: dict):
    """Phase 4: Evaluation & explainability."""
    logger.info("═══ PHASE 4: Evaluation & Explainability ═══")
    
    output_dir = Path(config["data"]["output_dir"])
    mapping_dir = output_dir / "mapping"
    
    if not (mapping_dir / "mapping_repository.json").exists():
        logger.error("Mapping repository not found. Run Phase 3 first.")
        return
        
    from mapping.evaluator import MappingEvaluator
    evaluator = MappingEvaluator(config)
    
    logger.info("Running Multi-LLM Evaluation...")
    evaluator.run_multi_llm_evaluation(str(mapping_dir))
    
    logger.info("Generating CLI Report...")
    evaluator.generate_cli_report(str(mapping_dir))
    
    logger.info("Generating HTML Dashboard...")
    evaluator.generate_dashboard(str(mapping_dir), str(mapping_dir))
    
    logger.info("Phase 4 complete. Dashboard saved to data/processed/mapping/mapping_dashboard.html")


def run_phase5(config: dict):
    """Phase 5: LangGraph agentic workflow."""
    logger.info("═══ PHASE 5: LangGraph Agentic Workflow ═══")
    logger.info("Phase 5 implementation pending — requires all prior phases")


def main():
    parser = argparse.ArgumentParser(
        description="BRSR-GRI Ontology-Guided Semantic Mapping Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5], help="Run specific phase")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--company", type=str, help="Process specific company (Phase 1)")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file path")

    args = parser.parse_args()

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    banner()
    config = load_config(args.config)

    if args.phase == 1 or args.all:
        run_phase1(config, args.company)
    if args.phase == 2 or args.all:
        run_phase2(config)
    if args.phase == 3 or args.all:
        run_phase3(config)
    if args.phase == 4 or args.all:
        run_phase4(config)
    if args.phase == 5 or args.all:
        run_phase5(config)

    if not args.phase and not args.all:
        parser.print_help()


if __name__ == "__main__":
    main()
