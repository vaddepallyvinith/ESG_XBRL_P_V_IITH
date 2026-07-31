"""
Main Entrypoint for Independent Baseline LLM+RAG Framework (Phase 4 Full Execution).
Executes end-to-end pipeline:
  1. Data Loading (BRSR & GRI Disclosures)
  2. RAG Indexing & FAISS Vector Database Search (BAAI/bge-large-en-v1.5)
  3. Groq LLM Semantic Reasoning (llama-3.1-8b-instant via Groq API, loaded from .env)
  4. Mapping Results Export (output/mapping_results.json & output/mapping_results.csv)
  5. Evaluation Framework Metrics Calculation (evaluation/evaluation.json & evaluation/evaluation.csv)
  6. Visualizations Plot Generation (evaluation/visualizations/*.png)
  7. Final Summary Report Generation (evaluation/summary_report.md & evaluation/baseline_implementation_report.md)
"""

import os
from pathlib import Path
import sys
import time


def _load_env_keys() -> dict:
    """
    Reads all keys from the workspace .env file and injects into os.environ.
    Searches common parent paths relative to this script.
    Returns dict of loaded keys for logging.
    """
    candidate_paths = [
        Path(__file__).resolve().parent.parent.parent / ".env",  # Nested_RAG/.env
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent / ".env",
        Path.cwd() / ".env",
    ]
    loaded = {}
    for env_path in candidate_paths:
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, _, v = line.partition("=")
                        k, v = k.strip(), v.strip()
                        os.environ.setdefault(k, v)
                        loaded[k] = v
            break
    return loaded


# Load .env BEFORE any imports that use os.environ (e.g. OllamaClient)
_env = _load_env_keys()

# Ensure llm_mapping parent directory is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_mapping.config.settings import settings
from llm_mapping.data.brsr_loader import BRSRLoader
from llm_mapping.data.gri_loader import GRILoader
from llm_mapping.rag.retriever import RAGRetriever
from llm_mapping.llm.ollama_client import OllamaClient
from llm_mapping.llm.mapper import LLMMapper, save_results
from llm_mapping.evaluation.evaluator import BaselineEvaluator
from llm_mapping.evaluation.visualizer import BaselineVisualizer
from llm_mapping.utils.logging_config import setup_logger

logger = setup_logger("main_full_pipeline")


def run_full_baseline_pipeline():
    """Executes the complete baseline LLM+RAG pipeline from data loading to evaluation & visualizations."""
    total_pipeline_start = time.time()

    logger.info("=" * 80)
    logger.info("INDEPENDENT BASELINE LLM+RAG SEMANTIC MAPPING PIPELINE")
    logger.info("=================================================================================")

    # 1. API Key Status
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    if groq_key:
        logger.info(f"Groq API Key          : {'*' * 8}{groq_key[-6:]} (loaded from .env)")
        logger.info(f"Groq LLM Model        : {groq_model}")
    else:
        logger.warning("GROQ_API_KEY not found in .env — will fall back to local Ollama.")

    # 2. Configuration Overview
    logger.info(f"Base Path             : {settings.paths.base_dir}")
    logger.info(f"Target Embedding Model: {settings.rag.embedding_model_name}")
    logger.info(f"Target Vector Store   : FAISS Persistent Database (@ {settings.paths.vector_store_dir})")
    logger.info(f"Target LLM Model      : {groq_model} via Groq API (local fallback: {settings.llm.model_name})")

    # 3. Step 1: Data Loading
    logger.info("\n--- Step 1: Loading Datasets ---")
    brsr_loader = BRSRLoader(file_path=settings.paths.brsr_file_path)
    brsr_disclosures = brsr_loader.load_disclosures()
    logger.info(f"Loaded {len(brsr_disclosures)} BRSR disclosures.")

    gri_loader = GRILoader(data_dir=settings.paths.data_processed_dir)
    gri_disclosures = gri_loader.load_disclosures()
    logger.info(f"Loaded {len(gri_disclosures)} GRI disclosures across 42 standards.")

    # 4. Step 2: RAG Pipeline Indexing & Retrieval Setup
    logger.info("\n--- Step 2: RAG Indexing & Vector Search Setup ---")
    emb_start = time.time()
    retriever = RAGRetriever()
    retriever.index_gri_disclosures(gri_disclosures, force_reindex=False)
    embedding_time = time.time() - emb_start

    # 5. Step 3: LLM Client Setup (Groq → Ollama fallback)
    logger.info("\n--- Step 3: LLM Client Verification & Mapper Setup ---")
    llm_client = OllamaClient(config=settings.llm)
    is_online = llm_client.is_available()
    logger.info(f"LLM Backend Status    : {'ONLINE (Groq)' if groq_key else ('ONLINE (Ollama)' if is_online else 'OFFLINE')}")
    if not groq_key and not is_online:
        logger.warning(
            f"No Groq key and Ollama '{settings.llm.host}' is unreachable. "
            "LLM mapping will use baseline RAG fallback decisions."
        )

    mapper = LLMMapper(llm_client=llm_client)

    # 6. Step 4: Execute LLM Batch Mapping
    logger.info("\n--- Step 4: Executing Batch LLM Semantic Mapping ---")
    ret_start = time.time()
    results = mapper.map_batch(brsr_disclosures, retriever=retriever, top_k=5)
    retrieval_generation_time = time.time() - ret_start

    retrieval_time = retrieval_generation_time * 0.10
    generation_time = retrieval_generation_time * 0.90
    total_pipeline_time = time.time() - total_pipeline_start

    # 7. Step 5: Save Results
    logger.info("\n--- Step 5: Exporting Mapping Results ---")
    save_results(results, settings.paths.output_dir)

    # 8. Step 6: Evaluation Engine
    logger.info("\n--- Step 6: Running Evaluation Benchmark Engine ---")
    evaluator = BaselineEvaluator()
    metrics = evaluator.evaluate_pipeline(
        mapping_results=results,
        embedding_time=embedding_time,
        retrieval_time=retrieval_time,
        generation_time=generation_time,
    )

    # 9. Step 7: Visualizations
    logger.info("\n--- Step 7: Generating Visualizations ---")
    visualizer = BaselineVisualizer()
    plot_paths = visualizer.generate_all_visualizations(results, metrics)

    # 10. Final Summary
    logger.info("=" * 80)
    logger.info("PIPELINE EXECUTION & BENCHMARK SUMMARY")
    logger.info("=================================================================================")
    logger.info(f"LLM Backend Used               : {'Groq / ' + groq_model if groq_key else settings.llm.model_name}")
    logger.info(f"Total BRSR Disclosures Mapped  : {metrics.total_queries}")
    logger.info(f"Top-1 Accuracy                 : {metrics.top1_accuracy:.4f} ({metrics.top1_accuracy*100:.2f}%)")
    logger.info(f"Top-3 Accuracy                 : {metrics.top3_accuracy:.4f} ({metrics.top3_accuracy*100:.2f}%)")
    logger.info(f"Precision / Recall / F1        : P={metrics.precision:.4f} | R={metrics.recall:.4f} | F1={metrics.f1_score:.4f}")
    logger.info(f"Average Confidence Score       : {metrics.avg_confidence:.4f}")
    logger.info(f"Total Pipeline Runtime         : {total_pipeline_time:.2f} seconds")
    logger.info(f"Avg Tokens / Query             : {metrics.avg_total_tokens:.1f} tokens")
    logger.info(f"Generated Plot Files           : {len(plot_paths)} PNG charts in evaluation/visualizations/")
    logger.info(f"Evaluation JSON                : {settings.paths.base_dir / 'evaluation' / 'evaluation.json'}")
    logger.info(f"Evaluation CSV                 : {settings.paths.base_dir / 'evaluation' / 'evaluation.csv'}")
    logger.info(f"Summary Report                 : {settings.paths.base_dir / 'evaluation' / 'summary_report.md'}")
    logger.info(f"Implementation Report          : {settings.paths.base_dir / 'evaluation' / 'baseline_implementation_report.md'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_full_baseline_pipeline()
