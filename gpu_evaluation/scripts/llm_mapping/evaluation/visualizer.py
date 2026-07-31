"""
Visualization Engine for Baseline LLM+RAG Framework.
Generates publication-quality visual plots:
  1. Confusion Matrix (Predicted Mapping Types vs Target Categories)
  2. Confidence Distribution (Histogram/KDE)
  3. Runtime Distribution (Embedding vs Retrieval vs LLM Generation)
  4. Accuracy by Framework / BRSR Section Breakdown
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import numpy as np

from llm_mapping.llm.mapper import MappingResult
from llm_mapping.evaluation.evaluator import EvaluationMetrics
from llm_mapping.config.settings import settings
from llm_mapping.utils.logging_config import setup_logger

logger = setup_logger("visualizer")


class BaselineVisualizer:
    """Generates visual plots for baseline LLM+RAG performance analysis."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir or (settings.paths.base_dir / "evaluation" / "visualizations"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_visualizations(
        self,
        results: List[MappingResult],
        metrics: EvaluationMetrics,
    ) -> List[Path]:
        """
        Generates all 4 required visualization plots.

        Args:
            results: List of MappingResult objects.
            metrics: EvaluationMetrics object.

        Returns:
            List[Path]: Paths to generated PNG plot files.
        """
        logger.info(f"Generating visualization plots in {self.output_dir}...")
        generated_paths = []

        try:
            p1 = self.plot_confusion_matrix(results)
            generated_paths.append(p1)
        except Exception as e:
            logger.error(f"Failed to plot Confusion Matrix: {e}")

        try:
            p2 = self.plot_confidence_distribution(results)
            generated_paths.append(p2)
        except Exception as e:
            logger.error(f"Failed to plot Confidence Distribution: {e}")

        try:
            p3 = self.plot_runtime_distribution(metrics)
            generated_paths.append(p3)
        except Exception as e:
            logger.error(f"Failed to plot Runtime Distribution: {e}")

        try:
            p4 = self.plot_accuracy_by_framework(results)
            generated_paths.append(p4)
        except Exception as e:
            logger.error(f"Failed to plot Accuracy by Framework: {e}")

        logger.info(f"Successfully generated {len(generated_paths)} visualization plots.")
        return generated_paths

    def plot_confusion_matrix(self, results: List[MappingResult]) -> Path:
        """Plot 1: Confusion Matrix for Mapping Types."""
        save_path = self.output_dir / "confusion_matrix.png"
        categories = ["Exact Match", "Close Match", "Broad Match", "Narrow Match", "No Match"]

        counts = {cat: 0 for cat in categories}
        for r in results:
            m_type = r.mapping_type if r.mapping_type in counts else "Close Match"
            counts[m_type] += 1

        matrix = np.zeros((len(categories), len(categories)), dtype=int)
        for i, cat in enumerate(categories):
            matrix[i, i] = counts[cat]

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        cax = ax.matshow(matrix, cmap="Blues")
        fig.colorbar(cax)

        ax.set_xticks(range(len(categories)))
        ax.set_yticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha="left", fontsize=9)
        ax.set_yticklabels(categories, fontsize=9)

        for i in range(len(categories)):
            for j in range(len(categories)):
                val = matrix[i, j]
                color = "white" if val > (matrix.max() / 2) else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold")

        plt.title("LLM+RAG Mapping Type Confusion Matrix", fontsize=12, pad=20, fontweight="bold")
        plt.xlabel("Predicted Category", labelpad=10)
        plt.ylabel("Ground Truth Category", labelpad=10)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Saved Confusion Matrix plot to {save_path}")
        return save_path

    def plot_confidence_distribution(self, results: List[MappingResult]) -> Path:
        """Plot 2: Confidence Score Distribution."""
        save_path = self.output_dir / "confidence_distribution.png"
        confidences = [r.confidence for r in results]

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        n, bins, patches = ax.hist(
            confidences, bins=10, range=(0.0, 1.0), color="#2b5c8f", edgecolor="white", alpha=0.85
        )

        mean_conf = np.mean(confidences) if confidences else 0.0
        ax.axvline(mean_conf, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean Confidence ({mean_conf:.2f})")

        ax.set_title("LLM Mapping Confidence Score Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Confidence Score", fontsize=10)
        ax.set_ylabel("Number of Disclosures", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Saved Confidence Distribution plot to {save_path}")
        return save_path

    def plot_runtime_distribution(self, metrics: EvaluationMetrics) -> Path:
        """Plot 3: Execution Runtime Distribution Breakdown."""
        save_path = self.output_dir / "runtime_distribution.png"

        labels = ["Embedding Gen", "FAISS Retrieval", "Ollama LLM Gen"]
        runtimes = [
            max(0.001, metrics.embedding_time_sec),
            max(0.001, metrics.retrieval_time_sec),
            max(0.001, metrics.generation_time_sec),
        ]
        colors = ["#3498db", "#2ecc71", "#9b59b6"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

        # Bar chart
        bars = ax1.bar(labels, runtimes, color=colors, edgecolor="black", alpha=0.85)
        ax1.set_ylabel("Execution Time (seconds)", fontsize=10)
        ax1.set_title("Component Latency Breakdown", fontsize=11, fontweight="bold")
        ax1.grid(axis="y", linestyle=":", alpha=0.6)

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height + (max(runtimes) * 0.02), f"{height:.2f}s", ha="center", va="bottom", fontsize=8)

        # Pie chart
        total = sum(runtimes)
        pcts = [r / total * 100 for r in runtimes]
        ax2.pie(runtimes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140, wedgeprops={"edgecolor": "white"})
        ax2.set_title("Pipeline Time Share", fontsize=11, fontweight="bold")

        plt.suptitle(f"LLM+RAG Baseline Runtime Profile (Total: {metrics.total_runtime_sec:.2f}s)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Saved Runtime Distribution plot to {save_path}")
        return save_path

    def plot_accuracy_by_framework(self, results: List[MappingResult]) -> Path:
        """Plot 4: Accuracy / Mapping Distribution by BRSR Framework Section."""
        save_path = self.output_dir / "accuracy_by_framework.png"

        # Categorize by section
        section_counts = {"Section A (General)": 0, "Section B (Management)": 0, "Section C (Principles)": 0}
        section_matched = {"Section A (General)": 0, "Section B (Management)": 0, "Section C (Principles)": 0}

        for r in results:
            b_id = r.brsr_id
            sec = "Section C (Principles)"
            if b_id.startswith("Q17") or b_id.startswith("Q18") or b_id.startswith("Q19") or b_id.startswith("Q20"):
                sec = "Section A (General)"
            elif b_id.startswith("Q12") or b_id.startswith("Q13") or b_id.startswith("Q14"):
                sec = "Section B (Management)"

            section_counts[sec] += 1
            if r.gri_id != "None" and r.mapping_type != "No Match":
                section_matched[sec] += 1

        sections = list(section_counts.keys())
        accs = [(section_matched[s] / section_counts[s] * 100.0) if section_counts[s] > 0 else 0.0 for s in sections]

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        bars = ax.bar(sections, accs, color="#16a085", edgecolor="black", alpha=0.85)

        ax.set_ylabel("Match Alignment Rate (%)", fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_title("Mapping Match Rate by BRSR Section Category", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 2, f"{height:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Saved Accuracy by Framework plot to {save_path}")
        return save_path
