"""
visualizer.py - Generates publication-quality visualizations for Phase 4 evaluation:
  1. Similarity Distribution
  2. Confidence Distribution
  3. Mapping Type Distribution (SKOS relations)
  4. Ontology Coverage
  5. Runtime Breakdown
  6. Confusion Matrix
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib
matplotlib.use("Agg")  # Headless non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

class Phase4Visualizer:
    """Visualization generator for Phase 4 evaluation."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_plots(
        self,
        mappings: List[Dict[str, Any]],
        runtime_dict: Dict[str, float],
        learned_weights: Dict[str, float]
    ) -> List[Path]:
        """Generates all 6 publication-quality plots."""
        logger.info(f"Generating Phase 4 evaluation plots in {self.output_dir}...")
        generated = []

        try:
            p1 = self.plot_similarity_distribution(mappings)
            generated.append(p1)
        except Exception as e:
            logger.error(f"Failed to plot similarity distribution: {e}")

        try:
            p2 = self.plot_confidence_distribution(mappings)
            generated.append(p2)
        except Exception as e:
            logger.error(f"Failed to plot confidence distribution: {e}")

        try:
            p3 = self.plot_mapping_type_distribution(mappings)
            generated.append(p3)
        except Exception as e:
            logger.error(f"Failed to plot mapping type distribution: {e}")

        try:
            p4 = self.plot_ontology_coverage(mappings)
            generated.append(p4)
        except Exception as e:
            logger.error(f"Failed to plot ontology coverage: {e}")

        try:
            p5 = self.plot_runtime_breakdown(runtime_dict)
            generated.append(p5)
        except Exception as e:
            logger.error(f"Failed to plot runtime breakdown: {e}")

        try:
            p6 = self.plot_confusion_matrix(mappings)
            generated.append(p6)
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}")

        return generated

    def plot_similarity_distribution(self, mappings: List[Dict[str, Any]]) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        lex = [m.get("lexical_score", 0.0) for m in mappings]
        struc = [m.get("structural_score", 0.0) for m in mappings]
        prop = [m.get("property_score", 0.0) for m in mappings]
        reas = [m.get("reasoning_score", 0.0) for m in mappings]

        ax.hist(lex, bins=15, alpha=0.6, label="Lexical Similarity", color="#1f77b4")
        ax.hist(struc, bins=15, alpha=0.6, label="Structural Similarity", color="#2ca02c")
        ax.hist(prop, bins=15, alpha=0.4, label="Property Similarity", color="#ff7f0e")
        ax.hist(reas, bins=15, alpha=0.4, label="Reasoning Score", color="#9467bd")

        ax.set_title("Distribution of Feature Similarities [L, S, P, R]", fontsize=12, fontweight="bold")
        ax.set_xlabel("Similarity Score", fontsize=10)
        ax.set_ylabel("Count of Candidate Pairs", fontsize=10)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)

        out_path = self.output_dir / "similarity_distribution.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_confidence_distribution(self, mappings: List[Dict[str, Any]]) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        confs = [m.get("overall_confidence", m.get("confidence_score", 0.0)) for m in mappings]
        
        n, bins, patches = ax.hist(confs, bins=20, color="#2b5c8f", edgecolor="black", alpha=0.8)
        
        # Highlight high vs low confidence zones
        for i in range(len(patches)):
            if bins[i] >= 70.0:
                patches[i].set_facecolor("#2e7d32")
            elif bins[i] >= 50.0:
                patches[i].set_facecolor("#f57c00")
            else:
                patches[i].set_facecolor("#d32f2f")

        ax.set_title("Learned Confidence Score Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Confidence Score (%)", fontsize=10)
        ax.set_ylabel("Number of Mappings", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        out_path = self.output_dir / "confidence_distribution.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_mapping_type_distribution(self, mappings: List[Dict[str, Any]]) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        skos_types = {}
        for m in mappings:
            rel = m.get("skos_relation", "") or str(m.get("ontology_path", "")).split("#")[-1]
            if not rel:
                rel = m.get("relationship", "relatedMatch")
            skos_types[rel] = skos_types.get(rel, 0) + 1

        labels = list(skos_types.keys())
        counts = list(skos_types.values())
        colors = ["#4caf50", "#2196f3", "#ff9800", "#9c27b0", "#f44336"][:len(labels)]

        bars = ax.bar(labels, counts, color=colors, edgecolor="black", alpha=0.85)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, int(yval), ha='center', va='bottom', fontweight='bold')

        ax.set_title("Distribution of SKOS Mapping Types", fontsize=12, fontweight="bold")
        ax.set_xlabel("SKOS Mapping Relation", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        out_path = self.output_dir / "mapping_type_distribution.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_ontology_coverage(self, mappings: List[Dict[str, Any]]) -> Path:
        fig, ax = plt.subplots(figsize=(7, 5))
        
        total_brsr_disclosures = 74
        mapped_brsr = len(set(m.get("brsr_id", m.get("brsr_uri")) for m in mappings))
        unmapped_brsr = max(0, total_brsr_disclosures - mapped_brsr)

        sizes = [mapped_brsr, unmapped_brsr]
        labels = [f"Mapped BRSR ({mapped_brsr})", f"Unmapped BRSR ({unmapped_brsr})"]
        colors = ["#388e3c", "#d32f2f"]

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, explode=(0.05, 0))
        ax.set_title("BRSR Ontology Coverage Analysis", fontsize=12, fontweight="bold")

        out_path = self.output_dir / "ontology_coverage.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_runtime_breakdown(self, runtime_dict: Dict[str, float]) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        stages = list(runtime_dict.keys())
        times = list(runtime_dict.values())

        bars = ax.barh(stages, times, color="#1565c0", edgecolor="black", alpha=0.85)
        for bar in bars:
            xval = bar.get_width()
            ax.text(xval + 0.1, bar.get_y() + bar.get_height()/2.0, f"{xval:.2f}s", ha='left', va='center', fontweight='bold')

        ax.set_title("Pipeline Execution Time Breakdown", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        out_path = self.output_dir / "runtime_breakdown.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_confusion_matrix(self, mappings: List[Dict[str, Any]]) -> Path:
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Calculate confusion matrix between SKOS mapping decision & LLM Verification decision
        # Accepted + High Conf => TP, Accepted + Low Conf => FP, Rejected + High Conf => FN, Rejected + Low Conf => TN
        tp = sum(1 for m in mappings if m.get("llm_verification") in ["Agree", "Accepted"] and m.get("overall_confidence", 0) >= 40)
        fp = sum(1 for m in mappings if m.get("llm_verification") in ["Agree", "Accepted"] and m.get("overall_confidence", 0) < 40)
        fn = sum(1 for m in mappings if m.get("llm_verification") in ["Disagree", "Rejected"] and m.get("overall_confidence", 0) >= 40)
        tn = sum(1 for m in mappings if m.get("llm_verification") in ["Disagree", "Rejected"] and m.get("overall_confidence", 0) < 40)

        cm = np.array([[tp, fp], [fn, tn]])
        
        cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
        fig.colorbar(cax)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontweight="bold", fontsize=14)

        ax.set_xticklabels(["", "Positive", "Negative"])
        ax.set_yticklabels(["", "Accepted", "Rejected"])
        ax.set_xlabel("Ontology Matcher Prediction", fontsize=10)
        ax.set_ylabel("LLM Verification Audit", fontsize=10)
        ax.set_title("Confusion Matrix (Matcher vs LLM Audit)", fontsize=12, fontweight="bold", pad=20)

        out_path = self.output_dir / "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path
