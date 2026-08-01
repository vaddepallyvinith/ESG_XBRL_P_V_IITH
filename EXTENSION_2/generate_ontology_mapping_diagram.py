"""
generate_ontology_mapping_diagram.py - Renders high-resolution, crystal-clear SVG & PNG diagrams
of the ESG Ontology Graph and BRSR <-> GRI Mappings in EXTENSION_2 using automatically learned weights.
"""

import os
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

def generate_diagram():
    output_dir = Path("/home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/data/processed/mapping")
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_file = output_dir / "mapping_repository.json"
    if not mapping_file.exists():
        mapping_file = Path("EXTENSION_2/data/processed/mapping/mapping_repository.json")

    with open(mapping_file, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    # Sort mappings by learned similarity score
    mappings.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)

    # Top 10 representative BRSR <-> GRI mappings for diagram
    selected_mappings = mappings[:10]

    # Create figure with high DPI and crystal-clear layout
    fig, ax = plt.subplots(figsize=(24, 15), dpi=300)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")

    # Define Column X Positions
    x_brsr_framework = -1.2
    x_brsr_topic     = -0.8
    x_brsr_disc      = -0.35
    x_gri_disc       =  0.35
    x_gri_topic      =  0.8
    x_gri_framework  =  1.2

    # Draw Title & Legend Banner
    ax.text(0, 1.15, "BRSR ↔ GRI Standards Ontology Mapping Graph (EXTENSION_2)", 
            fontsize=20, fontweight="bold", ha="center", va="center", color="#0F172A", family="sans-serif")
    ax.text(0, 1.09, "Automatically Learned Confidence Weights Model: w_lex=0.4297, w_str=0.1935, w_prop=0.0000, w_emb=0.3768", 
            fontsize=12, fontstyle="italic", ha="center", va="center", color="#475569", family="sans-serif")

    # Framework Header Boxes
    bbox_brsr_hdr = dict(boxstyle="round,pad=0.6", facecolor="#1E3A8A", edgecolor="#1E40AF", lw=2)
    bbox_gri_hdr  = dict(boxstyle="round,pad=0.6", facecolor="#065F46", edgecolor="#047857", lw=2)

    ax.text(x_brsr_framework, 0.98, "BRSR FRAMEWORK\n(SEBI Principles P1–P9)", fontsize=13, fontweight="bold", 
            ha="center", va="center", color="white", bbox=bbox_brsr_hdr)
    ax.text(x_gri_framework, 0.98, "GRI STANDARDS\n(GRI 100–400 Series)", fontsize=13, fontweight="bold", 
            ha="center", va="center", color="white", bbox=bbox_gri_hdr)

    # Define Topic Categories
    brsr_topics = [
        ("P6: Environment & Energy", -0.85, 0.65, "#2563EB"),
        ("P3: Employee Wellbeing & OHS", -0.85, 0.0, "#2563EB"),
        ("P2: Sustainable Sourcing", -0.85, -0.60, "#2563EB")
    ]

    gri_topics = [
        ("GRI 305: Emissions & Climate", 0.85, 0.70, "#059669"),
        ("GRI 302: Energy Use", 0.85, 0.35, "#059669"),
        ("GRI 403: Health & Safety", 0.85, 0.0, "#059669"),
        ("GRI 301: Materials & Waste", 0.85, -0.55, "#059669")
    ]

    for label, x, y, color in brsr_topics:
        ax.text(x, y, label, fontsize=10, fontweight="bold", ha="center", va="center", color="white",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor="#1D4ED8", lw=1.5))
        # Edge from framework
        ax.annotate("", xy=(x + 0.12, y), xytext=(x_brsr_framework + 0.15, 0.94),
                    arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=1.5, linestyle="--"))

    for label, x, y, color in gri_topics:
        ax.text(x, y, label, fontsize=10, fontweight="bold", ha="center", va="center", color="white",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor="#047857", lw=1.5))
        ax.annotate("", xy=(x - 0.12, y), xytext=(x_gri_framework - 0.15, 0.94),
                    arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=1.5, linestyle="--"))

    # Mapping Rows Configuration
    y_positions = np.linspace(0.80, -0.80, len(selected_mappings))

    for i, m in enumerate(selected_mappings):
        y = y_positions[i]
        b_label = m.get("brsr_label", "BRSR Disclosure")
        if len(b_label) > 42:
            b_label = b_label[:40] + "..."

        g_label = m.get("gri_label", "GRI Disclosure")
        if len(g_label) > 42:
            g_label = g_label[:40] + "..."

        score = m.get("similarity_score", 0.0) * 100.0
        skos_rel = str(m.get("ontology_path", "")).split("#")[-1] or m.get("relationship", "closeMatch")

        # Choose color by SKOS relation
        if "exact" in skos_rel.lower() or "close" in skos_rel.lower():
            edge_color = "#059669" # Emerald green
            rel_badge = f"skos:{skos_rel}\n({score:.1f}%)"
            badge_bg = "#D1FAE5"
            badge_fg = "#065F46"
        else:
            edge_color = "#D97706" # Amber
            rel_badge = f"skos:{skos_rel}\n({score:.1f}%)"
            badge_bg = "#FEF3C7"
            badge_fg = "#92400E"

        # BRSR Disclosure Card
        ax.text(x_brsr_disc, y, b_label, fontsize=8.5, fontweight="bold", ha="right", va="center", color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#E0F2FE", edgecolor="#0284C7", lw=1.2))

        # GRI Disclosure Card
        ax.text(x_gri_disc, y, g_label, fontsize=8.5, fontweight="bold", ha="left", va="center", color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#F3E8FF", edgecolor="#7C3AED", lw=1.2))

        # SKOS Semantic Alignment Arrow
        ax.annotate("", xy=(x_gri_disc - 0.08, y), xytext=(x_brsr_disc + 0.08, y),
                    arrowprops=dict(arrowstyle="-|>", color=edge_color, lw=2.2, mutation_scale=15))

        # SKOS Relation & Score Badge
        ax.text(0, y, rel_badge, fontsize=8, fontweight="bold", ha="center", va="center", color=badge_fg,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=badge_bg, edgecolor=edge_color, lw=1))

    # Add Legend Box at Bottom
    legend_elements = [
        mpatches.Patch(facecolor="#1E3A8A", label="Framework Domain Node"),
        mpatches.Patch(facecolor="#2563EB", label="BRSR Principle Topic Node"),
        mpatches.Patch(facecolor="#059669", label="GRI Standard Topic Node"),
        mpatches.Patch(facecolor="#E0F2FE", edgecolor="#0284C7", label="BRSR Disclosure Requirement"),
        mpatches.Patch(facecolor="#F3E8FF", edgecolor="#7C3AED", label="GRI Target Disclosure"),
        mpatches.Patch(facecolor="#D1FAE5", edgecolor="#059669", label="skos:closeMatch (High Confidence)"),
        mpatches.Patch(facecolor="#FEF3C7", edgecolor="#D97706", label="skos:broadMatch / narrowMatch")
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=4, fontsize=9.5, frameon=True, facecolor="white", edgecolor="#CBD5E1")

    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-0.95, 1.22)
    ax.axis("off")

    png_path = output_dir / "brsr_gri_ontology_mapping_graph.png"
    svg_path = output_dir / "brsr_gri_ontology_mapping_graph.svg"

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"✅ Successfully generated high-resolution PNG Diagram: {png_path} ({png_path.stat().st_size:,} bytes)")
    print(f"✅ Successfully generated crystal-clear SVG Diagram: {svg_path} ({svg_path.stat().st_size:,} bytes)")

if __name__ == "__main__":
    generate_diagram()
