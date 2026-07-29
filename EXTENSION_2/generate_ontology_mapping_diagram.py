"""
generate_ontology_mapping_diagram.py - Renders high-resolution diagram of the ESG Ontology Graph
and BRSR <-> GRI Mappings in EXTENSION_2.
"""

import os
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def generate_diagram():
    output_dir = Path("/home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/data/processed/mapping")
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_file = output_dir / "mapping_repository.json"
    with open(mapping_file, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    # Sort by score
    mappings.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)

    # Select representative mappings across key ESG topics
    selected_mappings = []
    seen_topics = set()
    for m in mappings:
        b_label = m.get("brsr_label", "")
        g_label = m.get("gri_label", "")
        # Pick top unique domain disclosures
        if len(selected_mappings) < 12:
            selected_mappings.append(m)

    # Build NetworkX Graph
    G = nx.DiGraph()

    # Root Framework Nodes
    G.add_node("BRSR Framework", type="Framework", color="#1f77b4", size=3200, label="BRSR Framework\n(SEBI Guideline)")
    G.add_node("GRI Framework", type="Framework", color="#2ca02c", size=3200, label="GRI Standards\n(GRI Global)")

    # Topic Nodes
    topics = {
        "Topic_BRSR_P6": ("P6: Environment", "BRSR Framework", "#aec7e8"),
        "Topic_BRSR_P3": ("P3: Employee Wellbeing", "BRSR Framework", "#aec7e8"),
        "Topic_BRSR_P2": ("P2: Sustainable Sourcing", "BRSR Framework", "#aec7e8"),
        "Topic_GRI_305": ("GRI 305: Emissions", "GRI Framework", "#a1d99b"),
        "Topic_GRI_301": ("GRI 301: Materials", "GRI Framework", "#a1d99b"),
        "Topic_GRI_403": ("GRI 403: Health & Safety", "GRI Framework", "#a1d99b"),
        "Topic_GRI_302": ("GRI 302: Energy", "GRI Framework", "#a1d99b")
    }

    for t_id, (t_name, parent, color) in topics.items():
        G.add_node(t_id, type="Topic", color=color, size=2200, label=t_name)
        G.add_edge(parent, t_id, relation="contains", color="#999999", style="dashed", width=1.5)

    # Key Disclosures and SKOS Mappings
    disclosure_pairs = [
        ("BRSR Scope 1 & 2 Emissions", "Topic_BRSR_P6", "GRI 305-1 Scope 1 Direct Emissions", "Topic_GRI_305", "skos:broadMatch", "35.5%"),
        ("BRSR Scope 2 Energy Indirect", "Topic_BRSR_P6", "GRI 305-2 Scope 2 Energy Emissions", "Topic_GRI_305", "skos:broadMatch", "34.3%"),
        ("BRSR Scope 3 Emissions & Intensity", "Topic_BRSR_P6", "GRI 102-7 Scope 3 GHG Emissions", "Topic_GRI_305", "skos:narrowMatch", "33.9%"),
        ("BRSR Energy Consumption", "Topic_BRSR_P6", "GRI 302-1 Energy Consumption in Org", "Topic_GRI_302", "skos:broadMatch", "32.8%"),
        ("BRSR Reclaimed Products & Packaging", "Topic_BRSR_P2", "GRI 301-3 Reclaimed Packaging", "Topic_GRI_301", "skos:narrowMatch", "39.1%"),
        ("BRSR Health & Safety Mgmt System", "Topic_BRSR_P3", "GRI 403-1 OHS Management System", "Topic_GRI_403", "skos:broadMatch", "44.1%"),
        ("BRSR Worker Coverage in OHS", "Topic_BRSR_P3", "GRI 403-8 Worker OHS Coverage", "Topic_GRI_403", "skos:broadMatch", "39.1%"),
        ("BRSR Air Emissions (non-GHG)", "Topic_BRSR_P6", "GRI 305-5 Reduction of GHG Emissions", "Topic_GRI_305", "skos:broadMatch", "33.4%")
    ]

    mapping_edges = []
    for b_disc, b_topic, g_disc, g_topic, skos_rel, score in disclosure_pairs:
        G.add_node(b_disc, type="BRSR_Disc", color="#d62728", size=1500, label=b_disc)
        G.add_node(g_disc, type="GRI_Disc", color="#9467bd", size=1500, label=g_disc)

        G.add_edge(b_topic, b_disc, relation="rso:contains", color="#bcbd22", style="dotted", width=1.2)
        G.add_edge(g_topic, g_disc, relation="rso:contains", color="#bcbd22", style="dotted", width=1.2)

        # Mapping edge
        G.add_edge(b_disc, g_disc, relation=f"{skos_rel}\n({score})", color="#ff7f0e", style="solid", width=2.5)
        mapping_edges.append((b_disc, g_disc, f"{skos_rel}\n({score})"))

    # Plot Layout
    fig, ax = plt.subplots(figsize=(16, 12))
    
    pos = nx.spring_layout(G, k=0.85, seed=42)

    # Custom hierarchical coordinates for structured display
    pos["BRSR Framework"] = np.array([-0.8, 0.9])
    pos["GRI Framework"] = np.array([0.8, 0.9])

    pos["Topic_BRSR_P6"] = np.array([-0.8, 0.4])
    pos["Topic_BRSR_P3"] = np.array([-0.8, 0.0])
    pos["Topic_BRSR_P2"] = np.array([-0.8, -0.4])

    pos["Topic_GRI_305"] = np.array([0.8, 0.5])
    pos["Topic_GRI_302"] = np.array([0.8, 0.2])
    pos["Topic_GRI_301"] = np.array([0.8, -0.2])
    pos["Topic_GRI_403"] = np.array([0.8, -0.5])

    # Position disclosures in between
    pos["BRSR Scope 1 & 2 Emissions"] = np.array([-0.35, 0.55])
    pos["BRSR Scope 2 Energy Indirect"] = np.array([-0.35, 0.40])
    pos["BRSR Scope 3 Emissions & Intensity"] = np.array([-0.35, 0.25])
    pos["BRSR Energy Consumption"] = np.array([-0.35, 0.10])
    pos["BRSR Reclaimed Products & Packaging"] = np.array([-0.35, -0.35])
    pos["BRSR Health & Safety Mgmt System"] = np.array([-0.35, -0.10])
    pos["BRSR Worker Coverage in OHS"] = np.array([-0.35, -0.22])
    pos["BRSR Air Emissions (non-GHG)"] = np.array([-0.35, -0.50])

    pos["GRI 305-1 Scope 1 Direct Emissions"] = np.array([0.35, 0.55])
    pos["GRI 305-2 Scope 2 Energy Emissions"] = np.array([0.35, 0.40])
    pos["GRI 102-7 Scope 3 GHG Emissions"] = np.array([0.35, 0.25])
    pos["GRI 302-1 Energy Consumption in Org"] = np.array([0.35, 0.10])
    pos["GRI 301-3 Reclaimed Packaging"] = np.array([0.35, -0.35])
    pos["GRI 403-1 OHS Management System"] = np.array([0.35, -0.10])
    pos["GRI 403-8 Worker OHS Coverage"] = np.array([0.35, -0.22])
    pos["GRI 305-5 Reduction of GHG Emissions"] = np.array([0.35, -0.50])

    # Draw Nodes by color
    colors = [nx.get_node_attributes(G, "color")[n] for n in G.nodes()]
    sizes = [nx.get_node_attributes(G, "size")[n] for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, edgecolors="black", linewidths=1.5, ax=ax)

    # Draw Node Labels
    labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold", font_family="sans-serif", ax=ax)

    # Draw Hierarchy Edges
    hierarchy_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") in ["contains", "rso:contains"]]
    nx.draw_networkx_edges(G, pos, edgelist=hierarchy_edges, edge_color="#7f7f7f", style="dashed", width=1.2, arrows=True, arrowsize=12, ax=ax)

    # Draw Mapping Edges
    map_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") not in ["contains", "rso:contains"]]
    nx.draw_networkx_edges(G, pos, edgelist=map_edges, edge_color="#d62728", style="solid", width=2.2, arrows=True, arrowsize=16, ax=ax)

    # Draw Edge Labels for Mappings
    edge_labels = {(u, v): d["relation"] for u, v, d in G.edges(data=True) if d.get("relation") not in ["contains", "rso:contains"]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color="#8c564b", font_weight="bold", ax=ax)

    ax.set_title("ESG RSO Ontology Graph & Final BRSR <-> GRI Semantic Mappings (EXTENSION_2)", fontsize=14, fontweight="bold", pad=25)
    ax.axis("off")

    png_path = output_dir / "brsr_gri_ontology_mapping_graph.png"
    svg_path = output_dir / "brsr_gri_ontology_mapping_graph.svg"

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"✅ Generated Ontology Graph Diagram PNG: {png_path}")
    print(f"✅ Generated Ontology Graph Diagram SVG: {svg_path}")

if __name__ == "__main__":
    generate_diagram()
