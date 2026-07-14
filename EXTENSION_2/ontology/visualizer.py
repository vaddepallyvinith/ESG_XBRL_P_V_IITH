"""
visualizer.py - Generates Mermaid diagrams from ontology CSV data.
"""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def generate_mermaid_diagram(nodes: List[Dict], edges: List[Dict], output_path: str, max_nodes: int = 150):
    """
    Generates a Mermaid graph representation of the ontology.
    Limits the number of nodes to avoid crashing markdown renderers.
    """
    if not nodes:
        return
        
    logger.info(f"Generating Mermaid visualization for up to {max_nodes} nodes...")
    
    # Filter nodes
    filtered_nodes = nodes[:max_nodes]
    valid_node_ids = {n["id:ID"] for n in filtered_nodes}
    
    # Filter edges connecting the valid nodes
    filtered_edges = [
        e for e in edges 
        if e[":START_ID"] in valid_node_ids and e[":END_ID"] in valid_node_ids
    ]
    
    mermaid_lines = [
        "```mermaid",
        "graph TD",
        "    classDef framework fill:#f9f,stroke:#333,stroke-width:2px;",
        "    classDef topic fill:#bbf,stroke:#333,stroke-width:2px;",
        "    classDef disclosure fill:#bfb,stroke:#333,stroke-width:1px;",
        "    classDef requirement fill:#fbb,stroke:#333,stroke-width:1px;"
    ]
    
    for n in filtered_nodes:
        nid = n["id:ID"].replace("-", "_").replace(".", "_")
        label = n["name"][:30].replace('"', "'") + ("..." if len(n["name"]) > 30 else "")
        node_class = n["label:LABEL"].lower()
        
        # Only add valid node lines
        mermaid_lines.append(f'    {nid}["{label}"]:::{node_class}')
        
    for e in filtered_edges:
        start_id = e[":START_ID"].replace("-", "_").replace(".", "_")
        end_id = e[":END_ID"].replace("-", "_").replace(".", "_")
        rel_type = e[":TYPE"]
        mermaid_lines.append(f'    {start_id} -- "{rel_type}" --> {end_id}')
        
    mermaid_lines.append("```")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(mermaid_lines))
        
    logger.info(f"✅ Exported Mermaid visualization to {output_path}")
