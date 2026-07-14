"""
neo4j_exporter.py - Translates an RDF Graph into Neo4j CSV imports.
"""
import csv
import logging
from pathlib import Path
from rdflib import Graph, URIRef, Literal, RDF
from ontology.schema import SCHEMA, BRSR, GRI

logger = logging.getLogger(__name__)

class Neo4jExporter:
    def __init__(self, graph: Graph, output_dir: str):
        self.graph = graph
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self):
        nodes_path = self.output_dir / "nodes.csv"
        edges_path = self.output_dir / "edges.csv"
        
        # 1. Export Nodes
        nodes = {} # dict of uri -> {id, label, identifier, name, text}
        for s, p, o in self.graph.triples((None, RDF.type, None)):
            if not isinstance(s, URIRef) or isinstance(o, Literal):
                continue
            if s not in nodes:
                nodes[s] = {
                    "id:ID": str(s),
                    ":LABEL": str(o).split("#")[-1],
                    "identifier": "",
                    "name": "",
                    "text": ""
                }
                
        # Fill data properties
        for s, p, o in self.graph:
            if s in nodes and isinstance(o, Literal):
                if p == SCHEMA.identifier:
                    nodes[s]["identifier"] = str(o).replace("\n", " ")
                elif p == SCHEMA.name:
                    nodes[s]["name"] = str(o).replace("\n", " ")
                elif p == SCHEMA.text:
                    nodes[s]["text"] = str(o).replace("\n", " ").replace("\"", "'")
                    
        with open(nodes_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id:ID", ":LABEL", "identifier", "name", "text"])
            writer.writeheader()
            for node in nodes.values():
                writer.writerow(node)
                
        logger.info(f"Exported {len(nodes)} nodes to {nodes_path.name}")
        
        # 2. Export Edges
        edges = []
        for s, p, o in self.graph:
            if s in nodes and o in nodes and p != RDF.type:
                rel_type = str(p).split("#")[-1]
                edges.append({
                    ":START_ID": str(s),
                    ":END_ID": str(o),
                    ":TYPE": rel_type
                })
                
        with open(edges_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[":START_ID", ":END_ID", ":TYPE"])
            writer.writeheader()
            for edge in edges:
                writer.writerow(edge)

        logger.info(f"Exported {len(edges)} edges to {edges_path.name}")
