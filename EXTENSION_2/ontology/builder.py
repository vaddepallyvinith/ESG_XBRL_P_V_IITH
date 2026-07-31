"""
builder.py - Parses processed JSON files and builds the unified RSO RDF knowledge graph.
"""
import json
import logging
import csv
from pathlib import Path
from typing import Dict, Any, List

from rdflib import Graph, URIRef, Literal, RDF, XSD
from ontology.schema import create_base_graph, RSO, SCHEMA, clean_uri

logger = logging.getLogger(__name__)

class OntologyBuilder:
    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)
        self.graph = create_base_graph()
        self.nodes_data = []
        self.edges_data = []

    def build(self):
        """Iterate over all JSON files and populate the graph."""
        json_files = list(self.processed_dir.glob("*.json"))
        logger.info(f"Building ontology from {len(json_files)} files in {self.processed_dir}")
        for json_path in json_files:
            if json_path.name == "manifest.json":
                continue
                
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                framework = data.get("framework", "")
                if framework == "BRSR":
                    self._parse_brsr(data)
                elif framework == "GRI":
                    self._parse_gri(data)
                else:
                    logger.warning(f"Unknown framework in {json_path.name}")
            except Exception as e:
                logger.error(f"Error parsing {json_path.name}: {e}")
                
        return self.graph
        
    def _add_node(self, uri: URIRef, rso_class: URIRef, properties: Dict[str, Any]):
        """Adds a node to the RDF graph and internal Neo4j CSV state."""
        self.graph.add((uri, RDF.type, rso_class))
        
        # Provenance & core properties
        if "identifier" in properties:
            self.graph.add((uri, SCHEMA.identifier, Literal(properties["identifier"], datatype=XSD.string)))
        if "name" in properties:
            self.graph.add((uri, SCHEMA.name, Literal(properties["name"], datatype=XSD.string)))
        if "text" in properties and properties["text"]:
            self.graph.add((uri, RSO.originalText, Literal(properties["text"], datatype=XSD.string)))
            self.graph.add((uri, SCHEMA.text, Literal(properties["text"], datatype=XSD.string)))
        if "sourceDocument" in properties:
            self.graph.add((uri, RSO.sourceDocument, Literal(properties["sourceDocument"], datatype=XSD.string)))
        if "pageStart" in properties:
            self.graph.add((uri, RSO.pageStart, Literal(properties["pageStart"], datatype=XSD.integer)))
        if "pageEnd" in properties:
            self.graph.add((uri, RSO.pageEnd, Literal(properties["pageEnd"], datatype=XSD.integer)))
        if "sectionLabel" in properties:
            self.graph.add((uri, RSO.sectionLabel, Literal(properties["sectionLabel"], datatype=XSD.string)))
        if "datatype" in properties and properties["datatype"]:
            self.graph.add((uri, RSO.hasDatatype, Literal(properties["datatype"], datatype=XSD.string)))
        if "applicability" in properties and properties["applicability"]:
            self.graph.add((uri, RSO.hasApplicability, Literal(properties["applicability"], datatype=XSD.string)))
            
        # Add to Neo4j list
        node_id = str(uri).replace(str(RSO), "")
        class_name = str(rso_class).replace(str(RSO), "")
        self.nodes_data.append({
            "id:ID": node_id,
            "name": properties.get("name", ""),
            "label:LABEL": class_name,
            "identifier": properties.get("identifier", ""),
            "text": properties.get("text", "").replace("\\n", " ") if properties.get("text") else "",
            "sourceDocument": properties.get("sourceDocument", ""),
            "pageStart": properties.get("pageStart", ""),
            "pageEnd": properties.get("pageEnd", ""),
            "sectionLabel": properties.get("sectionLabel", "")
        })
        
    def _add_edge(self, source_uri: URIRef, target_uri: URIRef, relation: URIRef):
        """Adds an edge to the RDF graph and internal Neo4j CSV state."""
        self.graph.add((source_uri, relation, target_uri))
        
        source_id = str(source_uri).replace(str(RSO), "")
        target_id = str(target_uri).replace(str(RSO), "")
        rel_type = str(relation).replace(str(RSO), "")
        self.edges_data.append({
            ":START_ID": source_id,
            ":END_ID": target_id,
            ":TYPE": rel_type
        })

    def _parse_brsr(self, data: Dict[str, Any]):
        doc_id = clean_uri(data.get("source_file", "brsr_doc"))
        doc_uri = RSO[f"Framework_{doc_id}"]
        source_file = data.get("source_file", "")
        
        doc_content = "\\n".join(data.get("content", []))
        
        self._add_node(doc_uri, RSO.Framework, {
            "identifier": "BRSR",
            "name": "Business Responsibility and Sustainability Report",
            "sourceDocument": source_file,
            "text": doc_content
        })
        
        # BRSR sections -> Topic
        for sec in data.get("sections", []):
            sec_id = clean_uri(sec.get("section_id", ""))
            if not sec_id:
                sec_id = clean_uri(sec.get("label", ""))[:20]
            sec_uri = RSO[f"Topic_{sec_id}"]
            
            sec_content = "\\n".join(sec.get("content", []))
            
            self._add_node(sec_uri, RSO.Topic, {
                "identifier": sec.get("section_id", ""),
                "name": sec.get("label", ""),
                "sourceDocument": source_file,
                "pageStart": sec.get("page_start", 0),
                "pageEnd": sec.get("page_end", 0),
                "sectionLabel": sec.get("label", ""),
                "text": sec_content
            })
            self._add_edge(doc_uri, sec_uri, RSO.contains)
            self._add_edge(sec_uri, doc_uri, RSO.belongsToFramework)
            
            # Non-principle disclosures (Questions in Section A & B) -> Disclosure
            for disc in sec.get("disclosures", []):
                self._parse_brsr_question(sec_uri, disc, source_file)
                
            # Principles (Section C) -> Topic (child of Section Topic)
            for prin in sec.get("principles", []):
                prin_id = clean_uri(prin.get("label", ""))
                prin_uri = RSO[f"Topic_Principle_{prin_id}"]
                
                prin_text = prin.get("text", "")
                prin_content = prin.get("content", [])
                if prin_content:
                    prin_text += "\\n" + "\\n".join(prin_content)
                    
                self._add_node(prin_uri, RSO.Topic, {
                    "identifier": str(prin.get("principle_num", "")),
                    "name": prin.get("label", ""),
                    "sourceDocument": source_file,
                    "pageStart": prin.get("page_start", 0),
                    "pageEnd": prin.get("page_end", 0),
                    "sectionLabel": prin.get("label", ""),
                    "text": prin_text
                })
                self._add_edge(sec_uri, prin_uri, RSO.contains)
                self._add_edge(prin_uri, sec_uri, RSO.childOf)
                
                # Indicator Groups (Essential / Leadership) contain disclosures
                for igroup in prin.get("indicator_groups", []):
                    for disc in igroup.get("disclosures", []):
                        self._parse_brsr_question(prin_uri, disc, source_file)

    def _parse_brsr_question(self, parent_uri: URIRef, disc: Dict[str, Any], source_file: str):
        q_id = clean_uri(disc.get("id", ""))
        q_uri = RSO[f"Disclosure_{q_id}_{hash(disc.get('text', ''))}"]
        
        q_text = disc.get("text", "")
        q_content = disc.get("content", [])
        if q_content:
            q_text += "\\n" + "\\n".join(q_content)
            
        self._add_node(q_uri, RSO.Disclosure, {
            "identifier": disc.get("id", ""),
            "name": disc.get("label", ""),
            "sourceDocument": source_file,
            "pageStart": disc.get("page_start", 0),
            "pageEnd": disc.get("page_end", 0),
            "sectionLabel": disc.get("label", ""),
            "text": q_text,
            "datatype": disc.get("datatype"),
            "applicability": disc.get("applicability")
        })
        self._add_edge(parent_uri, q_uri, RSO.contains)
        self._add_edge(q_uri, parent_uri, RSO.belongsToTopic)
        
        if disc.get("metric"):
            m_uri = RSO[f"Metric_{clean_uri(disc.get('metric'))}"]
            self._add_node(m_uri, RSO.Metric, {"name": disc.get("metric"), "identifier": disc.get("metric")})
            self._add_edge(q_uri, m_uri, RSO.hasMetric)
            
        if disc.get("unit"):
            u_uri = RSO[f"Unit_{clean_uri(disc.get('unit'))}"]
            self._add_node(u_uri, RSO.Unit, {"name": disc.get("unit"), "identifier": disc.get("unit")})
            self._add_edge(q_uri, u_uri, RSO.hasUnit)
            
    def _parse_gri(self, data: Dict[str, Any]):
        doc_id = clean_uri(data.get("source_file", "gri_doc"))
        doc_uri = RSO[f"Framework_{doc_id}"]
        source_file = data.get("source_file", "")
        
        doc_content = "\\n".join(data.get("content", []))
        
        self._add_node(doc_uri, RSO.Framework, {
            "identifier": "GRI",
            "name": source_file.replace(".pdf", ""),
            "sourceDocument": source_file,
            "text": doc_content
        })
        
        for std in data.get("standards", []):
            std_title = std.get("title", "")
            if not std_title:
                continue
                
            std_id = clean_uri(std.get("standard_id", ""))
            std_uri = RSO[f"Topic_{std_id}"]
            
            std_content = "\\n".join(std.get("content", []))
            
            self._add_node(std_uri, RSO.Topic, {
                "identifier": std.get("standard_id", ""),
                "name": std_title,
                "sourceDocument": source_file,
                "pageStart": std.get("page_start", 0),
                "pageEnd": std.get("page_end", 0),
                "sectionLabel": std_title,
                "text": std_content
            })
            self._add_edge(doc_uri, std_uri, RSO.contains)
            self._add_edge(std_uri, doc_uri, RSO.belongsToFramework)
            
            for disc in std.get("disclosures", []):
                disc_id = clean_uri(disc.get("id", ""))
                
                # If "topic" is in the ID, it acts as a Topic in GRI, but structurally it's a child topic
                # Otherwise, it's a regular disclosure.
                is_topic = "topic" in disc_id.lower()
                rso_class = RSO.Topic if is_topic else RSO.Disclosure
                disc_uri = RSO[f"{rso_class.split('#')[-1]}_{disc_id}"]
                
                disc_text = disc.get("text", "")
                disc_content = disc.get("content", [])
                if disc_content:
                    disc_text += "\\n" + "\\n".join(disc_content)
                    
                self._add_node(disc_uri, rso_class, {
                    "identifier": disc.get("id", ""),
                    "name": disc.get("label", ""),
                    "sourceDocument": source_file,
                    "pageStart": disc.get("page_start", 0),
                    "pageEnd": disc.get("page_end", 0),
                    "sectionLabel": disc.get("label", ""),
                    "text": disc_text,
                    "datatype": disc.get("datatype"),
                    "applicability": disc.get("applicability")
                })
                self._add_edge(std_uri, disc_uri, RSO.contains)
                self._add_edge(disc_uri, std_uri, RSO.belongsToTopic)
                
                if disc.get("metric"):
                    m_uri = RSO[f"Metric_{clean_uri(disc.get('metric'))}"]
                    self._add_node(m_uri, RSO.Metric, {"name": disc.get("metric"), "identifier": disc.get("metric")})
                    self._add_edge(disc_uri, m_uri, RSO.hasMetric)
                    
                if disc.get("unit"):
                    u_uri = RSO[f"Unit_{clean_uri(disc.get('unit'))}"]
                    self._add_node(u_uri, RSO.Unit, {"name": disc.get("unit"), "identifier": disc.get("unit")})
                    self._add_edge(disc_uri, u_uri, RSO.hasUnit)
                
                # Requirements
                for req in disc.get("requirements", []):
                    req_text = "\\n".join(req.get("content", []))
                    req_uri = RSO[f"Requirement_{hash(req_text)}"]
                    
                    self._add_node(req_uri, RSO.Requirement, {
                        "identifier": req.get("type", ""),
                        "name": req.get("type", ""),
                        "sourceDocument": source_file,
                        "pageStart": req.get("page_start", 0),
                        "pageEnd": req.get("page_end", 0),
                        "sectionLabel": req.get("type", ""),
                        "text": req_text,
                        "datatype": req.get("datatype"),
                        "applicability": req.get("applicability")
                    })
                    self._add_edge(disc_uri, req_uri, RSO.requires)
                    self._add_edge(req_uri, disc_uri, RSO.belongsToDisclosure)
                    
                    if req.get("metric"):
                        m_uri = RSO[f"Metric_{clean_uri(req.get('metric'))}"]
                        self._add_node(m_uri, RSO.Metric, {"name": req.get("metric"), "identifier": req.get("metric")})
                        self._add_edge(req_uri, m_uri, RSO.hasMetric)
                        
                    if req.get("unit"):
                        u_uri = RSO[f"Unit_{clean_uri(req.get('unit'))}"]
                        self._add_node(u_uri, RSO.Unit, {"name": req.get("unit"), "identifier": req.get("unit")})
                        self._add_edge(req_uri, u_uri, RSO.hasUnit)

    def export_neo4j_csv(self, output_dir: str):
        """Export nodes and relationships to CSV for Neo4j import."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        nodes_file = out_path / "neo4j_nodes.csv"
        edges_file = out_path / "neo4j_relationships.csv"
        
        if self.nodes_data:
            keys = self.nodes_data[0].keys()
            with open(nodes_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.nodes_data)
        
        if self.edges_data:
            keys = self.edges_data[0].keys()
            with open(edges_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.edges_data)
                
        logger.info(f"✅ Exported Neo4j CSVs to {output_dir}")
