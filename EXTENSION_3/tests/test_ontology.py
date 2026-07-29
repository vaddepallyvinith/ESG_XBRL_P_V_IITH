"""
test_ontology.py - Unit tests for the ontology builder and RSO schema.
"""
import pytest
from rdflib import Graph
from ontology.schema import create_base_graph, RSO, SCHEMA
from ontology.builder import OntologyBuilder
import os
import tempfile
import json

@pytest.fixture
def dummy_brsr_json(tmp_path):
    data = {
        "framework": "BRSR",
        "source_file": "dummy_brsr.pdf",
        "content": ["Introduction to BRSR."],
        "sections": [
            {
                "section_id": "Section A",
                "label": "General Disclosures",
                "content": ["General disclosure section."],
                "page_start": 1,
                "page_end": 2,
                "principles": [
                    {
                        "principle_num": 1,
                        "label": "Ethics",
                        "content": ["Principle 1 description."],
                        "page_start": 2,
                        "page_end": 3,
                        "indicator_groups": [
                            {
                                "group_type": "Essential",
                                "disclosures": [
                                    {
                                        "id": "1",
                                        "label": "Training provided",
                                        "text": "Percentage of employees trained.",
                                        "content": ["Additional guidance."],
                                        "page_start": 3,
                                        "page_end": 3
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }
    file_path = tmp_path / "dummy_brsr.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return tmp_path

def test_schema_creation():
    g = create_base_graph()
    assert len(g) > 10
    
    # Check if classes are defined
    classes = [sub for sub, pred, obj in g.triples((None, None, None)) if obj.endswith("Class")]
    assert RSO.Framework in classes
    assert RSO.Topic in classes
    assert RSO.Disclosure in classes
    assert RSO.Requirement in classes
    
def test_ontology_builder(dummy_brsr_json):
    builder = OntologyBuilder(str(dummy_brsr_json))
    graph = builder.build()
    
    # Check nodes
    assert len(builder.nodes_data) == 4
    
    node_labels = [n["label:LABEL"] for n in builder.nodes_data]
    assert "Framework" in node_labels
    assert "Topic" in node_labels # Section
    assert "Disclosure" in node_labels
    
    # Check edges
    assert len(builder.edges_data) == 6 # 3 pairs of bidirectional edges
    
    # Verify specific belongsTo subproperties are used in the graph
    predicates = set(graph.predicates(None, None))
    assert RSO.belongsToFramework in predicates
    assert RSO.belongsToTopic in predicates
    
    # Check provenance
    disc_node = next(n for n in builder.nodes_data if n["label:LABEL"] == "Disclosure")
    assert disc_node["pageStart"] == 3
    assert disc_node["pageEnd"] == 3
    assert disc_node["sourceDocument"] == "dummy_brsr.pdf"
    
def test_neo4j_csv_export(dummy_brsr_json):
    builder = OntologyBuilder(str(dummy_brsr_json))
    builder.build()
    
    out_dir = dummy_brsr_json / "graph"
    builder.export_neo4j_csv(str(out_dir))
    
    assert os.path.exists(out_dir / "neo4j_nodes.csv")
    assert os.path.exists(out_dir / "neo4j_relationships.csv")
