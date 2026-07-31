"""
schema.py - Defines the unified RSO-inspired RDF namespaces and ontology structure.
"""
from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef
import re

# Define Namespaces
RSO = Namespace("http://example.org/ontology/rso#")
SCHEMA = Namespace("http://schema.org/")

def create_base_graph() -> Graph:
    """Initialize a Graph with the required namespaces bound and define the ontology structure."""
    g = Graph()
    g.bind("rso", RSO)
    g.bind("schema", SCHEMA)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)

    # ---------------------------------------------------------
    # Define RSO Ontology Classes
    # ---------------------------------------------------------
    g.add((RSO.Framework, RDF.type, OWL.Class))
    g.add((RSO.Topic, RDF.type, OWL.Class))
    g.add((RSO.Disclosure, RDF.type, OWL.Class))
    g.add((RSO.Requirement, RDF.type, OWL.Class))
    g.add((RSO.Metric, RDF.type, OWL.Class))
    g.add((RSO.Unit, RDF.type, OWL.Class))
    g.add((RSO.Definition, RDF.type, OWL.Class))
    g.add((RSO.Dimension, RDF.type, OWL.Class))
    g.add((RSO.Applicability, RDF.type, OWL.Class))

    # ---------------------------------------------------------
    # Define Object Properties (Relationships)
    # ---------------------------------------------------------
    g.add((RSO.belongsTo, RDF.type, OWL.ObjectProperty))
    g.add((RSO.belongsToFramework, RDF.type, OWL.ObjectProperty))
    g.add((RSO.belongsToFramework, RDFS.subPropertyOf, RSO.belongsTo))
    g.add((RSO.belongsToTopic, RDF.type, OWL.ObjectProperty))
    g.add((RSO.belongsToTopic, RDFS.subPropertyOf, RSO.belongsTo))
    g.add((RSO.belongsToDisclosure, RDF.type, OWL.ObjectProperty))
    g.add((RSO.belongsToDisclosure, RDFS.subPropertyOf, RSO.belongsTo))
    
    g.add((RSO.contains, RDF.type, OWL.ObjectProperty))
    g.add((RSO.requires, RDF.type, OWL.ObjectProperty))
    g.add((RSO.hasMetric, RDF.type, OWL.ObjectProperty))
    g.add((RSO.hasUnit, RDF.type, OWL.ObjectProperty))
    g.add((RSO.definedBy, RDF.type, OWL.ObjectProperty))
    g.add((RSO.childOf, RDF.type, OWL.ObjectProperty))
    g.add((RSO.parentOf, RDF.type, OWL.ObjectProperty))
    g.add((RSO.hasDatatype, RDF.type, OWL.DatatypeProperty))
    g.add((RSO.hasApplicability, RDF.type, OWL.DatatypeProperty))

    # ---------------------------------------------------------
    # Define Data Properties (Attributes & Provenance)
    # ---------------------------------------------------------
    g.add((SCHEMA.identifier, RDF.type, OWL.DatatypeProperty))
    g.add((SCHEMA.name, RDF.type, OWL.DatatypeProperty))
    g.add((SCHEMA.text, RDF.type, OWL.DatatypeProperty))
    
    # Provenance fields
    g.add((RSO.sourceDocument, RDF.type, OWL.DatatypeProperty))
    g.add((RSO.pageStart, RDF.type, OWL.DatatypeProperty))
    g.add((RSO.pageEnd, RDF.type, OWL.DatatypeProperty))
    g.add((RSO.sectionLabel, RDF.type, OWL.DatatypeProperty))
    g.add((RSO.originalText, RDF.type, OWL.DatatypeProperty))

    return g

def clean_uri(text: str) -> str:
    """Helper to generate valid URIs from labels."""
    if not text:
        return "unknown"
    # Remove special chars and spaces
    cleaned = "".join(c if c.isalnum() else "_" for c in text.strip())
    # Merge consecutive underscores
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned
