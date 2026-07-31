"""
test_verification.py - Unit tests for Semantic Alignment Engine, Reasoner, and LLM Verification pipeline.
"""
import pytest
from matcher.models import OntologyConcept, MappingEvidence, MappingCandidate
from matcher.ontology_reasoner import OntologyReasoner
from verifier.llm_verifier import LLMVerifier
from matcher.engine import SemanticMappingEngine

def test_ontology_reasoner_logging():
    config = {
        "reasoner": {},
        "mapping": {}
    }
    reasoner = OntologyReasoner(config)
    
    # Create two concepts
    brsr_concept = OntologyConcept(
        uri="http://example.org/ontology/rso#Disclosure_Q1",
        framework="BRSR",
        label="Principle 6 Environmental indicators",
        concept_type="Disclosure",
        definition="Details on water consumed",
        unit="kl",
        datatype="float",
        hierarchy_path=["Principle 6"]
    )
    gri_concept = OntologyConcept(
        uri="http://example.org/ontology/rso#Topic_303-1",
        framework="GRI",
        label="GRI 303 Water and Effluents",
        concept_type="Topic",
        definition="Water withdrawal and consumption details",
        unit="kl",
        datatype="float",
        hierarchy_path=["GRI 303"]
    )
    
    cand = MappingCandidate(
        brsr_concept=brsr_concept,
        gri_concept=gri_concept,
        evidence=MappingEvidence(
            label_similarity=0.8,
            hierarchy_similarity=0.6,
            datatype_compatibility=1.0,
            unit_compatibility=1.0
        ),
        similarity_score=0.8
    )
    
    repaired = reasoner.check_consistency([cand])
    assert len(repaired) == 1
    # Check that m.reasoning list contains descriptive rule strings
    assert len(repaired[0].reasoning) > 0
    # Both are under environmental, so there shouldn't be a disjointness violation penalty, but taxonomic check etc
    has_rule_1 = any("Rule 1" in r for r in repaired[0].reasoning)
    assert has_rule_1

def test_llm_verifier_rich_batch():
    verifier = LLMVerifier()
    # Mocking self.enabled to True and self.api_key to a fake value to test formatting if needed,
    # but here we can just test if the formatting methods run without error.
    fake_payload = [
        {
            "brsr_id": "Disclosure_Q1",
            "brsr_label": "Water consumed",
            "brsr_definition": "Details on water consumed",
            "brsr_hierarchy": ["Principle 6"],
            "gri_id": "Topic_303-1",
            "gri_label": "Water withdrawal",
            "gri_definition": "Water withdrawal and consumption details",
            "gri_hierarchy": ["GRI 303"],
            "lexical_score": 0.8,
            "structural_score": 0.6,
            "property_score": 1.0,
            "reasoning_score": 0.8,
            "overall_confidence": 80.0,
            "skos_relation": "closeMatch",
            "evidence_summary": {},
            "reasoning": ["Rule 1: Both Environmental"]
        }
    ]
    
    # If API key is not present, it will return the default verification response, which we can assert
    results = verifier.verify_mappings_rich_batch(fake_payload)
    assert len(results) == 1
    assert "verification" in results[0]
    assert "confidence" in results[0]
    assert "explanation" in results[0]
