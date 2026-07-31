from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class OntologyConcept(BaseModel):
    uri: str
    framework: str # "BRSR" or "GRI"
    label: str
    concept_type: str # Disclosure, Requirement, Topic
    definition: str = ""
    metric: Optional[str] = None
    unit: Optional[str] = None
    datatype: Optional[str] = None
    applicability: Optional[str] = None
    topic: Optional[str] = None
    parent_topic: Optional[str] = None
    hierarchy_path: List[str] = Field(default_factory=list)
    relationships: List[str] = Field(default_factory=list)

class MappingEvidence(BaseModel):
    label_similarity: float = 0.0
    definition_similarity: float = 0.0
    embedding_similarity: float = 0.0
    unit_compatibility: float = 0.0
    datatype_compatibility: float = 0.0
    hierarchy_similarity: float = 0.0
    relationship_similarity: float = 0.0
    topic_similarity: float = 0.0
    context_similarity: float = 0.0

class MappingCandidate(BaseModel):
    brsr_concept: OntologyConcept
    gri_concept: OntologyConcept
    evidence: MappingEvidence
    similarity_score: float = 0.0
    reasoning: List[str] = Field(default_factory=list)

class FinalMapping(BaseModel):
    # Legacy fields (for backward compatibility)
    brsr_uri: str
    gri_uri: str
    brsr_label: str
    gri_label: str
    relationship: str # Equivalent, Partial Equivalent, Broader, Narrower, NotMapped
    confidence_score: float = 0.0
    similarity_score: float = 0.0
    evidence_summary: Dict[str, Any]
    llm_verification: str = "Uncertain"
    llm_explanation: str = ""
    ontology_path: str = ""
    
    # New requested fields
    brsr_id: str = ""
    gri_id: str = ""
    lexical_score: float = 0.0
    structural_score: float = 0.0
    property_score: float = 0.0
    reasoning_score: float = 0.0
    overall_confidence: float = 0.0
    skos_relation: str = ""
    reasoning: List[str] = Field(default_factory=list)
    verification: Dict[str, Any] = Field(default_factory=dict)
