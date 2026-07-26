"""
Prompt templates for baseline LLM-based disclosure mapping and explanation.
Completely independent of ontology, RDF, OWL, SKOS, or graph reasoning.
"""

BRSR_TO_GRI_SYSTEM_PROMPT = """You are an ESG disclosure mapping expert.
Map the given BRSR requirement to the single best matching GRI disclosure from the candidates (or "None" if no match).

Categories: Exact Match, Close Match, Broad Match, Narrow Match, No Match.

Respond ONLY with valid JSON matching this schema:
{
  "brsr_id": "<BRSR ID>",
  "gri_id": "<Best GRI Disclosure ID or 'None'>",
  "mapping_type": "<Exact Match | Close Match | Broad Match | Narrow Match | No Match>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<brief analysis>",
  "explanation": "<summary justification>"
}"""

BRSR_TO_GRI_MAPPING_PROMPT_TEMPLATE = """BRSR Requirement:
ID: {brsr_id}
Label: {brsr_label}
Principle: {brsr_principle}
Text: {brsr_text}

GRI Candidates:
{retrieved_candidates_text}

Select the single best matching GRI disclosure ID or "None". Return strictly JSON."""
