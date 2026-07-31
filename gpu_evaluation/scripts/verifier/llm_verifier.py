import os
import json
import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Tuple, Dict, Any
from matcher.models import MappingCandidate

logger = logging.getLogger(__name__)

class LLMVerifier:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.model_name = model_name
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found. LLM Verification will be skipped.")
            self.enabled = False
            return
            
        self.enabled = True
        
        # Configure robust retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def verify_mappings_rich_batch(self, mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify mappings in batch using the LLM.
        Input items are dictionaries containing:
        - candidate mapping (BRSR & GRI detail)
        - ontology evidence
        - lexical score, structural score, property score, reasoning score
        - confidence
        - SKOS relation
        
        Returns a list of verification result dictionaries:
        {
          "verification": "Agree/Disagree",
          "confidence": "High/Medium/Low",
          "explanation": "..."
        }
        """
        if not self.enabled or not mappings:
            return [
                {
                    "verification": "Disagree",
                    "confidence": "Low",
                    "explanation": "LLM Verification skipped or failed"
                }
                for _ in range(len(mappings))
            ]
            
        chunk_size = 15
        results = []
        for i in range(0, len(mappings), chunk_size):
            chunk = mappings[i:i + chunk_size]
            logger.info(f"Processing LLM verification chunk {i//chunk_size + 1}/{(len(mappings)-1)//chunk_size + 1} ({len(chunk)} items)...")
            chunk_res = self._verify_mappings_rich_chunk(chunk)
            results.extend(chunk_res)
            
        return results

    def _verify_mappings_rich_chunk(self, mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        default_res = [
            {
                "verification": "Disagree",
                "confidence": "Low",
                "explanation": "LLM Verification skipped or failed"
            }
            for _ in range(len(mappings))
        ]
        
        if not self.enabled or not mappings:
            return default_res
            
        # Build batched prompt
        prompt_parts = []
        for i, m in enumerate(mappings):
            prompt_parts.append(f"""
            --- Candidate {i} ---
            BRSR Concept:
              ID: {m.get('brsr_id', 'Unknown')}
              Label: {m.get('brsr_label', 'Unknown')}
              Definition: {m.get('brsr_definition', 'Unknown')}
              Hierarchy: {" > ".join(m.get('brsr_hierarchy', []))}
            GRI Concept:
              ID: {m.get('gri_id', 'Unknown')}
              Label: {m.get('gri_label', 'Unknown')}
              Definition: {m.get('gri_definition', 'Unknown')}
              Hierarchy: {" > ".join(m.get('gri_hierarchy', []))}
            SKOS Relation: {m.get('skos_relation', 'Unknown')}
            
            Ontology Evidence:
              Lexical Score: {m.get('lexical_score', 0.0):.4f}
              Structural Score: {m.get('structural_score', 0.0):.4f}
              Property Score: {m.get('property_score', 0.0):.4f}
              Reasoning Score: {m.get('reasoning_score', 0.0):.4f}
              Overall Confidence: {m.get('overall_confidence', 0.0):.2f}%
              Evidence summary: {json.dumps(m.get('evidence_summary', {}))}
            """)
            
        joined_candidates = "\n".join(prompt_parts)
        
        prompt = f"""
        You are an expert ESG Ontology engineer. Verify the following batches of semantic mappings.
        The match/relationship itself was decided by the deterministic ontology matcher. Your task is only to VERIFY if the match is correct ("Agree") or incorrect ("Disagree").
        
        Criteria:
        - Output "Agree" if the BRSR concept and GRI concept share a meaningful conceptual overlap within the ESG domain (even if one is broader/narrower or they are of slightly different granularities).
        - Output "Disagree" if they represent completely unrelated ESG topics (e.g. water vs. waste, carbon emissions vs. gender diversity).
        
        Here are the candidates:
        {joined_candidates}
        
        Reply strictly in valid JSON format. Your output MUST be a JSON array of objects, where each object corresponds to a candidate in the exact same order they were provided.
        Example output format:
        [
            {{
                "id": 0,
                "verification": "Agree",
                "confidence": "High",
                "explanation": "Both concepts specifically address greenhouse gas emissions scope 1 reporting."
            }},
            ...
        ]
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a precise JSON-only outputting API. Always return a JSON array."},
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            # Sleep slightly to avoid spamming the API
            time.sleep(1.0)
            base_url = "https://api.groq.com/openai/v1/chat/completions"
            logger.info(f"Sending LLM batch verify request to {base_url} using model {self.model_name}...")
            response = self.session.post(base_url, headers=headers, json=payload, timeout=20)
            
            logger.info(f"LLM API response status code: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Groq API Error: {response.text}")
                
            response.raise_for_status()
            
            data = response.json()
            raw_text = data['choices'][0]['message']['content'].strip()
            
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3].strip()
                
            start_idx = raw_text.find('[')
            end_idx = raw_text.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                raw_text = raw_text[start_idx:end_idx]
                
            parsed_data = json.loads(raw_text)
            
            results = []
            if isinstance(parsed_data, list) and len(parsed_data) == len(mappings):
                for item in parsed_data:
                    verification = item.get("verification", "Disagree")
                    if verification not in ["Agree", "Disagree"]:
                        verification = "Disagree"
                    confidence = item.get("confidence", "Medium")
                    if confidence not in ["High", "Medium", "Low"]:
                        confidence = "Medium"
                    explanation = item.get("explanation", "Parsed verification")
                    
                    results.append({
                        "verification": verification,
                        "confidence": confidence,
                        "explanation": explanation
                    })
                return results
            else:
                logger.warning(f"LLM returned malformed JSON or incorrect array length. Defaulting to Disagree.")
                return default_res
                
        except Exception as e:
            err_msg = str(e)
            logger.error(f"LLM Verification batch failed: {err_msg}")
            if "429" in err_msg or "quota" in err_msg.lower() or "402" in err_msg:
                logger.warning("Quota/Rate limit exceeded! Disabling LLM verification.")
                self.enabled = False
            return default_res
            
    def verify_mappings_batch(self, candidates: List[MappingCandidate]) -> List[Tuple[str, str]]:
        rich_candidates = []
        for cand in candidates:
            rich_candidates.append({
                "brsr_id": cand.brsr_concept.uri.split("#")[-1],
                "brsr_label": cand.brsr_concept.label,
                "brsr_definition": cand.brsr_concept.definition,
                "brsr_hierarchy": cand.brsr_concept.hierarchy_path,
                "gri_id": cand.gri_concept.uri.split("#")[-1],
                "gri_label": cand.gri_concept.label,
                "gri_definition": cand.gri_concept.definition,
                "gri_hierarchy": cand.gri_concept.hierarchy_path,
                "lexical_score": cand.evidence.label_similarity,
                "structural_score": cand.evidence.hierarchy_similarity,
                "property_score": cand.evidence.datatype_compatibility,
                "reasoning_score": cand.similarity_score,
                "overall_confidence": cand.similarity_score * 100,
                "skos_relation": "closeMatch",
                "evidence_summary": cand.evidence.model_dump()
            })
        res = self.verify_mappings_rich_batch(rich_candidates)
        return [(r["verification"], r["explanation"]) for r in res]
        
    def verify_mapping(self, candidate: MappingCandidate) -> tuple[str, str]:
        res = self.verify_mappings_batch([candidate])
        return res[0]
