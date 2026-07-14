import os
import json
import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Tuple
from mapping.models import MappingCandidate

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

    def verify_mappings_batch(self, candidates: List[MappingCandidate]) -> List[Tuple[str, str]]:
        if not self.enabled or not candidates:
            return [("Uncertain", "LLM disabled or no candidates")] * len(candidates)
            
        # Build batched prompt
        prompt_parts = []
        for i, cand in enumerate(candidates):
            prompt_parts.append(f"""
            --- Candidate {i} ---
            BRSR: {cand.brsr_concept.label} | Context: {" > ".join(cand.brsr_concept.hierarchy_path)} | Def: {cand.brsr_concept.definition} | Metric: {cand.brsr_concept.metric}
            GRI: {cand.gri_concept.label} | Context: {" > ".join(cand.gri_concept.hierarchy_path)} | Def: {cand.gri_concept.definition} | Metric: {cand.gri_concept.metric}
            Similarity Score: {cand.similarity_score:.2f}/1.00
            """)
            
        joined_candidates = "\n".join(prompt_parts)
        
        prompt = f"""
        You are an expert ESG Ontology engineer. Verify the following batches of semantic mappings.
        For each candidate, decide if the BRSR concept and GRI concept represent an equivalent, partial, or broader/narrower relationship.
        If they share a meaningful conceptual overlap (even if one is broader than the other), you MUST output "Agree". 
        Only output "Disagree" if they represent completely unrelated ESG topics (e.g. water vs waste, or governance vs emissions).
        Do NOT propose new mappings.

        {joined_candidates}

        Reply strictly in valid JSON format. Your output MUST be a JSON array of objects, where each object corresponds to a candidate in the exact same order they were provided.
        Example format:
        [
            {{
                "id": 0,
                "decision": "Agree" | "Disagree" | "Uncertain",
                "explanation": "Brief 1-sentence reason why."
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
            response = self.session.post(base_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                logger.error(f"Groq API Error: {response.text}")
                
            response.raise_for_status()
            
            data = response.json()
            raw_text = data['choices'][0]['message']['content'].strip()
            
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3].strip()
                
            # Extra cleanup just in case there's leading/trailing garbage
            start_idx = raw_text.find('[')
            end_idx = raw_text.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                raw_text = raw_text[start_idx:end_idx]
                
            parsed_data = json.loads(raw_text)
            
            results = []
            if isinstance(parsed_data, list) and len(parsed_data) == len(candidates):
                for item in parsed_data:
                    decision = item.get("decision", "Uncertain")
                    explanation = item.get("explanation", "Failed to parse explanation")
                    if decision not in ["Agree", "Disagree", "Uncertain"]:
                        decision = "Uncertain"
                    results.append((decision, explanation))
                return results
            else:
                logger.warning(f"LLM returned malformed JSON or incorrect array length ({len(parsed_data) if isinstance(parsed_data, list) else 'not a list'} instead of {len(candidates)}). Defaulting to Uncertain.")
                return [("Uncertain", "Malformed LLM batched output")] * len(candidates)
            
        except Exception as e:
            err_msg = str(e)
            logger.error(f"LLM Verification batch failed: {err_msg}")
            
            if "429" in err_msg or "quota" in err_msg.lower() or "402" in err_msg:
                logger.warning("Quota/Rate limit exceeded! Disabling LLM verification.")
                self.enabled = False
                
            return [("Uncertain", f"Error calling LLM: {err_msg}")] * len(candidates)
            
    def verify_mapping(self, candidate: MappingCandidate) -> tuple[str, str]:
        res = self.verify_mappings_batch([candidate])
        return res[0]
