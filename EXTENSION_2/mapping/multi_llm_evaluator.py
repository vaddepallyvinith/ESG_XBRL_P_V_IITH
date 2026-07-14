import os
import json
import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

PROVIDER_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "cerebras": "https://api.cerebras.ai/v1/chat/completions"
}

class MultiLLMEvaluator:
    def __init__(self, config: dict):
        self.models_config = config.get("evaluation", {}).get("models", [])
        
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
        


    def evaluate_mappings(self, mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate mappings across multiple LLMs and calculate agreement and cost metrics.
        """
        if not self.models_config:
            logger.warning("No evaluation models configured in settings.yaml")
            return {}

        if not mappings:
            return {}

        results = {}
        
        for model_cfg in self.models_config:
            name = model_cfg["name"]
            provider = model_cfg["provider"]
            model_id = model_cfg["model"]
            env_var = model_cfg.get("env_var", "")
            cost_per_1k = model_cfg.get("cost_per_1k_tokens", 0.0)
            
            api_key = os.environ.get(env_var)
            if not api_key:
                logger.warning(f"Skipping {name} due to missing API key in env: {env_var}")
                continue
                
            base_url = PROVIDER_BASE_URLS.get(provider)
            if not base_url:
                logger.warning(f"Unsupported provider {provider} for {name}")
                continue
                
            logger.info(f"Running evaluation with {name} ({model_id})...")
            
            model_results = self._run_model_verification(
                name, provider, model_id, api_key, base_url, mappings
            )
            
            # Calculate metrics
            total_tokens = model_results.get("total_tokens", 0)
            total_time = model_results.get("total_time", 0.0)
            decisions = model_results.get("decisions", [])
            
            # Cost per 100 mappings
            # If total_mappings took total_tokens, then 100 mappings takes:
            avg_tokens_per_mapping = total_tokens / len(mappings) if mappings else 0
            cost_per_100 = (avg_tokens_per_mapping * 100 / 1000.0) * cost_per_1k
            
            results[name] = {
                "decisions": decisions,
                "avg_response_time": total_time / len(mappings) if mappings else 0.0,
                "cost_per_100_mappings": cost_per_100,
                "total_tokens": total_tokens
            }
            
        return results
        
    def _run_model_verification(self, name: str, provider: str, model_id: str, api_key: str, base_url: str, mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
        decisions = []
        total_tokens = 0
        total_time = 0.0
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Depending on provider, Gemini doesn't use Bearer token if we pass it as key params, 
        # but OpenAI-compatible proxy allows Bearer authorization. We'll stick to Bearer.
        # Actually for Gemini API key, usually it expects "?key=API_KEY" but with google's OpenAI compat, Bearer works.
        
        # Batch size for evaluation
        batch_size = 10
        
        for i in range(0, len(mappings), batch_size):
            batch = mappings[i:i+batch_size]
            
            prompt_parts = []
            for j, m in enumerate(batch):
                prompt_parts.append(f"""
                --- Candidate {j} ---
                BRSR: {m.get('brsr_label', 'Unknown')}
                GRI: {m.get('gri_label', 'Unknown')}
                Similarity Score: {m.get('similarity_score', 0):.2f}/1.00
                """)
                
            joined_candidates = "\n".join(prompt_parts)
            
            prompt = f"""
            You are an expert ESG Ontology engineer. Verify the following batches of semantic mappings.
            For each candidate, decide if the BRSR concept and GRI concept represent an equivalent, partial, or broader/narrower relationship.
            If they share a meaningful conceptual overlap, output "Agree". 
            Only output "Disagree" if they represent completely unrelated ESG topics.
            
            {joined_candidates}
            
            Reply strictly in valid JSON format. Your output MUST be a JSON array of objects with NO explanation.
            Example:
            [
                {{"id": 0, "decision": "Agree"}}
            ]
            """
            
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": "You are a precise JSON API. Return a JSON array only. Do not provide explanations."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0
            }
            
            start_t = time.time()
            try:
                time.sleep(1.0) # rate limiting
                response = self.session.post(base_url, headers=headers, json=payload, timeout=60)
                
                # Check for Gemini specific auth if 401
                if response.status_code == 401 and provider == "gemini":
                    # Gemini fallback: pass key in query string, remove Bearer
                    fallback_url = f"{base_url}?key={api_key}"
                    fallback_headers = {"Content-Type": "application/json"}
                    response = self.session.post(fallback_url, headers=fallback_headers, json=payload, timeout=60)
                
                if response.status_code != 200:
                    logger.error(f"{name} API Error: {response.text}")
                    decisions.extend(["Uncertain"] * len(batch))
                    continue
                    
                data = response.json()
                total_time += (time.time() - start_t)
                
                usage = data.get('usage', {})
                total_tokens += usage.get('total_tokens', 500) # Fallback to 500 if missing
                
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
                
                if isinstance(parsed_data, list) and len(parsed_data) == len(batch):
                    for item in parsed_data:
                        dec = item.get("decision", "Uncertain")
                        if dec not in ["Agree", "Disagree", "Uncertain"]:
                            dec = "Uncertain"
                        decisions.append(dec)
                else:
                    decisions.extend(["Uncertain"] * len(batch))
                    
            except Exception as e:
                logger.error(f"{name} evaluation failed: {e}")
                decisions.extend(["Uncertain"] * len(batch))
                
        return {
            "decisions": decisions,
            "total_tokens": total_tokens,
            "total_time": total_time
        }
