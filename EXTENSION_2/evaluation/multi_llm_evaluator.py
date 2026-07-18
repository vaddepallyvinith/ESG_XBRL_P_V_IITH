import os
import json
import time
import logging
import random
import threading
import traceback
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import requests
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError, Timeout

logger = logging.getLogger(__name__)

# Custom Exceptions for clean error propagation
class RateLimitException(Exception):
    """Raised when HTTP 429 is received and all retries are exhausted."""
    pass

class InsufficientBalanceException(Exception):
    """Raised when HTTP 402 is received."""
    pass

class InvalidApiKeyException(Exception):
    """Raised when HTTP 401 is received."""
    pass

class PermissionDeniedException(Exception):
    """Raised when HTTP 403 is received."""
    pass

class ProviderOfflineException(Exception):
    """Raised when connection retries are exhausted."""
    pass


@dataclass
class ProviderConfig:
    name: str
    provider: str
    model: str
    env_var: str
    base_url: str
    cost_per_1k_tokens: float
    temperature: float = 0.0
    timeout: int = 30
    max_tokens: int = 2000
    batch_size: int = 10
    retries: int = 5
    delay: float = 1.0
    concurrency: int = 2
    api_key: str = ""


class EvaluationCache:
    """Thread-safe, file-backed cache for prompt evaluations."""
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.cache = {}
        self.load()

    def load(self):
        with self.lock:
            if os.path.exists(self.cache_path):
                try:
                    with open(self.cache_path, "r") as f:
                        self.cache = json.load(f)
                    logger.info(f"Loaded {len(self.cache)} entries from prompt cache.")
                except Exception as e:
                    logger.error(f"Failed to load cache: {e}")
                    self.cache = {}

    def get(self, model_id: str, prompt: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            key = f"{model_id}:{prompt}"
            return self.cache.get(key)

    def set(self, model_id: str, prompt: str, response_data: Dict[str, Any]):
        with self.lock:
            key = f"{model_id}:{prompt}"
            self.cache[key] = response_data
            try:
                temp_path = self.cache_path + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(self.cache, f, indent=4)
                os.replace(temp_path, self.cache_path)
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")


class MultiLLMEvaluator:
    def __init__(self, config: dict):
        self.models_config: List[ProviderConfig] = []
        raw_models = config.get("evaluation", {}).get("models", [])
        for m in raw_models:
            cfg = ProviderConfig(
                name=m["name"],
                provider=m["provider"],
                model=m["model"],
                env_var=m.get("env_var", ""),
                base_url=m.get("base_url", ""),
                cost_per_1k_tokens=float(m.get("cost_per_1k_tokens", 0.0)),
                temperature=float(m.get("temperature", 0.0)),
                timeout=int(m.get("timeout", 30)),
                max_tokens=int(m.get("max_tokens", 2000)),
                batch_size=int(m.get("batch_size", 10)),
                retries=int(m.get("retries", 5)),
                delay=float(m.get("delay", 1.0)),
                concurrency=int(m.get("concurrency", 2)),
                api_key=os.environ.get(m.get("env_var", ""), "")
            )
            self.models_config.append(cfg)
            
        # Cache setup
        cache_dir = "data/processed/mapping"
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = EvaluationCache(os.path.join(cache_dir, "evaluation_cache.json"))
        
        self.session = requests.Session()
        
        # In-memory status table
        self.provider_statuses: Dict[str, str] = {}
        self.provider_reasons: Dict[str, str] = {}
        self.provider_active_flags: Dict[str, bool] = {}
        
        # Execution statistics
        self.stats: Dict[str, Dict[str, Any]] = {}
        
        for m in self.models_config:
            self.stats[m.name] = {
                "api_calls": 0,
                "cached_calls": 0,
                "execution_time": 0.0,
                "cost_estimate": 0.0,
                "failed_requests": 0,
                "total_tokens": 0,
                "success_rate": 100.0
            }
            # Initial status
            if not m.api_key:
                self.provider_statuses[m.name] = "Skipped"
                self.provider_reasons[m.name] = f"Missing API Key in env: {m.env_var}"
            else:
                self.provider_statuses[m.name] = "Available"
                self.provider_reasons[m.name] = "API Key present"

    def evaluate_mappings(self, mappings: List[Dict[str, Any]], model_names: List[str] = None, checkpoint_path: str = None) -> Dict[str, Any]:
        """
        Evaluate mappings across multiple LLMs and calculate agreement and cost metrics.
        """
        if not self.models_config:
            logger.warning("No evaluation models configured in settings.yaml")
            return {}

        if not mappings:
            return {}

        # Load checkpoint for resume support
        checkpoint_data = {}
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r") as f:
                    checkpoint_data = json.load(f)
                logger.info(f"Loaded existing checkpoint from {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")

        # Scheduled order: OpenAI, Claude, Gemini, Groq, Cerebras, GitHub Models, Mistral, DeepSeek, OpenRouter
        SCHEDULING_ORDER = ["OpenAI", "Claude", "Gemini", "Groq", "Cerebras", "GitHub Models", "Mistral", "DeepSeek", "OpenRouter"]
        ordered_models = []
        for name in SCHEDULING_ORDER:
            for m in self.models_config:
                if m.name == name:
                    ordered_models.append(m)
        # Fallback for extra models not in list
        for m in self.models_config:
            if m not in ordered_models:
                ordered_models.append(m)

        for config in ordered_models:
            if model_names is not None and config.name not in model_names:
                continue

            if self.provider_statuses[config.name] == "Skipped":
                logger.info(f"INFO Provider Skipped: {config.name} ({self.provider_reasons[config.name]})")
                continue

            # Determine completed status
            existing_model_res = checkpoint_data.get(config.name, {})
            existing_decisions = existing_model_res.get("decisions", [])
            
            if len(existing_decisions) == len(mappings) and all(d is not None for d in existing_decisions):
                self.provider_statuses[config.name] = "Completed"
                self.provider_reasons[config.name] = "Loaded from checkpoint"
                logger.info(f"INFO Provider Completed (from checkpoint): {config.name}")
                continue

            logger.info(f"INFO Provider Started: {config.name}")
            self.provider_active_flags[config.name] = True
            
            start_time = time.time()
            try:
                decisions = self._evaluate_provider_mappings(config, mappings, existing_decisions, checkpoint_path, checkpoint_data)
                
                if not self.provider_active_flags[config.name]:
                    raise Exception(self.provider_reasons[config.name])
                
                self.provider_statuses[config.name] = "Completed"
                self.provider_reasons[config.name] = "Successfully evaluated all mappings"
                logger.info(f"INFO Provider Completed: {config.name}")
                
            except RateLimitException as e:
                self.provider_statuses[config.name] = "Rate Limited"
                self.provider_reasons[config.name] = str(e)
                logger.warning(f"WARNING Rate Limited: {config.name} - {e}")
            except InsufficientBalanceException as e:
                self.provider_statuses[config.name] = "No Credits"
                self.provider_reasons[config.name] = str(e)
                logger.warning(f"WARNING Insufficient Balance: {config.name} - {e}")
            except InvalidApiKeyException as e:
                self.provider_statuses[config.name] = "Invalid API Key"
                self.provider_reasons[config.name] = str(e)
                logger.warning(f"WARNING Invalid API Key: {config.name} - {e}")
            except PermissionDeniedException as e:
                self.provider_statuses[config.name] = "Permission Denied"
                self.provider_reasons[config.name] = str(e)
                logger.warning(f"WARNING Permission Denied: {config.name} - {e}")
            except ProviderOfflineException as e:
                self.provider_statuses[config.name] = "Offline"
                self.provider_reasons[config.name] = str(e)
                logger.warning(f"WARNING Offline: {config.name} - {e}")
            except Exception as e:
                self.provider_statuses[config.name] = "Offline"
                self.provider_reasons[config.name] = f"Unexpected error: {e}"
                logger.error(f"Unexpected error for {config.name}: {e}", exc_info=True)
                
            duration = time.time() - start_time
            self.stats[config.name]["execution_time"] = duration
            
            # Calculate cost estimate
            cost_per_1k = config.cost_per_1k_tokens
            total_tokens = self.stats[config.name]["total_tokens"]
            self.stats[config.name]["cost_estimate"] = (total_tokens / 1000.0) * cost_per_1k
            
            # Update success rate
            api_calls = self.stats[config.name]["api_calls"]
            failed_calls = self.stats[config.name]["failed_requests"]
            if api_calls > 0:
                self.stats[config.name]["success_rate"] = ((api_calls - failed_calls) / api_calls) * 100.0
            else:
                self.stats[config.name]["success_rate"] = 100.0

        # Save final report to JSON and CSV
        self._generate_final_reports()
        
        logger.info("INFO Pipeline Completed")
        
        # Load and return final multi_llm_results.json
        final_results = {}
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r") as f:
                    final_results = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load final results: {e}")
        return final_results

    def _evaluate_provider_mappings(self, config: ProviderConfig, mappings: List[Dict[str, Any]], existing_decisions: List[str], checkpoint_path: str, checkpoint_data: Dict[str, Any]) -> List[str]:
        N = len(mappings)
        decisions = [None] * N
        
        # Copy existing decisions from checkpoint if they exist
        for idx in range(min(len(existing_decisions), N)):
            decisions[idx] = existing_decisions[idx]
            
        # Determine missing batch indices
        batches = []
        batch_size = config.batch_size
        for idx in range(0, N, batch_size):
            batch_completed = True
            for b_idx in range(idx, min(idx + batch_size, N)):
                if decisions[b_idx] is None:
                    batch_completed = False
                    break
            if not batch_completed:
                batches.append((idx, mappings[idx : idx + batch_size]))

        if not batches:
            return decisions

        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        checkpoint_lock = threading.Lock()
        stats_lock = threading.Lock()
        
        def process_batch(start_idx: int, batch_items: List[Dict[str, Any]]):
            if not self.provider_active_flags.get(config.name, True):
                return start_idx, ["Uncertain"] * len(batch_items), 0, 0.0
                
            prompt = self._build_batch_prompt(batch_items)
            
            # Check Cache
            cached_res = self.cache.get(config.model, prompt)
            if cached_res:
                logger.info(f"INFO Cached Response Used for {config.name} at batch index {start_idx}")
                with stats_lock:
                    self.stats[config.name]["cached_calls"] += 1
                return start_idx, cached_res.get("decisions", ["Uncertain"] * len(batch_items)), cached_res.get("tokens", 0), 0.0
                
            # Perform API call with retries
            logger.info(f"INFO Sending Batch {start_idx // config.batch_size + 1}/{(N + config.batch_size - 1) // config.batch_size} for {config.name}")
            
            payload = {}
            if config.provider == "anthropic":
                payload = {
                    "model": config.model,
                    "system": "You are a precise JSON API. Return a JSON array only. Do not provide explanations.",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature
                }
            else:
                payload = {
                    "model": config.model,
                    "messages": [
                        {"role": "system", "content": "You are a precise JSON API. Return a JSON array only. Do not provide explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens
                }

            start_t = time.time()
            try:
                response_json = self._execute_request_with_retry(config, payload)
                duration = time.time() - start_t
                
                tokens = response_json.get("usage", {}).get("total_tokens", 0)
                raw_text = ""
                
                if config.provider == "anthropic":
                    content_list = response_json.get("content", [])
                    if content_list and "text" in content_list[0]:
                        raw_text = content_list[0]["text"].strip()
                else:
                    choices = response_json.get("choices", [])
                    if choices:
                        raw_text = choices[0].get("message", {}).get("content", "").strip()
                
                # Parse decisions
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:-3].strip()
                elif raw_text.startswith("```"):
                    raw_text = raw_text[3:-3].strip()
                    
                start_json = raw_text.find('[')
                end_json = raw_text.rfind(']') + 1
                if start_json != -1 and end_json != 0:
                    raw_text = raw_text[start_json:end_json]
                    
                batch_decisions = []
                try:
                    parsed_data = json.loads(raw_text)
                    if isinstance(parsed_data, list) and len(parsed_data) == len(batch_items):
                        for item in parsed_data:
                            dec = item.get("decision", "Uncertain")
                            if dec not in ["Agree", "Disagree", "Uncertain"]:
                                dec = "Uncertain"
                            batch_decisions.append(dec)
                    else:
                        logger.warning(f"Response array length mismatch or invalid format for {config.name}")
                        batch_decisions = ["Uncertain"] * len(batch_items)
                except Exception as parse_err:
                    logger.error(f"Failed to parse LLM JSON response from {config.name}: {parse_err}")
                    batch_decisions = ["Uncertain"] * len(batch_items)
                
                # Cache response
                self.cache.set(config.model, prompt, {
                    "decisions": batch_decisions,
                    "tokens": tokens
                })
                
                return start_idx, batch_decisions, tokens, duration
                
            except (RateLimitException, InsufficientBalanceException, InvalidApiKeyException, PermissionDeniedException, ProviderOfflineException) as e:
                self.provider_active_flags[config.name] = False
                self.provider_reasons[config.name] = str(e)
                raise e
            except Exception as e:
                logger.error(f"Non-fatal error in batch {start_idx} for {config.name}: {e}", exc_info=True)
                return start_idx, ["Uncertain"] * len(batch_items), 0, 0.0

        # Run concurrent batches
        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            futures = [executor.submit(process_batch, start_idx, batch_items) for start_idx, batch_items in batches]
            
            for fut in as_completed(futures):
                try:
                    start_idx, batch_decisions, tokens, duration = fut.result()
                    
                    # Fill pre-allocated decisions
                    for offset, dec in enumerate(batch_decisions):
                        decisions[start_idx + offset] = dec
                        
                    with stats_lock:
                        self.stats[config.name]["total_tokens"] += tokens
                    
                    # Checkpoint immediately
                    if checkpoint_path:
                        with checkpoint_lock:
                            try:
                                existing = {}
                                if os.path.exists(checkpoint_path):
                                    with open(checkpoint_path, "r") as f:
                                        existing = json.load(f)
                                
                                # Cost calculation
                                avg_tokens = self.stats[config.name]["total_tokens"] / N if N else 0
                                cost_per_100 = (avg_tokens * 100 / 1000.0) * config.cost_per_1k_tokens
                                
                                existing[config.name] = {
                                    "decisions": decisions,
                                    "avg_response_time": self.stats[config.name]["execution_time"] / N if N else 0.0,
                                    "cost_per_100_mappings": cost_per_100,
                                    "total_tokens": self.stats[config.name]["total_tokens"]
                                }
                                
                                temp_path = checkpoint_path + ".tmp"
                                with open(temp_path, "w") as f:
                                    json.dump(existing, f, indent=4)
                                os.replace(temp_path, checkpoint_path)
                                logger.info(f"INFO Checkpoint Saved for model '{config.name}' to {checkpoint_path}")
                            except Exception as checkpoint_err:
                                logger.error(f"Failed to save checkpoint for {config.name}: {checkpoint_err}")
                                
                except Exception as e:
                    logger.error(f"Fatal exception occurred during concurrent evaluation of {config.name}: {e}")
                    raise e
                    
        return decisions

    def _execute_request_with_retry(self, config: ProviderConfig, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        if config.provider == "anthropic":
            headers = {
                "x-api-key": config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            
        use_gemini_query = (config.provider == "gemini" and "openai" not in config.base_url)
        
        current_timeout = config.timeout
        read_timeout_retries = 0
        connection_retries = 0
        rate_limit_retries = 0
        max_rate_limit_retries = config.retries
        
        # Increment API calls count
        with threading.Lock():
            self.stats[config.name]["api_calls"] += 1
            
        while True:
            if not self.provider_active_flags.get(config.name, True):
                raise Exception(f"Provider {config.name} deactivated")
                
            # Inter-request delay
            if config.delay > 0:
                time.sleep(config.delay)
                
            url = config.base_url
            req_headers = headers.copy()
            
            if use_gemini_query:
                url = f"{config.base_url}?key={config.api_key}"
                req_headers.pop("Authorization", None)
                
            try:
                response = self.session.post(
                    url,
                    headers=req_headers,
                    json=payload,
                    timeout=current_timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                    
                elif response.status_code == 429:
                    if rate_limit_retries >= max_rate_limit_retries:
                        logger.warning(f"WARNING Rate Limited: {config.name}")
                        with threading.Lock():
                            self.stats[config.name]["failed_requests"] += 1
                        raise RateLimitException(f"HTTP 429 Rate Limit exceeded after {max_rate_limit_retries} retries for model {config.name}")
                    
                    retry_after = response.headers.get("Retry-After")
                    sleep_delay = None
                    if retry_after:
                        try:
                            sleep_delay = float(retry_after)
                            logger.info(f"INFO Respecting Retry-After header: sleeping {sleep_delay:.2f}s")
                        except ValueError:
                            pass
                            
                    if sleep_delay is not None and sleep_delay > 30.0:
                        logger.warning(f"WARNING Retry-After {sleep_delay:.2f}s is too long (limit 30s). Deactivating provider {config.name}.")
                        with threading.Lock():
                            self.stats[config.name]["failed_requests"] += 1
                        raise RateLimitException(f"HTTP 429 Rate Limit Retry-After {sleep_delay:.2f}s exceeds limit of 30s")
                            
                    if sleep_delay is None:
                        # Exponential backoff with random jitter (0-2s)
                        sleep_delay = (5.0 * (2 ** rate_limit_retries)) + random.uniform(0.0, 2.0)
                        
                    logger.info(f"INFO Retry {rate_limit_retries + 1}/{max_rate_limit_retries} for {config.name} in {sleep_delay:.2f}s due to HTTP 429")
                    time.sleep(sleep_delay)
                    rate_limit_retries += 1
                    continue
                    
                elif response.status_code == 402:
                    logger.warning(f"WARNING Insufficient Balance for {config.name}")
                    with threading.Lock():
                        self.stats[config.name]["failed_requests"] += 1
                    raise InsufficientBalanceException("Insufficient Balance (HTTP 402)")
                    
                elif response.status_code == 401:
                    logger.warning(f"WARNING Invalid API Key for {config.name}")
                    with threading.Lock():
                        self.stats[config.name]["failed_requests"] += 1
                    raise InvalidApiKeyException("Invalid API Key (HTTP 401)")
                    
                elif response.status_code == 403:
                    logger.warning(f"WARNING Permission Denied for {config.name}")
                    with threading.Lock():
                        self.stats[config.name]["failed_requests"] += 1
                    raise PermissionDeniedException("Permission Denied (HTTP 403)")
                    
                elif response.status_code in [500, 502, 503, 504]:
                    if rate_limit_retries >= max_rate_limit_retries:
                        logger.error(f"HTTP Error {response.status_code} for {config.name} after {max_rate_limit_retries} retries: {response.text}")
                        with threading.Lock():
                            self.stats[config.name]["failed_requests"] += 1
                        response.raise_for_status()
                    
                    sleep_delay = (3.0 * (2 ** rate_limit_retries)) + random.uniform(0.0, 2.0)
                    logger.info(f"INFO Retry {rate_limit_retries + 1}/{max_rate_limit_retries} for {config.name} in {sleep_delay:.2f}s due to HTTP {response.status_code}")
                    time.sleep(sleep_delay)
                    rate_limit_retries += 1
                    continue
                    
                else:
                    logger.error(f"HTTP Error {response.status_code} for {config.name}: {response.text}")
                    with threading.Lock():
                        self.stats[config.name]["failed_requests"] += 1
                    response.raise_for_status()
                    
            except (ReadTimeout, Timeout) as e:
                if read_timeout_retries >= 2:
                    logger.error(f"ReadTimeout exceeded for {config.name}: {e}")
                    with threading.Lock():
                        self.stats[config.name]["failed_requests"] += 1
                    raise ProviderOfflineException(f"ReadTimeout: {e}")
                read_timeout_retries += 1
                current_timeout += 15
                logger.info(f"INFO Retry {read_timeout_retries}/2 for {config.name} due to ReadTimeout")
                continue
                
            except ConnectionError as e:
                if connection_retries >= 3:
                    logger.error(f"ConnectionError exceeded for {config.name}: {e}")
                    with threading.Lock():
                        self.stats[config.name]["failed_requests"] += 1
                    raise ProviderOfflineException(f"ConnectionError: {e}")
                connection_retries += 1
                logger.info(f"INFO Retry {connection_retries}/3 for {config.name} due to ConnectionError")
                time.sleep(2.0)
                continue
                
            except Exception as e:
                logger.error(f"Exception raised in API request: {e}")
                with threading.Lock():
                    self.stats[config.name]["failed_requests"] += 1
                raise e

    def _build_batch_prompt(self, batch: List[Dict[str, Any]]) -> str:
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
        return prompt

    def _generate_final_reports(self):
        report_data = {
            "Evaluation Summary": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "Provider Status": []
        }
        
        csv_rows = []
        csv_rows.append([
            "Provider", "Status", "Reason", "API Calls", "Cached Calls", 
            "Execution Time (s)", "Cost Estimate ($)", "Failed Requests", "Success Rate (%)"
        ])
        
        for config in self.models_config:
            name = config.name
            status = self.provider_statuses.get(name, "Skipped")
            reason = self.provider_reasons.get(name, "Skipped")
            
            s = self.stats.get(name, {
                "api_calls": 0,
                "cached_calls": 0,
                "execution_time": 0.0,
                "cost_estimate": 0.0,
                "failed_requests": 0,
                "success_rate": 100.0
            })
            success_rate = s["success_rate"]
            
            status_entry = {
                "provider": name,
                "status": status,
                "reason": reason,
                "api_calls": s["api_calls"],
                "cached_calls": s["cached_calls"],
                "execution_time_sec": round(s["execution_time"], 2),
                "cost_estimate_usd": round(s["cost_estimate"], 5),
                "failed_requests": s["failed_requests"],
                "success_rate_percent": round(success_rate, 2)
            }
            report_data["Provider Status"].append(status_entry)
            
            csv_rows.append([
                name, status, reason, s["api_calls"], s["cached_calls"],
                round(s["execution_time"], 2), round(s["cost_estimate"], 5),
                s["failed_requests"], f"{success_rate:.2f}%"
            ])
            
        report_dir = "data/processed/mapping"
        os.makedirs(report_dir, exist_ok=True)
        
        # Save JSON Report
        json_path = os.path.join(report_dir, "evaluation_report.json")
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"INFO Report saved as JSON to {json_path}")
        
        # Save CSV Report
        csv_path = os.path.join(report_dir, "evaluation_report.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        logger.info(f"INFO Report saved as CSV to {csv_path}")
