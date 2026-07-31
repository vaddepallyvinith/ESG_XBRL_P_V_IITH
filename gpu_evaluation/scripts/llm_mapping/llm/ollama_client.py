"""
Unified LLM REST API Client.
Priority order:
  1. Groq API  (GROQ_API_KEY from .env) → meta/llama-3.1-8b-instant  [FREE, fast]
  2. Local Ollama (/api/generate)        → llama3.1:8b                [Fallback]

Provides structured generation, exact token metrics, response timing, and error handling.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional

import requests  # faster, bypasses Cloudflare blocks that affect urllib

from llm_mapping.config.settings import LLMConfig
from llm_mapping.utils.logging_config import setup_logger

logger = setup_logger("ollama_client")


# ---------------------------------------------------------------------------
# .env auto-loader
# ---------------------------------------------------------------------------

def _load_env_keys() -> None:
    """Auto-load API keys from workspace .env into os.environ."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent.parent / ".env",  # Nested_RAG/.env
        Path(__file__).resolve().parent.parent.parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            k, _, v = line.partition("=")
                            os.environ.setdefault(k.strip(), v.strip())
                logger.debug(f"Loaded .env from: {env_path}")
                break
            except Exception:
                pass


_load_env_keys()


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class OllamaResponse:
    """Structured container for LLM generation output and execution metrics."""
    text: str
    prompt_tokens: int = 0
    eval_tokens: int = 0
    total_tokens: int = 0
    response_time_sec: float = 0.0
    model_name: str = ""
    status: str = "success"
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "eval_tokens": self.eval_tokens,
            "total_tokens": self.total_tokens,
            "response_time_sec": round(self.response_time_sec, 3),
            "model_name": self.model_name,
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class OllamaClient:
    """LLM client: Groq (Llama 3.1 8B) → local Ollama fallback."""

    GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.host = self.config.host.rstrip("/")
        self.model_name = self.config.model_name          # local Ollama model
        self.timeout = self.config.request_timeout

        # Groq API key (GROQ_API_KEY from .env)
        self.groq_key = os.getenv("GROQ_API_KEY", "").strip()

        # Groq model for Llama 3.1 8B (fastest free Groq model)
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

        # Groq free tier: 6000 tokens/min limit → cap prompt chars (~4 chars/token)
        self._max_prompt_chars = 4000  # ~1000 tokens safety headroom

        if self.groq_key:
            logger.info(
                f"Groq API key loaded (****{self.groq_key[-6:]}). "
                f"Primary model: {self.groq_model}"
            )
        else:
            logger.warning(
                "GROQ_API_KEY not found in .env. Will use local Ollama only."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Returns True if Groq key is set or local Ollama is reachable."""
        if self.groq_key:
            return True
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> OllamaResponse:
        """
        Generates text. Uses Groq if API key is set, otherwise falls back to local Ollama.
        """
        if self.groq_key:
            return self._generate_groq(prompt, system_prompt)
        return self._generate_ollama(prompt, system_prompt)

    # ------------------------------------------------------------------
    # Groq backend (Llama 3.1 8B — fast, free)
    # ------------------------------------------------------------------

    def _generate_groq(self, prompt: str, system_prompt: Optional[str] = None) -> OllamaResponse:
        """Calls Groq REST API for Llama 3.1 8B generation with retry on rate-limit."""

        # Truncate prompt to stay under the free-tier 6000 TPM limit
        if len(prompt) > self._max_prompt_chars:
            prompt = prompt[:self._max_prompt_chars] + "\n\n[Prompt truncated for length]"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.groq_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": min(self.config.max_tokens, 512),  # cap at 512 for free tier
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.groq_key}",
        }

        max_retries = 3
        start_time = time.perf_counter()

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    self.GROQ_URL,
                    json=payload,
                    headers=headers,
                    timeout=30,
                )
                elapsed_sec = time.perf_counter() - start_time

                if resp.status_code == 200:
                    resp_body = resp.json()
                    text = (
                        resp_body.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                    usage = resp_body.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    eval_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens + eval_tokens)
                    actual_model = resp_body.get("model", self.groq_model)

                    return OllamaResponse(
                        text=text,
                        prompt_tokens=prompt_tokens,
                        eval_tokens=eval_tokens,
                        total_tokens=total_tokens,
                        response_time_sec=elapsed_sec,
                        model_name=actual_model,
                        status="success",
                        raw_payload=resp_body,
                    )

                elif resp.status_code == 429:  # Rate limit — wait and retry
                    wait_sec = 10 * (attempt + 1)
                    logger.warning(
                        f"Groq rate limit (429) on attempt {attempt+1}/{max_retries}. "
                        f"Waiting {wait_sec}s before retry..."
                    )
                    time.sleep(wait_sec)
                    continue

                elif resp.status_code == 413:  # Request too large — truncate further
                    self._max_prompt_chars = int(self._max_prompt_chars * 0.7)
                    prompt = prompt[:self._max_prompt_chars] + "\n\n[Prompt truncated for length]"
                    messages[-1]["content"] = prompt
                    payload["messages"] = messages
                    logger.warning(
                        f"Groq 413 (too large). Reduced prompt to {self._max_prompt_chars} chars. Retrying..."
                    )
                    continue

                else:
                    logger.warning(
                        f"Groq API HTTP {resp.status_code}: {resp.text[:200]}. "
                        "Falling back to local Ollama..."
                    )
                    break

            except requests.exceptions.Timeout:
                logger.warning(f"Groq API timed out (attempt {attempt+1}). Retrying...")
                time.sleep(5)
            except Exception as e:
                logger.warning(f"Groq API error: {e}. Falling back to local Ollama...")
                break

        logger.warning("All Groq retries exhausted. Falling back to local Ollama...")
        return self._generate_ollama(prompt, system_prompt)

    # ------------------------------------------------------------------
    # Local Ollama fallback
    # ------------------------------------------------------------------

    def _generate_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> OllamaResponse:
        """Fallback: local Ollama /api/generate REST API."""
        url = f"{self.host}/api/generate"

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        start_time = time.perf_counter()
        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            elapsed_sec = time.perf_counter() - start_time

            if resp.status_code == 200:
                resp_body = resp.json()
                text = resp_body.get("response", "").strip()
                prompt_tokens = resp_body.get("prompt_eval_count", 0)
                eval_tokens = resp_body.get("eval_count", 0)
                total_tokens = prompt_tokens + eval_tokens

                return OllamaResponse(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    eval_tokens=eval_tokens,
                    total_tokens=total_tokens,
                    response_time_sec=elapsed_sec,
                    model_name=self.model_name,
                    status="success",
                    raw_payload=resp_body,
                )
            else:
                return OllamaResponse(
                    text="",
                    response_time_sec=time.perf_counter() - start_time,
                    model_name=self.model_name,
                    status=f"http_error_{resp.status_code}",
                )

        except Exception as e:
            elapsed_sec = time.perf_counter() - start_time
            return OllamaResponse(
                text="",
                response_time_sec=elapsed_sec,
                model_name=self.model_name,
                status=f"error: {e}",
            )
