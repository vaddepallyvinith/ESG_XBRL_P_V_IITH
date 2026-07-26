"""
Unified LLM REST API Client supporting OpenRouter API (from .env) and Local Ollama instance.
Provides structured model generation, exact token metrics, response timing, and error handling.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional
import urllib.request
import urllib.error

from llm_mapping.config.settings import LLMConfig
from llm_mapping.utils.logging_config import setup_logger

logger = setup_logger("ollama_client")


def _load_env_keys():
    """Helper to auto-load API keys from workspace .env file."""
    env_paths = [
        Path(__file__).resolve().parent.parent.parent.parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            os.environ[k.strip()] = v.strip()
                break
            except Exception:
                pass


_load_env_keys()


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


class OllamaClient:
    """Client for performing generation requests against OpenRouter API or Ollama instance."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.host = self.config.host.rstrip("/")
        self.model_name = self.config.model_name
        self.timeout = self.config.request_timeout

        # Check for OpenRouter API key in environment / .env
        self.openrouter_key = (
            os.getenv("OWL_ALPHA_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or ""
        ).strip()

        # Target OpenRouter model: Llama 3.1 8B by default (configurable via env)
        self.openrouter_model = os.getenv(
            "OPENROUTER_MODEL",
            "meta-llama/llama-3.1-8b-instruct"   # fast, cheap, confirmed available
        )

    def is_available(self) -> bool:
        """Checks if OpenRouter API key is configured or local Ollama endpoint is reachable."""
        if self.openrouter_key:
            return True

        url = f"{self.host}/api/tags"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> OllamaResponse:
        """
        Sends completion request to OpenRouter API (if API key present) or local Ollama /api/generate.
        """
        if self.openrouter_key:
            return self._generate_openrouter(prompt, system_prompt)
        return self._generate_ollama(prompt, system_prompt)

    def _generate_openrouter(self, prompt: str, system_prompt: Optional[str] = None) -> OllamaResponse:
        """Generates text via OpenRouter REST API with automatic model failover."""
        url = "https://openrouter.ai/api/v1/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        candidate_models = [
            self.openrouter_model,                        # meta-llama/llama-3.1-8b-instruct
            "meta-llama/llama-3.3-70b-instruct",          # heavier fallback
            "openrouter/free",                            # last resort free-tier auto-route
        ]

        start_time = time.perf_counter()

        for model in candidate_models:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "HTTP-Referer": "https://github.com/vaddepallyvinith/ESG_XBRL_P_V_IITH",
                    "X-Title": "ESG Baseline LLM Mapping",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=8) as response:
                    elapsed_sec = time.perf_counter() - start_time
                    if response.status == 200:
                        resp_body = json.loads(response.read().decode("utf-8"))
                        text = resp_body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

                        usage = resp_body.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        eval_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", prompt_tokens + eval_tokens)

                        actual_model = resp_body.get("model", model)

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
            except Exception as e:
                logger.debug(f"OpenRouter model '{model}' failed or timed out ({e}). Retrying with next model...")

        logger.warning("All OpenRouter models failed. Falling back to local Ollama...")
        return self._generate_ollama(prompt, system_prompt)

    def _generate_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> OllamaResponse:
        """Fallback to local Ollama /api/generate REST API."""
        url = f"{self.host}/api/generate"

        payload = {
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

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start_time = time.perf_counter()

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                elapsed_sec = time.perf_counter() - start_time
                if response.status == 200:
                    resp_body = json.loads(response.read().decode("utf-8"))
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
                        response_time_sec=elapsed_sec,
                        model_name=self.model_name,
                        status=f"http_error_{response.status}",
                    )
        except Exception as e:
            elapsed_sec = time.perf_counter() - start_time
            return OllamaResponse(
                text="",
                response_time_sec=elapsed_sec,
                model_name=self.model_name,
                status=f"error: {e}",
            )
