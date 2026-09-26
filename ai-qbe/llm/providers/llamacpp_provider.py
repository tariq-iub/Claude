"""ILLMProvider implementation backed by a raw llama.cpp server
(`llama-server` / `llama.cpp/server`).

Kept alongside OllamaProvider for the cases Ollama doesn't cover well:
GBNF grammar-constrained decoding for hard schema compliance, and
fine-grained control over GPU layer offload (`--n-gpu-layers`) when tuning
exactly how much of a model fits in 8GB of VRAM. See
docs/PHASE0-DESIGN.md section 1.
"""

from __future__ import annotations

import time
from typing import Optional

import requests

from .base import GenerationResult, ILLMProvider, ProviderMetadata


def json_schema_to_gbnf(json_schema: dict) -> Optional[str]:
    """Best-effort JSON-Schema -> GBNF grammar conversion.

    This is intentionally conservative: it only handles the flat/nested
    "object of primitives and arrays" shapes the MCQ schemas actually use
    (see llm/schemas/mcq_schema.json). Anything it can't confidently
    convert returns None, and the caller falls back to prompt-only
    enforcement — a partial grammar that silently accepts invalid shapes
    would be worse than no grammar at all.
    """
    if json_schema.get("type") != "object":
        return None
    # A real implementation would walk `properties`/`required` recursively.
    # Left as a documented extension point for Phase 5 rather than
    # speculatively built out now — grammar quality should be validated
    # against the actual selected model's tokenizer during Phase 1.
    return None


class LlamaCppProvider(ILLMProvider):
    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://localhost:8080",
        model_version: str = "unknown",
        quantization: Optional[str] = None,
        context_window: int = 4096,
        request_timeout: float = 300.0,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.model_version = model_version
        self.quantization = quantization
        self.context_window = context_window
        self.request_timeout = request_timeout

    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_type="llamacpp",
            model_name=self.model_name,
            model_version=self.model_version,
            quantization=self.quantization,
            context_window=self.context_window,
        )

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def _complete(self, prompt: str, params: dict) -> GenerationResult:
        payload = {"prompt": prompt, **params}
        start = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/completion", json=payload, timeout=self.request_timeout
        )
        latency = time.perf_counter() - start
        resp.raise_for_status()
        body = resp.json()

        timings = body.get("timings", {})
        return GenerationResult(
            text=body.get("content", ""),
            prompt_tokens=timings.get("prompt_n"),
            completion_tokens=timings.get("predicted_n"),
            latency_seconds=latency,
            raw_response=body,
        )

    def generate_text(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        max_tokens: int = 1024,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        params = {
            "temperature": temperature,
            "top_p": top_p,
            "n_predict": max_tokens,
        }
        if top_k is not None:
            params["top_k"] = top_k
        if repeat_penalty is not None:
            params["repeat_penalty"] = repeat_penalty
        if seed is not None:
            params["seed"] = seed
        return self._complete(full_prompt, params)

    def generate_structured(
        self,
        prompt: str,
        *,
        json_schema: dict,
        system: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        max_tokens: int = 1024,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        full_prompt = (
            f"{system}\n\n" if system else ""
        ) + (
            "Respond with a single JSON object only. It MUST validate "
            f"against this JSON Schema:\n{json_schema}\n\nPrompt:\n{prompt}"
        )
        params = {
            "temperature": temperature,
            "top_p": top_p,
            "n_predict": max_tokens,
        }
        if top_k is not None:
            params["top_k"] = top_k
        if repeat_penalty is not None:
            params["repeat_penalty"] = repeat_penalty
        if seed is not None:
            params["seed"] = seed

        grammar = json_schema_to_gbnf(json_schema)
        if grammar:
            params["grammar"] = grammar

        return self._complete(full_prompt, params)
