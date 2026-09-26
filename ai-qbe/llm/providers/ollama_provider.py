"""ILLMProvider implementation backed by a local Ollama server.

Ollama is the MVP default runtime (see docs/PHASE0-DESIGN.md section 1):
it wraps llama.cpp, serves 4-bit GGUF models, and exposes GPU-layer
offload control via Modelfile `num_gpu` / the `OLLAMA_NUM_GPU` setting,
which is what makes 7-9B Q4 models feasible on an 8GB-VRAM workstation.
"""

from __future__ import annotations

import time
from typing import Optional

import requests

from .base import GenerationResult, ILLMProvider, ProviderMetadata


class OllamaProvider(ILLMProvider):
    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://localhost:11434",
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
            provider_type="ollama",
            model_name=self.model_name,
            model_version=self.model_version,
            quantization=self.quantization,
            context_window=self.context_window,
        )

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

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
        options = {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "num_ctx": self.context_window,
        }
        if top_k is not None:
            options["top_k"] = top_k
        if repeat_penalty is not None:
            options["repeat_penalty"] = repeat_penalty
        if seed is not None:
            options["seed"] = seed

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": options,
        }

        start = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/api/generate", json=payload, timeout=self.request_timeout
        )
        latency = time.perf_counter() - start
        resp.raise_for_status()
        body = resp.json()

        return GenerationResult(
            text=body.get("response", ""),
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
            latency_seconds=latency,
            raw_response=body,
        )

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
        # Ollama supports format="json" to bias decoding toward valid JSON.
        # This is weaker than true grammar-constrained decoding against the
        # specific schema (Ollama does not accept a JSON Schema directly),
        # so the schema is still spelled out in-prompt and the caller
        # (backend/validation's structural validator, Phase 7) must still
        # validate the result — never trust format="json" alone.
        options = {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "num_ctx": self.context_window,
        }
        if top_k is not None:
            options["top_k"] = top_k
        if repeat_penalty is not None:
            options["repeat_penalty"] = repeat_penalty
        if seed is not None:
            options["seed"] = seed

        instruction = (
            "Respond with a single JSON object only. It MUST validate "
            f"against this JSON Schema:\n{json_schema}\n\nPrompt:\n{prompt}"
        )
        payload = {
            "model": self.model_name,
            "prompt": instruction,
            "system": system or "",
            "stream": False,
            "format": "json",
            "options": options,
        }

        start = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/api/generate", json=payload, timeout=self.request_timeout
        )
        latency = time.perf_counter() - start
        resp.raise_for_status()
        body = resp.json()

        return GenerationResult(
            text=body.get("response", ""),
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
            latency_seconds=latency,
            raw_response=body,
        )
