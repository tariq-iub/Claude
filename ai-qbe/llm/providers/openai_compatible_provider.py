"""ILLMProvider implementation for any OpenAI-compatible chat-completions
endpoint (vLLM, LM Studio, text-generation-webui, or a hosted API).

Exists so AI-QBE is never locked to Ollama/llama.cpp: migrating to a
larger GPU/server later (docs/PHASE0-DESIGN.md section 17, "scale-out")
is a config change to point at a vLLM endpoint, not a rewrite.
"""

from __future__ import annotations

import time
from typing import Optional

import requests

from .base import GenerationResult, ILLMProvider, ProviderMetadata


class OpenAICompatibleProvider(ILLMProvider):
    def __init__(
        self,
        model_name: str,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        model_version: str = "unknown",
        quantization: Optional[str] = None,
        context_window: int = 4096,
        request_timeout: float = 300.0,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_version = model_version
        self.quantization = quantization
        self.context_window = context_window
        self.request_timeout = request_timeout

    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_type="openai_compatible",
            model_name=self.model_name,
            model_version=self.model_version,
            quantization=self.quantization,
            context_window=self.context_window,
        )

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def health_check(self) -> bool:
        try:
            resp = requests.get(
                f"{self.base_url}/models", headers=self._headers(), timeout=5
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def _chat(self, messages: list, params: dict) -> GenerationResult:
        payload = {"model": self.model_name, "messages": messages, **params}
        start = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self.request_timeout,
        )
        latency = time.perf_counter() - start
        resp.raise_for_status()
        body = resp.json()

        choice = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        return GenerationResult(
            text=choice,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
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
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        params = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        if seed is not None:
            params["seed"] = seed
        # top_k / repeat_penalty are not part of the standard OpenAI API;
        # pass through as vendor extension fields where the backend supports it.
        if top_k is not None:
            params["top_k"] = top_k
        if repeat_penalty is not None:
            params["repetition_penalty"] = repeat_penalty

        return self._chat(messages, params)

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
        messages = []
        sys_text = (system + "\n\n") if system else ""
        sys_text += (
            "You must respond with a single JSON object only, matching the "
            "given JSON Schema exactly. No prose, no markdown fences."
        )
        messages.append({"role": "system", "content": sys_text})
        messages.append(
            {"role": "user", "content": f"JSON Schema:\n{json_schema}\n\nPrompt:\n{prompt}"}
        )

        params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if seed is not None:
            params["seed"] = seed

        return self._chat(messages, params)
