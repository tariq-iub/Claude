"""Builds an ILLMProvider from configuration. The one place that knows
about all concrete provider classes, so callers (generation executor,
Celery tasks, API startup) depend only on `ILLMProvider` and this factory,
never on a specific backend class.
"""

from __future__ import annotations

from .base import ILLMProvider
from .llamacpp_provider import LlamaCppProvider
from .mock_provider import MockProvider
from .ollama_provider import OllamaProvider
from .openai_compatible_provider import OpenAICompatibleProvider


def build_provider(
    provider_type: str,
    model_name: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model_version: str = "unknown",
    quantization: str | None = None,
    context_window: int = 4096,
) -> ILLMProvider:
    if provider_type == "mock":
        return MockProvider(model_name)
    if provider_type == "ollama":
        return OllamaProvider(
            model_name,
            base_url=base_url or "http://localhost:11434",
            model_version=model_version,
            quantization=quantization,
            context_window=context_window,
        )
    if provider_type == "llamacpp":
        return LlamaCppProvider(
            model_name,
            base_url=base_url or "http://localhost:8080",
            model_version=model_version,
            quantization=quantization,
            context_window=context_window,
        )
    if provider_type == "openai_compatible":
        if not base_url:
            raise ValueError("base_url is required for provider_type=openai_compatible")
        return OpenAICompatibleProvider(
            model_name,
            base_url=base_url,
            api_key=api_key,
            model_version=model_version,
            quantization=quantization,
            context_window=context_window,
        )
    raise ValueError(f"Unknown provider_type: {provider_type}")
