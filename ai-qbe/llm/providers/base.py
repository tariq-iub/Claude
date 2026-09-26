"""Model-independent LLM provider interface (ILLMProvider).

Every backend (Ollama, raw llama.cpp server, an OpenAI-compatible endpoint)
implements this same interface so the rest of AI-QBE — the generation
planner, the benchmark harness, the validators — never depends on a
specific model or runtime. See docs/PHASE0-DESIGN.md section 1 and 3.
"""

from __future__ import annotations

import abc
import dataclasses
import time
from typing import Any, Optional


@dataclasses.dataclass
class GenerationResult:
    """Uniform result shape returned by every provider implementation."""

    text: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    latency_seconds: float
    raw_response: Any = None

    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.completion_tokens is None or self.latency_seconds <= 0:
            return None
        return self.completion_tokens / self.latency_seconds


@dataclasses.dataclass
class ProviderMetadata:
    """Static facts about the model a provider is currently serving.

    Populated from provider config, not measured — VRAM/RAM usage is
    measured separately by the benchmark harness's resource monitor,
    since providers themselves usually cannot report GPU memory reliably.
    """

    provider_type: str
    model_name: str
    model_version: str
    quantization: Optional[str]
    context_window: int


class ILLMProvider(abc.ABC):
    """Interface every local/remote LLM backend must implement.

    Concrete providers: OllamaProvider, LlamaCppProvider,
    OpenAICompatibleProvider (see sibling modules in this package).
    """

    @abc.abstractmethod
    def metadata(self) -> ProviderMetadata:
        ...

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Return True if the backend is reachable and ready to serve."""

    @abc.abstractmethod
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
        """Free-form text generation. Used for tasks that don't require
        structured JSON output (e.g. free-text explanation quality checks).
        """

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
        """Generate output intended to validate against `json_schema`.

        Default implementation appends a strict-JSON instruction and
        relies on the caller to validate/retry (see
        backend/generation's schema-validation-with-bounded-retries
        policy, Phase 5). Providers that support native grammar-
        constrained decoding (e.g. a llama.cpp GBNF grammar derived from
        the schema) should override this method to use it instead —
        that override point is exactly why this is a provider-level
        method and not a free function.
        """
        instruction = (
            "You must respond with a single JSON object only — no prose, "
            "no markdown code fences, no explanation outside the JSON. "
            "The JSON MUST validate against this JSON Schema:\n"
            f"{json_schema}\n\nPrompt:\n{prompt}"
        )
        return self.generate_text(
            instruction,
            system=system,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            max_tokens=max_tokens,
            seed=seed,
        )


def timed(fn):
    """Decorator that measures wall-clock latency around a provider call
    and stamps it onto the returned GenerationResult."""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        result.latency_seconds = time.perf_counter() - start
        return result

    return wrapper
