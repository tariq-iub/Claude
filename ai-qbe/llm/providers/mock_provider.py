"""A deterministic, template-based ILLMProvider implementation used as the
default in dev/test/CI environments that have no GPU or LLM runtime
available (this repository's own development sandbox included).

This is NOT a benchmarked model and must never be treated as one --
`docs/PHASE0-DESIGN.md` explicitly forbids claiming unmeasured
performance/quality numbers, and the same rule applies here: MockProvider
exists purely to exercise the generation/validation/API plumbing
end-to-end before Phase 1's real benchmark selects an actual model.
`GenerationModel.provider_type == "mock"` is rejected by a startup check
in production configuration (see backend/config.py) -- it is a
development convenience, not a deployment option.
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

from .base import GenerationResult, ILLMProvider, ProviderMetadata


class MockProvider(ILLMProvider):
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name

    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_type="mock",
            model_name=self.model_name,
            model_version="mock-0",
            quantization=None,
            context_window=4096,
        )

    def health_check(self) -> bool:
        return True

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
        start = time.perf_counter()
        text = f"[mock response to: {prompt[:60]}...]"
        latency = time.perf_counter() - start
        return GenerationResult(text, len(prompt.split()), len(text.split()), latency)

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
        start = time.perf_counter()
        obj = self._synthesize(prompt, json_schema)
        text = json.dumps(obj)
        latency = time.perf_counter() - start
        return GenerationResult(text, len(prompt.split()), len(text.split()), latency, raw_response=obj)

    def _synthesize(self, prompt: str, json_schema: dict) -> dict:
        properties = json_schema.get("properties", {})

        if "verdict" in properties:
            return {
                "verdict": "UNCERTAIN",
                "confidence": 0.5,
                "reason": "MockProvider cannot perform real verification; flagged for human review.",
                "evidence": [],
            }

        if set(properties.keys()) == {"answer"}:
            return {"answer": "0"}

        if "items" in properties and properties["items"].get("type") == "array":
            # Phase 5 batch-generation shape: {"items": [<mcq>, ...]}.
            # Parse "Generate exactly N ..." from the executor's own batch
            # prompt wording; fall back to the schema's own bound if that
            # phrasing isn't found (e.g. a hand-written test prompt).
            count_match = re.search(r"[Gg]enerate exactly (\d+)", prompt)
            if count_match:
                n = int(count_match.group(1))
            else:
                n = properties["items"].get("maxItems", 1)
            return {"items": [self._synthesize_mcq(prompt, index=i) for i in range(n)]}

        return self._synthesize_mcq(prompt, index=0)

    def _synthesize_mcq(self, prompt: str, *, index: int) -> dict:
        # Derive a plausible-looking stem/options from the prompt's own
        # "Instruction:" / topic text, varied by `index`, so a batch of N
        # items -- or N separate single-item calls -- don't all produce
        # byte-identical output (which would trivially pass dedup checks
        # it shouldn't) and so per-item processing can be tested for real.
        topic_match = re.search(r"Instruction:\s*(.+)", prompt)
        topic_text = topic_match.group(1)[:80] if topic_match else "the supplied context"
        return {
            "question": f"[MOCK #{index}] Which statement best follows from {topic_text}?",
            "options": [f"Option A-{index}", f"Option B-{index}", f"Option C-{index}", f"Option D-{index}"],
            "correct_option": 0,
            "explanation": "MockProvider placeholder explanation -- not a real academic answer.",
            "topic": "mock-topic",
            "subtopic": "",
            "difficulty": "medium",
            "bloom_level": "understand",
            "sources": [],
            "confidence": 0.5,
        }
