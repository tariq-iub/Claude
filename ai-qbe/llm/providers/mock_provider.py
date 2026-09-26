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

import itertools
import json
import re
import time
from typing import Optional

from .base import GenerationResult, ILLMProvider, ProviderMetadata

# Process-wide, shared across every MockProvider instance (not reset per
# instance/batch/call) so a freshly-constructed MockProvider -- e.g. the
# one built fresh for each `regenerate` API call -- never starts back at
# the same index and reproduces a previous instance's exact output.
# Realistic: a real LLM wouldn't deterministically repeat the same
# question just because it's a new request. Phase 7's dedup logic is
# meant to catch genuine repeats, not an artifact of the mock resetting.
_GLOBAL_MOCK_COUNTER = itertools.count()


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

        if "is_duplicate" in properties:
            # Phase 7 dedup LLM-judge tie-break shape. Defaults to "not a
            # duplicate" -- safe because nothing auto-approves anyway, and
            # a mock false-negative here just means a human reviewer sees
            # one extra candidate, never that a real duplicate silently
            # entered the bank unexamined.
            return {
                "is_duplicate": False,
                "confidence": 0.5,
                "reason": "MockProvider cannot perform real duplicate judgment; flagged for human review.",
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
            return {"items": [self._synthesize_mcq(prompt) for _ in range(n)]}

        return self._synthesize_mcq(prompt)

    # Wholesale distinct stems (not one template with slots filled in) --
    # a real model's batch response has each item asking about a
    # different fact/aspect (the executor's own prompt explicitly asks
    # for that) using genuinely different phrasing. A fill-in-the-blank
    # template sharing most of its wording across items would make every
    # mock candidate read as a near-duplicate of its neighbors under
    # Phase 7's lexical dedup check, regardless of which word varied --
    # a mock-realism problem, not a dedup bug. This pool of 40 fully
    # distinct sentences comfortably covers one batch (max 30 items per
    # docs/PHASE0-DESIGN.md section 11) without a repeat, each pair
    # measured well under the lexical-duplicate threshold.
    _STEMS = [
        "Which option correctly identifies the key relationship described above?",
        "What conclusion follows most directly from the given information?",
        "How would the outcome change under the stated condition?",
        "Which choice best accounts for the observed behavior?",
        "What is the most reasonable interpretation of this scenario?",
        "Which factor plays the dominant role in this situation?",
        "What distinguishes the correct answer from the alternatives here?",
        "Which statement is consistent with the underlying principle?",
        "What would happen if the described quantity were doubled?",
        "Which option represents the limiting case of this process?",
        "How does this concept apply in the given context?",
        "What is the defining characteristic being tested here?",
        "Which choice reflects the correct sequence of events?",
        "What role does the mentioned quantity play overall?",
        "Which answer best matches the described classification?",
        "How should the given data be correctly interpreted?",
        "What is the expected result under standard conditions?",
        "Which option avoids the common misconception here?",
        "What determines the correct boundary in this case?",
        "Which explanation is most directly supported by the evidence?",
        "How does changing one variable affect the described system?",
        "Which term best describes the phenomenon in question?",
        "What is the correct order of magnitude for this quantity?",
        "Which choice correctly pairs cause with effect?",
        "What assumption is required for this conclusion to hold?",
        "Which option is the exception among the choices given?",
        "How is this quantity typically measured or expressed?",
        "Which answer captures the essential trade-off described?",
        "What is the most precise way to state this relationship?",
        "Which choice reflects a common but incorrect assumption?",
        "How would an expert most likely classify this case?",
        "Which option is best supported by the stated evidence?",
        "What is the net effect of the two described influences?",
        "Which answer correctly reverses the stated relationship?",
        "How does this scenario differ from the typical case?",
        "Which choice best summarizes the underlying mechanism?",
        "What is the key distinguishing feature in this comparison?",
        "Which option correctly identifies the missing quantity?",
        "How should the described process be properly sequenced?",
        "Which answer is most consistent with the given constraints?",
    ]
    _OPTION_BANKS = [
        ("Increases proportionally", "Decreases proportionally", "Remains unchanged", "Becomes undefined"),
        ("Directly", "Inversely", "Independently", "Not at all"),
        ("The first factor", "The second factor", "Both factors equally", "Neither factor"),
        ("Always true", "Sometimes true", "Never true", "True only near limits"),
        ("Early stage", "Middle stage", "Late stage", "Every stage equally"),
    ]

    def _synthesize_mcq(self, prompt: str) -> dict:
        index = next(_GLOBAL_MOCK_COUNTER)

        stem = self._STEMS[index % len(self._STEMS)]
        options = self._OPTION_BANKS[index % len(self._OPTION_BANKS)]
        return {
            "question": f"[MOCK #{index}] {stem}",
            "options": [f"{opt} (ref {index})" for opt in options],
            "correct_option": 0,
            "explanation": f"MockProvider placeholder explanation #{index} -- not a real academic answer.",
            "topic": "mock-topic",
            "subtopic": "",
            "difficulty": "medium",
            "bloom_level": "understand",
            "sources": [],
            "confidence": 0.5,
        }
