"""IEmbeddingProvider: the embedding-model abstraction, mirrored on
ILLMProvider (llm/providers/base.py) so the RAG pipeline is never locked
to one embedding model any more than generation is locked to one LLM.
See docs/PHASE0-DESIGN.md section 6.
"""

from __future__ import annotations

import abc
import dataclasses


@dataclasses.dataclass
class EmbeddingProviderMetadata:
    provider_type: str
    model_name: str
    dimension: int


class IEmbeddingProvider(abc.ABC):
    @abc.abstractmethod
    def metadata(self) -> EmbeddingProviderMetadata:
        ...

    @abc.abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Returns one embedding vector per input text, same order."""

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
