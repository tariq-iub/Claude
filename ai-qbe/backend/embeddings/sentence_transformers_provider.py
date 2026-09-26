"""Production embedding provider: BAAI/bge-small-en-v1.5 via
sentence-transformers (docs/PHASE0-DESIGN.md section 6's default choice).

This module has NOT been exercised end-to-end in this development
sandbox: downloading model weights from huggingface.co is blocked by this
environment's outbound network policy (confirmed: a direct request to
huggingface.co returns a proxy-level 403). The code is written to the
same `IEmbeddingProvider` interface as the dependency-free
`HashingEmbeddingProvider` used for dev/test, so switching a deployment
from the hashing fallback to this real model is a one-line config change
(`AIQBE_EMBEDDING_PROVIDER_TYPE=sentence_transformers`), not a rewrite.
Run it on the target workstation (which has normal internet access) to
confirm it loads and to benchmark bge-small vs. bge-base per Phase 0
section 6 before relying on it in production.
"""

from __future__ import annotations

from .base import EmbeddingProviderMetadata, IEmbeddingProvider


class SentenceTransformersEmbeddingProvider(IEmbeddingProvider):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        from sentence_transformers import SentenceTransformer  # local import: heavy, optional dep

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def metadata(self) -> EmbeddingProviderMetadata:
        return EmbeddingProviderMetadata(
            provider_type="sentence_transformers",
            model_name=self.model_name,
            dimension=self._dimension,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()
