"""Application configuration via environment variables (pydantic-settings).

No secret defaults ship in code beyond a dev-only JWT secret that is
loudly not-for-production -- see docs/PHASE0-DESIGN.md section 14 on
secret management.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AIQBE_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./aiqbe_dev.db"

    jwt_secret_key: str = "dev-only-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 8

    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    # Run Celery tasks synchronously in-process instead of via a broker/
    # worker. Production sets this False and runs `celery -A ... worker`;
    # it defaults True here so the API and test suite work without Redis
    # or a worker process running, per docs/PHASE0-DESIGN.md section 17's
    # "operational simplicity" goal for the single-workstation deployment.
    celery_task_always_eager: bool = True

    default_llm_provider_type: str = "mock"  # 'mock' | 'ollama' | 'llamacpp' | 'openai_compatible'
    default_llm_base_url: str = "http://localhost:11434"
    default_llm_model_name: str = "mock-model"

    # RAG (Phase 3). 'hashing' is a dependency-free, non-semantic dev/test
    # default -- see backend/embeddings/hashing_provider.py for why.
    # Production sets this to 'sentence_transformers' once Phase 1's model
    # benchmark also validates an embedding model choice on real hardware.
    embedding_provider_type: str = "hashing"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    # ':memory:' runs an embedded Qdrant instance with no server (dev/test
    # default); production points this at a real Qdrant deployment, e.g.
    # 'http://localhost:6333'.
    vector_store_location: str = ":memory:"

    # Controlled Internet research (Phase 4). Internet retrieval is
    # opt-in per generation job (GenerationJobCreate) and additionally
    # gated by the approved-domains list (rag/web/domain_policy.py) --
    # these settings only affect HOW an already-approved fetch behaves.
    web_fetch_timeout_seconds: float = 15.0
    web_fetch_max_bytes: int = 20 * 1024 * 1024
    web_fetch_respect_robots_txt: bool = True
    # None (default) leaves search disabled (NullSearchAdapter): an
    # institution wanting web discovery, not just direct URL ingestion,
    # points this at its own self-hosted SearXNG instance.
    searxng_base_url: str | None = None

    # MCQ generation engine (Phase 5). Upper bound on how many questions
    # one plan cell requests per LLM call -- docs/PHASE0-DESIGN.md
    # section 11 / master prompt section 9: "batches of 10-30 questions",
    # never one call per question and never thousands in one request
    # (master prompt section 52). A cell needing fewer than this simply
    # requests fewer; this is a ceiling, not a floor.
    generation_max_batch_size: int = 20
    # Upper bound on additional over-generation rounds a job will run to
    # try to close the gap between generated-valid-so-far and
    # requested_count (docs/PHASE0-DESIGN.md section 30). A job's own
    # `max_attempts` field (set per job, default 3) is the actual limit
    # used; this is only a hard ceiling against a misconfigured job.
    generation_max_rounds_ceiling: int = 10


settings = Settings()
