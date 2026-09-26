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


settings = Settings()
