"""Central, env-driven configuration for the Claude Legal pipeline.

Every module reads its settings from a single :class:`Settings` instance so the
orchestrator (``pipeline.py``) stays the only wiring point. Values come from
environment variables prefixed ``CLAUDE_LEGAL_`` (see ``.env.example``).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_LEGAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Module 1: Core Data & Storage ----
    local_db_path: Path = Path("./claude_legal_data/claude_legal.db")
    backup_dir: Path = Path("./claude_legal_data/backups")
    backup_key: Optional[str] = None
    gdrive_credentials_json: Optional[Path] = None
    gdrive_folder_id: Optional[str] = None

    # ---- Module 2: Processing, Integration & Routing ----
    api_host: str = "127.0.0.1"
    api_port: int = 8044
    webhook_urls: List[str] = Field(default_factory=list)

    # ---- Module 3: AI Analysis & Verification ----
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-opus-4-8"

    # ---- Block A: Vector-Matrix engine tuning ----
    tfidf_max_features: int = 50_000
    tfidf_ngram_max: int = 2
    lsh_nbits: int = 256
    query_top_k: int = 10

    @field_validator("webhook_urls", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [u.strip() for u in v.split(",") if u.strip()]
        return v

    # Convenience predicates the modules use to decide "real call" vs "no-op".
    @property
    def gdrive_enabled(self) -> bool:
        return bool(self.gdrive_credentials_json and self.gdrive_folder_id)

    @property
    def backup_enabled(self) -> bool:
        return bool(self.backup_key)

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def webhooks_enabled(self) -> bool:
        return bool(self.webhook_urls)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
