"""Configuration management for the open-agent project.

Settings are loaded from environment variables prefixed with ``OPEN_AGENT_``.
Uses plain pydantic models so no ``pydantic-settings`` dependency is required.
"""
from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel

_PROVIDER_VALUES = ("openai", "anthropic", "ollama")


class Settings(BaseModel):
    """Runtime configuration for the agent."""

    model_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o-mini"
    max_steps: int = 10
    request_timeout: float = 60.0
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    short_term_memory_size: int = 20
    session_storage_dir: str = ".open_agent_sessions"

    # RAG / embeddings
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    chunk_size: int = 500
    chunk_overlap: int = 50
    split_unit: str = "char"
    rag_top_k: int = 5

    @classmethod
    def load(cls) -> Settings:
        """Load settings from environment variables (``OPEN_AGENT_`` prefix)."""

        def env(name: str, default: str | None = None) -> str | None:
            return os.environ.get(f"OPEN_AGENT_{name}", default)

        provider_raw = env("MODEL_PROVIDER", "openai") or "openai"
        provider: Literal["openai", "anthropic", "ollama"] = (
            provider_raw if provider_raw in _PROVIDER_VALUES else "openai"
        )
        return cls(
            model_provider=provider,
            api_key=env("API_KEY", "") or "",
            base_url=env("BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1",
            model_name=env("MODEL_NAME", "gpt-4o-mini") or "gpt-4o-mini",
            max_steps=int(env("MAX_STEPS", "10") or 10),
            request_timeout=float(env("REQUEST_TIMEOUT", "60") or 60),
            server_host=env("SERVER_HOST", "127.0.0.1") or "127.0.0.1",
            server_port=int(env("SERVER_PORT", "8000") or 8000),
            short_term_memory_size=int(env("SHORT_TERM_MEMORY_SIZE", "20") or 20),
            session_storage_dir=env("SESSION_STORAGE_DIR", ".open_agent_sessions")
            or ".open_agent_sessions",
            embedding_model=env("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
            or "BAAI/bge-small-zh-v1.5",
            chunk_size=int(env("CHUNK_SIZE", "500") or 500),
            chunk_overlap=int(env("CHUNK_OVERLAP", "50") or 50),
            split_unit=env("SPLIT_UNIT", "char") or "char",
            rag_top_k=int(env("RAG_TOP_K", "5") or 5),
        )


_DEFAULT_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Return a cached singleton :class:`Settings` loaded from the environment."""
    global _DEFAULT_SETTINGS
    if _DEFAULT_SETTINGS is None:
        _DEFAULT_SETTINGS = Settings.load()
    return _DEFAULT_SETTINGS
