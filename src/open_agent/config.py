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
        )


_DEFAULT_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Return a cached singleton :class:`Settings` loaded from the environment."""
    global _DEFAULT_SETTINGS
    if _DEFAULT_SETTINGS is None:
        _DEFAULT_SETTINGS = Settings.load()
    return _DEFAULT_SETTINGS
