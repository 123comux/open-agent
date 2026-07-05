"""Tests for the :class:`Settings` configuration model."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from open_agent.config import Settings

# Every OPEN_AGENT_ env var recognized by Settings.load().
_ENV_KEYS = [
    "MODEL_PROVIDER",
    "API_KEY",
    "BASE_URL",
    "MODEL_NAME",
    "MAX_STEPS",
    "REQUEST_TIMEOUT",
    "SERVER_HOST",
    "SERVER_PORT",
    "SHORT_TERM_MEMORY_SIZE",
    "SESSION_STORAGE_DIR",
]


def test_settings_defaults():
    settings = Settings()
    assert settings.model_provider == "openai"
    assert settings.api_key == ""
    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.model_name == "gpt-4o-mini"
    assert settings.max_steps == 10
    assert settings.request_timeout == 60.0
    assert settings.server_host == "127.0.0.1"
    assert settings.server_port == 8000
    assert settings.short_term_memory_size == 20


def test_settings_load_defaults_when_no_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(f"OPEN_AGENT_{key}", raising=False)

    settings = Settings.load()
    assert settings.model_provider == "openai"
    assert settings.api_key == ""
    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.model_name == "gpt-4o-mini"
    assert settings.max_steps == 10
    assert settings.request_timeout == 60.0
    assert settings.server_host == "127.0.0.1"
    assert settings.server_port == 8000
    assert settings.short_term_memory_size == 20


def test_settings_load_with_env_vars(monkeypatch):
    monkeypatch.setenv("OPEN_AGENT_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("OPEN_AGENT_API_KEY", "secret-key")
    monkeypatch.setenv("OPEN_AGENT_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("OPEN_AGENT_MODEL_NAME", "claude-3")
    monkeypatch.setenv("OPEN_AGENT_MAX_STEPS", "5")
    monkeypatch.setenv("OPEN_AGENT_REQUEST_TIMEOUT", "120.5")
    monkeypatch.setenv("OPEN_AGENT_SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("OPEN_AGENT_SERVER_PORT", "9000")
    monkeypatch.setenv("OPEN_AGENT_SHORT_TERM_MEMORY_SIZE", "50")

    settings = Settings.load()
    assert settings.model_provider == "anthropic"
    assert settings.api_key == "secret-key"
    assert settings.base_url == "https://api.anthropic.com"
    assert settings.model_name == "claude-3"
    assert settings.max_steps == 5
    assert settings.request_timeout == 120.5
    assert settings.server_host == "0.0.0.0"
    assert settings.server_port == 9000
    assert settings.short_term_memory_size == 50


def test_settings_load_supports_ollama_provider(monkeypatch):
    monkeypatch.setenv("OPEN_AGENT_MODEL_PROVIDER", "ollama")
    settings = Settings.load()
    assert settings.model_provider == "ollama"


def test_settings_load_invalid_provider_falls_back_to_openai(monkeypatch):
    monkeypatch.setenv("OPEN_AGENT_MODEL_PROVIDER", "invalid-provider")
    settings = Settings.load()
    assert settings.model_provider == "openai"


def test_malformed_int_env_falls_back_to_default(monkeypatch):
    """A non-numeric OPEN_AGENT_MAX_STEPS falls back to the default of 10."""
    monkeypatch.setenv("OPEN_AGENT_MAX_STEPS", "abc")
    settings = Settings.load()
    assert settings.max_steps == 10


def test_to_safe_dict_masks_secret_fields():
    """to_safe_dict() masks fields whose name ends with _key/_token/_secret."""
    settings = Settings(
        api_key="sk-secret",
        api_auth_token="tok-abc",
        langfuse_secret_key="lf-secret",
    )
    safe = settings.to_safe_dict()
    assert safe["api_key"] == "<redacted>"
    assert safe["api_auth_token"] == "<redacted>"
    assert safe["langfuse_secret_key"] == "<redacted>"
    # Non-secret fields remain visible.
    assert safe["model_provider"] == "openai"


def test_chunk_overlap_must_be_less_than_chunk_size():
    """chunk_overlap >= chunk_size raises a ValidationError."""
    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=101)
