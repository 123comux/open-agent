"""Configuration management for the open-agent project.

Settings are loaded from environment variables prefixed with ``OPEN_AGENT_``.
Uses plain pydantic models so no ``pydantic-settings`` dependency is required.
"""
from __future__ import annotations

import os
from typing import Literal, cast

from pydantic import BaseModel, model_validator

_PROVIDER_VALUES = ("openai", "anthropic", "ollama", "zhipu")
_OBSERVABILITY_VALUES = ("local", "langsmith", "langfuse")


def _parse_bool(name: str, default: bool) -> bool:
    """Parse OPEN_AGENT_<name> as a bool. Accepts true/1/yes/on (case-insensitive) as True."""
    raw = os.environ.get(f"OPEN_AGENT_{name}")
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("true", "1", "yes", "on")


def _parse_int(name: str, default: int) -> int:
    """Parse OPEN_AGENT_<name> as int, falling back to default on parse error."""
    raw = os.environ.get(f"OPEN_AGENT_{name}")
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def _parse_float(name: str, default: float) -> float:
    """Parse OPEN_AGENT_<name> as float, falling back to default on parse error."""
    raw = os.environ.get(f"OPEN_AGENT_{name}")
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        return default


def _parse_list(name: str, default: list[str]) -> list[str]:
    """Parse OPEN_AGENT_<name> as a comma-separated list."""
    raw = os.environ.get(f"OPEN_AGENT_{name}")
    if raw is None or raw == "":
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings(BaseModel):
    """Runtime configuration for the agent."""

    # Secret field names — values are masked in to_safe_dict()
    _SECRET_FIELD_SUFFIXES = ("_key", "_token", "_secret")

    model_provider: Literal["openai", "anthropic", "ollama", "zhipu"] = "openai"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o-mini"
    max_steps: int = 10
    max_context_tokens: int = 8000
    request_timeout: float = 60.0
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    short_term_memory_size: int = 20
    session_storage_dir: str = ".open_agent_sessions"

    # RAG / embeddings
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    chunk_size: int = 500
    chunk_overlap: int = 50
    split_unit: Literal["char", "paragraph"] = "char"
    rag_top_k: int = 5

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_k: int = 20

    # MCP servers
    mcp_servers_file: str = ""

    # Tool enablement
    enabled_tools: list[str] = []

    # Long-term memory
    enable_long_term_memory: bool = False
    long_term_memory_dir: str = ".open_agent_long_term"
    long_term_memory_top_k: int = 3

    # Tool sandbox
    enable_tool_sandbox: bool = False
    sandbox_allowed_paths: list[str] = []
    sandbox_blocked_paths: list[str] = []

    # API security
    api_auth_token: str = ""
    cors_origins: list[str] = []

    # Observability
    enable_observability: bool = True
    observability_output_dir: str = ".open_agent_traces"
    observability_provider: Literal["local", "langsmith", "langfuse"] = "local"
    langsmith_api_key: str = ""
    langsmith_api_url: str = ""
    langsmith_project: str = "open-agent"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @classmethod
    def load(cls) -> Settings:
        """Load settings from environment variables (``OPEN_AGENT_`` prefix)."""

        def env(name: str, default: str | None = None) -> str | None:
            return os.environ.get(f"OPEN_AGENT_{name}", default)

        provider_raw = env("MODEL_PROVIDER", "openai") or "openai"
        provider: Literal["openai", "anthropic", "ollama", "zhipu"] = cast(
            "Literal['openai', 'anthropic', 'ollama', 'zhipu']",
            provider_raw if provider_raw in _PROVIDER_VALUES else "openai",
        )
        obs_provider_raw = env("OBSERVABILITY_PROVIDER", "local") or "local"
        observability_provider: Literal["local", "langsmith", "langfuse"] = cast(
            "Literal['local', 'langsmith', 'langfuse']",
            obs_provider_raw
            if obs_provider_raw in _OBSERVABILITY_VALUES
            else "local",
        )
        su_raw = env("SPLIT_UNIT", "char") or "char"
        split_unit: Literal["char", "paragraph"] = cast(
            "Literal['char', 'paragraph']",
            su_raw if su_raw in ("char", "paragraph") else "char",
        )
        return cls(
            model_provider=provider,
            api_key=env("API_KEY", "") or "",
            base_url=env("BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1",
            model_name=env("MODEL_NAME", "gpt-4o-mini") or "gpt-4o-mini",
            max_steps=_parse_int("MAX_STEPS", 10),
            max_context_tokens=_parse_int("MAX_CONTEXT_TOKENS", 8000),
            request_timeout=_parse_float("REQUEST_TIMEOUT", 60.0),
            server_host=env("SERVER_HOST", "127.0.0.1") or "127.0.0.1",
            server_port=_parse_int("SERVER_PORT", 8000),
            short_term_memory_size=_parse_int("SHORT_TERM_MEMORY_SIZE", 20),
            session_storage_dir=env("SESSION_STORAGE_DIR", ".open_agent_sessions")
            or ".open_agent_sessions",
            embedding_model=env("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
            or "BAAI/bge-small-zh-v1.5",
            chunk_size=_parse_int("CHUNK_SIZE", 500),
            chunk_overlap=_parse_int("CHUNK_OVERLAP", 50),
            split_unit=split_unit,
            rag_top_k=_parse_int("RAG_TOP_K", 5),
            reranker_model=env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
            or "BAAI/bge-reranker-v2-m3",
            rerank_k=_parse_int("RERANK_K", 20),
            mcp_servers_file=env("MCP_SERVERS_FILE", "") or "",
            enabled_tools=_parse_list("ENABLED_TOOLS", []),
            enable_long_term_memory=_parse_bool("ENABLE_LONG_TERM_MEMORY", False),
            long_term_memory_dir=env(
                "LONG_TERM_MEMORY_DIR", ".open_agent_long_term"
            )
            or ".open_agent_long_term",
            long_term_memory_top_k=_parse_int("LONG_TERM_MEMORY_TOP_K", 3),
            enable_tool_sandbox=_parse_bool("ENABLE_TOOL_SANDBOX", False),
            sandbox_allowed_paths=_parse_list("SANDBOX_ALLOWED_PATHS", []),
            sandbox_blocked_paths=_parse_list("SANDBOX_BLOCKED_PATHS", []),
            api_auth_token=env("API_AUTH_TOKEN", "") or "",
            cors_origins=_parse_list("CORS_ORIGINS", []),
            enable_observability=_parse_bool("ENABLE_OBSERVABILITY", True),
            observability_output_dir=env(
                "OBSERVABILITY_OUTPUT_DIR", ".open_agent_traces"
            )
            or ".open_agent_traces",
            observability_provider=observability_provider,
            langsmith_api_key=env("LANGSMITH_API_KEY", "") or "",
            langsmith_api_url=env(
                "LANGSMITH_API_URL", "https://api.smith.langchain.com"
            )
            or "https://api.smith.langchain.com",
            langsmith_project=env("LANGSMITH_PROJECT", "open-agent")
            or "open-agent",
            langfuse_public_key=env("LANGFUSE_PUBLIC_KEY", "") or "",
            langfuse_secret_key=env("LANGFUSE_SECRET_KEY", "") or "",
            langfuse_host=env("LANGFUSE_HOST", "https://cloud.langfuse.com")
            or "https://cloud.langfuse.com",
        )

    @model_validator(mode="after")
    def _validate_chunk_sizes(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    def to_safe_dict(self) -> dict[str, object]:
        """Return settings as a dict with secret fields masked.

        Fields whose name ends with ``_key``, ``_token``, or ``_secret`` have
        their value replaced with ``"<redacted>"`` (or ``""`` if unset) so the
        dict can be safely returned from a GET /api/settings endpoint without
        leaking credentials.
        """
        data = self.model_dump()
        for field_name, value in list(data.items()):
            if field_name.endswith(self._SECRET_FIELD_SUFFIXES):
                data[field_name] = "<redacted>" if value else ""
        return data


_DEFAULT_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Return a cached singleton :class:`Settings` loaded from the environment."""
    global _DEFAULT_SETTINGS
    if _DEFAULT_SETTINGS is None:
        _DEFAULT_SETTINGS = Settings.load()
    return _DEFAULT_SETTINGS


def set_settings(settings: Settings) -> None:
    """Replace the cached singleton :class:`Settings`.

    Used by the server to apply runtime configuration changes from the UI.
    Callers are responsible for rebuilding any objects that depend on the old
    settings (model, agent, registry, etc.).
    """
    global _DEFAULT_SETTINGS
    _DEFAULT_SETTINGS = settings


def reload_settings() -> Settings:
    """Force re-read settings from the environment, clearing the cached singleton."""
    global _DEFAULT_SETTINGS
    _DEFAULT_SETTINGS = None
    return get_settings()
