"""Structured logging configuration for open-agent."""
from __future__ import annotations

import logging
import sys
from typing import Any


class AgentFormatter(logging.Formatter):
    """Custom formatter with structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        # Build a structured log line
        level = record.levelname
        name = record.name
        msg = record.getMessage()

        # Add extra fields if present
        extras: list[str] = []
        for key in ("session_id", "tool_name", "step", "model", "duration_ms"):
            val = getattr(record, key, None)
            if val is not None:
                extras.append(f"{key}={val}")

        extra_str = f" [{', '.join(extras)}]" if extras else ""
        return f"{level:<7} {name:<20} | {msg}{extra_str}"


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the open-agent package."""
    root = logging.getLogger("open_agent")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(AgentFormatter())
        root.addHandler(handler)

    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the open_agent namespace."""
    return logging.getLogger(f"open_agent.{name}")
