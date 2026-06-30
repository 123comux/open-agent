"""open-agent: a general-purpose Agentic RAG autonomous work assistant.

The top-level package intentionally keeps imports light so that the core
library can be imported without pulling in optional dependencies such as
FastAPI. Submodules (``open_agent.agent``, ``open_agent.models``, ...) are
imported explicitly by consumers that need them.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
