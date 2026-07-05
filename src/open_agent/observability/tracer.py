"""Lightweight tracing implementation for Open Agent.

A ``Tracer`` records a tree of spans for each agent run. Spans capture timing,
inputs/outputs and metrics (tokens, cost, latency). ``LocalJsonlTracer``
persists traces to a local JSONL file so they can be inspected later.
"""
from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class TraceSpan:
    """A single operation within a trace (e.g. an LLM call or tool execution)."""

    id: str
    parent_id: str | None
    trace_id: str
    type: str
    name: str
    start_time: str
    input: Any = field(default_factory=dict)
    output: Any = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    end_time: str | None = None
    children: list[TraceSpan] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "type": self.type,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "status": self.status,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class Trace:
    """Top-level trace representing one end-to-end agent run."""

    id: str
    name: str
    start_time: str
    input: Any
    metadata: dict[str, Any]
    root_span: TraceSpan | None = None
    end_time: str | None = None
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "input": self.input,
            "metadata": self.metadata,
            "status": self.status,
            "root_span": self.root_span.to_dict() if self.root_span else None,
        }


class Tracer(ABC):
    """Abstract tracer interface."""

    @abstractmethod
    def start_trace(
        self, name: str, input_data: Any, metadata: dict[str, Any] | None = None
    ) -> Trace:
        """Begin a new top-level trace."""

    @abstractmethod
    def start_span(
        self,
        trace: Trace,
        parent: TraceSpan | None,
        type_: str,
        name: str,
        input_data: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan:
        """Begin a child span."""

    @abstractmethod
    def end_span(
        self,
        span: TraceSpan,
        output_data: Any | None = None,
        metrics: dict[str, Any] | None = None,
        status: str = "ok",
    ) -> None:
        """Finalize a span and attach its output/metrics."""

    @abstractmethod
    def end_trace(
        self,
        trace: Trace,
        output_data: Any | None = None,
        status: str = "ok",
    ) -> None:
        """Finalize and persist a trace."""

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


class NoOpTracer(Tracer):
    """Tracer that does nothing; used when observability is disabled."""

    def start_trace(
        self, name: str, input_data: Any, metadata: dict[str, Any] | None = None
    ) -> Trace:
        return Trace(
            id=str(uuid.uuid4()),
            name=name,
            start_time=self._now(),
            input=input_data,
            metadata=metadata or {},
        )

    def start_span(
        self,
        trace: Trace,
        parent: TraceSpan | None,
        type_: str,
        name: str,
        input_data: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan:
        return TraceSpan(
            id=str(uuid.uuid4()),
            parent_id=parent.id if parent else None,
            trace_id=trace.id,
            type=type_,
            name=name,
            start_time=self._now(),
            input=input_data,
            metadata=metadata or {},
        )

    def end_span(
        self,
        span: TraceSpan,
        output_data: Any | None = None,
        metrics: dict[str, Any] | None = None,
        status: str = "ok",
    ) -> None:
        span.end_time = self._now()
        span.output = output_data
        span.metrics = metrics or {}
        span.status = status

    def end_trace(
        self,
        trace: Trace,
        output_data: Any | None = None,
        status: str = "ok",
    ) -> None:
        trace.end_time = self._now()
        trace.status = status


class LocalJsonlTracer(NoOpTracer):
    """Persist traces to a local JSONL file, one trace per line.

    Args:
        output_dir: Directory where ``traces.jsonl`` is written. Created if
            it does not exist.
    """

    def __init__(self, output_dir: str | Path = ".open_agent_traces") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._file = self.output_dir / "traces.jsonl"
        # Open in line-buffered mode (``buffering=1``) so each completed line
        # is pushed to the OS as soon as it is written.
        #
        # Known limitation: ``end_trace`` is intentionally synchronous to
        # honour the shared :class:`Tracer` interface (callers in
        # ``Agent.run``/``run_stream`` invoke it directly from async code), so
        # the write still blocks the event loop briefly. Fully non-blocking
        # persistence would require either making ``end_trace`` a coroutine
        # (breaking the sync contract used by ``NoOpTracer``) or wrapping the
        # call site in ``asyncio.to_thread`` in the agent layer (out of scope
        # for this module). Line buffering keeps each write cheap and bounds
        # the blocking window to a single line.
        self._file_handle = self._file.open("a", encoding="utf-8", buffering=1)

    def end_trace(
        self,
        trace: Trace,
        output_data: Any | None = None,
        status: str = "ok",
    ) -> None:
        super().end_trace(trace, output_data, status)
        self._file_handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        self._file_handle.flush()

    async def aclose(self) -> None:
        """Close the persistent file handle. Safe to call multiple times."""
        if not self._file_handle.closed:
            self._file_handle.close()

    def list_traces(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent persisted traces (newest first)."""
        if not self._file.exists():
            return []
        lines = self._file.read_text(encoding="utf-8").strip().split("\n")
        traces = [json.loads(line) for line in lines if line.strip()]
        traces.reverse()
        return traces[:limit]

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Load a single trace by id."""
        for trace in self.list_traces(limit=10000):
            if trace.get("id") == trace_id:
                return trace
        return None


class LangfuseTracer(NoOpTracer):
    """Forward traces and spans to Langfuse.

    Requires the optional ``langfuse`` package (v2+). Falls back to the
    in-memory behaviour of :class:`NoOpTracer` for any Langfuse API errors so
    that agent execution is never blocked by the observability backend.

    Args:
        public_key: Langfuse public key. If ``None``, reads from the standard
            ``LANGFUSE_PUBLIC_KEY`` environment variable.
        secret_key: Langfuse secret key. If ``None``, reads from the standard
            ``LANGFUSE_SECRET_KEY`` environment variable.
        host: Langfuse API host. If ``None``, uses the default
            ``https://cloud.langfuse.com``.
    """

    _TYPE_MAP: dict[str, Literal["agent", "generation", "tool", "retriever", "span"]] = {
        "agent": "agent",
        "llm": "generation",
        "tool": "tool",
        "retrieval": "retriever",
        "retriever": "retriever",
    }

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
    ) -> None:
        from langfuse import Langfuse

        self.client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        self._observations: dict[str, Any] = {}

    @staticmethod
    def _to_observation_type(
        span_type: str,
    ) -> Literal["agent", "generation", "tool", "retriever", "span"]:
        return LangfuseTracer._TYPE_MAP.get(span_type, "span")

    def _start_observation(
        self,
        name: str,
        span_type: str,
        input_data: Any | None,
        metadata: dict[str, Any] | None,
    ) -> Any:
        """Call Langfuse ``start_observation`` with a literal ``as_type``.

        ``start_observation`` is overloaded on ``as_type`` (each literal yields
        a different return type). We dispatch dynamically here, so the type is
        narrowed via conditional branches to satisfy mypy's overload
        resolution. The result is returned as ``Any`` because callers store it
        in a heterogeneous dict.
        """
        obs_type = self._to_observation_type(span_type)
        input_val: Any = input_data or {}
        meta_val: dict[str, Any] = metadata or {}
        if obs_type == "agent":
            return self.client.start_observation(
                name=name, as_type="agent", input=input_val, metadata=meta_val
            )
        if obs_type == "generation":
            return self.client.start_observation(
                name=name, as_type="generation", input=input_val, metadata=meta_val
            )
        if obs_type == "tool":
            return self.client.start_observation(
                name=name, as_type="tool", input=input_val, metadata=meta_val
            )
        if obs_type == "retriever":
            return self.client.start_observation(
                name=name, as_type="retriever", input=input_val, metadata=meta_val
            )
        return self.client.start_observation(
            name=name, as_type="span", input=input_val, metadata=meta_val
        )

    def start_trace(
        self, name: str, input_data: Any, metadata: dict[str, Any] | None = None
    ) -> Trace:
        trace = super().start_trace(name, input_data, metadata)
        try:
            obs = self.client.start_observation(
                name=name,
                as_type="chain",
                input=input_data,
                metadata=metadata or {},
            )
            self._observations[trace.id] = obs
        except Exception as exc:  # pragma: no cover - optional backend
            logger.debug("observability backend call failed: %s", exc, exc_info=True)
        return trace

    def start_span(
        self,
        trace: Trace,
        parent: TraceSpan | None,
        type_: str,
        name: str,
        input_data: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan:
        span = super().start_span(trace, parent, type_, name, input_data, metadata)
        try:
            if parent is None:
                # Top-level spans are reported as separate observations on the
                # trace so their type is preserved in Langfuse.
                obs = self._start_observation(name, type_, input_data, metadata)
            else:
                parent_obs = self._observations.get(parent.id)
                if parent_obs is not None:
                    obs = parent_obs.start_observation(
                        name=name,
                        as_type=self._to_observation_type(type_),
                        input=input_data or {},
                        metadata=metadata or {},
                    )
                else:
                    obs = self._start_observation(name, type_, input_data, metadata)
            self._observations[span.id] = obs
        except Exception as exc:  # pragma: no cover - optional backend
            logger.debug("observability backend call failed: %s", exc, exc_info=True)
        return span

    def end_span(
        self,
        span: TraceSpan,
        output_data: Any | None = None,
        metrics: dict[str, Any] | None = None,
        status: str = "ok",
    ) -> None:
        super().end_span(span, output_data, metrics, status)
        try:
            obs = self._observations.get(span.id)
            if obs is not None:
                level = "DEFAULT" if status == "ok" else "ERROR"
                obs.end(
                    output=output_data if output_data is not None else {},
                    metadata={"metrics": metrics or {}, "status": status},
                    level=level,
                    status_message=None if status == "ok" else str(output_data),
                )
            # Drop the observation so the dict does not grow unbounded across
            # many traces/spans; safe to pop even when no obs was recorded.
            self._observations.pop(span.id, None)
        except Exception as exc:  # pragma: no cover - optional backend
            logger.debug("langfuse end_span failed: %s", exc, exc_info=True)

    def end_trace(
        self,
        trace: Trace,
        output_data: Any | None = None,
        status: str = "ok",
    ) -> None:
        super().end_trace(trace, output_data, status)
        try:
            obs = self._observations.get(trace.id)
            if obs is not None:
                level = "DEFAULT" if status == "ok" else "ERROR"
                obs.end(
                    output=output_data if output_data is not None else {},
                    metadata={"status": status},
                    level=level,
                    status_message=None if status == "ok" else str(output_data),
                )
            # Drop the observation so the dict does not grow unbounded across
            # many traces/spans; safe to pop even when no obs was recorded.
            self._observations.pop(trace.id, None)
        except Exception as exc:  # pragma: no cover - optional backend
            logger.debug("langfuse end_trace failed: %s", exc, exc_info=True)


class LangSmithTracer(NoOpTracer):
    """Forward traces and spans to LangSmith.

    Requires the optional ``langsmith`` package. Falls back to the in-memory
    behaviour of :class:`NoOpTracer` for any LangSmith API errors so that agent
    execution is never blocked by the observability backend.

    Args:
        api_key: LangSmith API key. If ``None``, reads from the standard
            ``LANGSMITH_API_KEY`` environment variable.
        api_url: LangSmith API endpoint. If ``None``, uses the client default.
        project_name: LangSmith project name. If ``None``, uses the default
            project configured for the API key.
    """

    _RUN_TYPE_MAP: dict[str, Literal["tool", "chain", "llm", "retriever"]] = {
        "agent": "chain",
        "llm": "llm",
        "tool": "tool",
        "retrieval": "retriever",
        "retriever": "retriever",
    }

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        project_name: str | None = None,
    ) -> None:
        from langsmith import Client

        self.client = Client(api_key=api_key, api_url=api_url)
        self.project_name = project_name

    @staticmethod
    def _to_run_type(
        span_type: str,
    ) -> Literal["tool", "chain", "llm", "retriever"]:
        return LangSmithTracer._RUN_TYPE_MAP.get(span_type, "chain")

    @staticmethod
    def _to_dt(iso_str: str) -> datetime:
        return datetime.fromisoformat(iso_str)

    def start_trace(
        self, name: str, input_data: Any, metadata: dict[str, Any] | None = None
    ) -> Trace:
        trace = super().start_trace(name, input_data, metadata)
        try:
            self.client.create_run(
                name=name,
                run_type="chain",
                id=uuid.UUID(trace.id),
                inputs=input_data,
                start_time=self._to_dt(trace.start_time),
                extra={"metadata": metadata or {}},
                project_name=self.project_name,
            )
        except Exception as exc:  # pragma: no cover - optional backend
            logger.debug("observability backend call failed: %s", exc, exc_info=True)
        return trace

    def start_span(
        self,
        trace: Trace,
        parent: TraceSpan | None,
        type_: str,
        name: str,
        input_data: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan:
        span = super().start_span(trace, parent, type_, name, input_data, metadata)
        try:
            parent_id = uuid.UUID(parent.id) if parent else uuid.UUID(trace.id)
            self.client.create_run(
                name=name,
                run_type=self._to_run_type(type_),
                id=uuid.UUID(span.id),
                parent_run_id=parent_id,
                inputs=input_data or {},
                start_time=self._to_dt(span.start_time),
                extra={"metadata": metadata or {}, "span_type": type_},
                project_name=self.project_name,
            )
        except Exception as exc:  # pragma: no cover - optional backend
            logger.debug("observability backend call failed: %s", exc, exc_info=True)
        return span

    def end_span(
        self,
        span: TraceSpan,
        output_data: Any | None = None,
        metrics: dict[str, Any] | None = None,
        status: str = "ok",
    ) -> None:
        super().end_span(span, output_data, metrics, status)
        try:
            error = None
            if status != "ok":
                error = str(output_data) if output_data is not None else status
            self.client.update_run(
                run_id=uuid.UUID(span.id),
                outputs=output_data if output_data is not None else {},
                end_time=self._to_dt(span.end_time) if span.end_time else None,
                error=error,
                extras={"metrics": metrics or {}, "status": status},
            )
        except Exception as exc:  # pragma: no cover - optional backend
            logger.debug("observability backend call failed: %s", exc, exc_info=True)

    def end_trace(
        self,
        trace: Trace,
        output_data: Any | None = None,
        status: str = "ok",
    ) -> None:
        super().end_trace(trace, output_data, status)
        try:
            error = None
            if status != "ok":
                error = str(output_data) if output_data is not None else status
            self.client.update_run(
                run_id=uuid.UUID(trace.id),
                outputs=output_data if output_data is not None else {},
                end_time=self._to_dt(trace.end_time) if trace.end_time else None,
                error=error,
                extras={"status": status},
            )
        except Exception as exc:  # pragma: no cover - optional backend
            logger.debug("observability backend call failed: %s", exc, exc_info=True)
