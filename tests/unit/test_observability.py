"""Tests for the observability tracer."""
from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from open_agent.observability.tracer import (
    LangfuseTracer,
    LangSmithTracer,
    LocalJsonlTracer,
    NoOpTracer,
)


def test_noop_tracer() -> None:
    tracer = NoOpTracer()
    trace = tracer.start_trace("test", {"q": "hello"})
    span = tracer.start_span(trace, None, "llm", "step_1")
    tracer.end_span(span, {"answer": "hi"}, {"latency_ms": 10})
    trace.root_span = span
    tracer.end_trace(trace, {"answer": "hi"})
    assert trace.status == "ok"
    assert span.trace_id == trace.id


def test_local_jsonl_tracer(tmp_path: Path) -> None:
    tracer = LocalJsonlTracer(output_dir=str(tmp_path))
    trace = tracer.start_trace("agent.run", {"user_input": "hello"})
    root = tracer.start_span(trace, None, "agent", "react_loop")
    trace.root_span = root
    llm = tracer.start_span(trace, root, "llm", "step_1")
    tracer.end_span(llm, {"content": "hi"})
    root.children.append(llm)
    tracer.end_span(root, {"response": "hi"})
    tracer.end_trace(trace, {"response": "hi"})

    traces = tracer.list_traces()
    assert len(traces) == 1
    assert traces[0]["name"] == "agent.run"
    assert traces[0]["root_span"]["children"][0]["type"] == "llm"

    loaded = tracer.get_trace(trace.id)
    assert loaded is not None
    assert loaded["id"] == trace.id


def test_local_jsonl_tracer_limit(tmp_path: Path) -> None:
    tracer = LocalJsonlTracer(output_dir=str(tmp_path))
    for i in range(3):
        trace = tracer.start_trace("run", {"i": i})
        tracer.end_trace(trace)
    assert len(tracer.list_traces(limit=2)) == 2


def test_langsmith_tracer_forwards_runs() -> None:
    mock_client = MagicMock()
    with patch("langsmith.Client", return_value=mock_client):
        tracer = LangSmithTracer(api_key="fake-key", project_name="test-project")

    trace = tracer.start_trace("agent.run", {"user_input": "hello"})
    root = tracer.start_span(trace, None, "agent", "react_loop")
    trace.root_span = root
    llm = tracer.start_span(trace, root, "llm", "step_1", input_data={"messages": []})
    tracer.end_span(llm, {"content": "hi"}, {"latency_ms": 10})
    root.children.append(llm)
    tracer.end_span(root, {"response": "hi"})
    tracer.end_trace(trace, {"response": "hi"})

    # create_run called for trace root, root span and llm span
    assert mock_client.create_run.call_count == 3
    assert mock_client.update_run.call_count == 3

    # Verify root trace run uses the trace id and correct project
    trace_call = mock_client.create_run.call_args_list[0]
    assert trace_call.kwargs["name"] == "agent.run"
    assert trace_call.kwargs["run_type"] == "chain"
    assert trace_call.kwargs["project_name"] == "test-project"
    assert trace_call.kwargs["id"] == uuid.UUID(trace.id)

    # Verify child span parent relationship
    llm_call = mock_client.create_run.call_args_list[2]
    assert llm_call.kwargs["name"] == "step_1"
    assert llm_call.kwargs["run_type"] == "llm"
    assert llm_call.kwargs["parent_run_id"] == uuid.UUID(root.id)


def test_langsmith_tracer_maps_run_types() -> None:
    mock_client = MagicMock()
    with patch("langsmith.Client", return_value=mock_client):
        tracer = LangSmithTracer(api_key="fake-key")

    trace = tracer.start_trace("run", {})
    cases = [
        ("agent", "chain"),
        ("llm", "llm"),
        ("tool", "tool"),
        ("retrieval", "retriever"),
        ("unknown", "chain"),
    ]
    for span_type, expected in cases:
        tracer.start_span(trace, None, span_type, span_type)
        call = mock_client.create_run.call_args_list[-1]
        assert call.kwargs["run_type"] == expected, span_type


def test_langsmith_tracer_graceful_on_backend_error() -> None:
    mock_client = MagicMock()
    mock_client.create_run.side_effect = RuntimeError("LangSmith down")
    mock_client.update_run.side_effect = RuntimeError("LangSmith down")
    with patch("langsmith.Client", return_value=mock_client):
        tracer = LangSmithTracer(api_key="fake-key")

    trace = tracer.start_trace("run", {"q": "hello"})
    span = tracer.start_span(trace, None, "llm", "step_1")
    tracer.end_span(span, {"answer": "hi"})
    tracer.end_trace(trace, {"answer": "hi"})

    # No exception should be raised; in-memory state is still maintained.
    assert trace.status == "ok"


def test_langfuse_tracer_forwards_observations() -> None:
    """Langfuse tracer creates observations for trace and spans."""
    mock_client = MagicMock()
    mock_root_obs = MagicMock()
    mock_llm_obs = MagicMock()
    mock_client.start_observation.return_value = mock_root_obs
    mock_root_obs.start_observation.return_value = mock_llm_obs

    with patch("langfuse.Langfuse", return_value=mock_client):
        tracer = LangfuseTracer(
            public_key="pk", secret_key="sk", host="https://langfuse.example.com"
        )

    trace = tracer.start_trace("agent.run", {"user_input": "hello"})
    root = tracer.start_span(trace, None, "agent", "react_loop")
    trace.root_span = root
    llm = tracer.start_span(trace, root, "llm", "step_1", input_data={"messages": []})
    tracer.end_span(llm, {"content": "hi"}, {"latency_ms": 10})
    root.children.append(llm)
    tracer.end_span(root, {"response": "hi"})
    tracer.end_trace(trace, {"response": "hi"})

    # Trace and root observations created via client.start_observation.
    assert mock_client.start_observation.call_count == 2
    trace_call = mock_client.start_observation.call_args_list[0]
    assert trace_call.kwargs["name"] == "agent.run"
    assert trace_call.kwargs["as_type"] == "chain"
    root_call = mock_client.start_observation.call_args_list[1]
    assert root_call.kwargs["name"] == "react_loop"
    assert root_call.kwargs["as_type"] == "agent"

    # LLM span created as child of root observation.
    mock_root_obs.start_observation.assert_called_once()
    llm_call = mock_root_obs.start_observation.call_args
    assert llm_call.kwargs["name"] == "step_1"
    assert llm_call.kwargs["as_type"] == "generation"

    # Both spans and the trace are ended.
    assert mock_llm_obs.end.call_count == 1
    assert mock_root_obs.end.call_count == 2


def test_langfuse_tracer_maps_observation_types() -> None:
    """Span types are mapped to Langfuse observation types."""
    mock_client = MagicMock()
    mock_parent_obs = MagicMock()
    mock_client.start_observation.return_value = mock_parent_obs
    mock_parent_obs.start_observation.return_value = mock_parent_obs
    with patch("langfuse.Langfuse", return_value=mock_client):
        tracer = LangfuseTracer(public_key="pk", secret_key="sk")

    trace = tracer.start_trace("run", {})
    parent = tracer.start_span(trace, None, "agent", "parent")
    cases = [
        ("agent", "agent"),
        ("llm", "generation"),
        ("tool", "tool"),
        ("retrieval", "retriever"),
        ("unknown", "span"),
    ]
    for span_type, expected in cases:
        tracer.start_span(trace, parent, span_type, span_type)
        call = mock_parent_obs.start_observation.call_args_list[-1]
        assert call.kwargs["as_type"] == expected, span_type


def test_langfuse_tracer_graceful_on_backend_error() -> None:
    """Backend errors do not break the in-memory trace state."""
    mock_client = MagicMock()
    mock_client.start_observation.side_effect = RuntimeError("Langfuse down")
    with patch("langfuse.Langfuse", return_value=mock_client):
        tracer = LangfuseTracer(public_key="pk", secret_key="sk")

    trace = tracer.start_trace("run", {"q": "hello"})
    span = tracer.start_span(trace, None, "llm", "step_1")
    tracer.end_span(span, {"answer": "hi"})
    tracer.end_trace(trace, {"answer": "hi"})

    assert trace.status == "ok"
    assert span.status == "ok"
