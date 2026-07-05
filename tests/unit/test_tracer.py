"""Tests for tracer observation lifecycle and error logging."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from open_agent.observability.tracer import LangfuseTracer, LangSmithTracer


def test_langfuse_observations_cleared_after_end() -> None:
    """``end_span``/``end_trace`` must drop observations so ``_observations``
    does not grow unbounded across many runs (regression test for H1)."""
    mock_client = MagicMock()
    mock_root_obs = MagicMock()
    mock_llm_obs = MagicMock()
    mock_client.start_observation.return_value = mock_root_obs
    mock_root_obs.start_observation.return_value = mock_llm_obs

    with patch("langfuse.Langfuse", return_value=mock_client):
        tracer = LangfuseTracer(public_key="pk", secret_key="sk")

    trace = tracer.start_trace("agent.run", {"user_input": "hello"})
    root = tracer.start_span(trace, None, "agent", "react_loop")
    llm = tracer.start_span(trace, root, "llm", "step_1", input_data={"messages": []})

    # Observations are tracked while the span/trace are live.
    assert trace.id in tracer._observations
    assert root.id in tracer._observations
    assert llm.id in tracer._observations

    tracer.end_span(llm, {"content": "hi"})
    assert llm.id not in tracer._observations
    mock_llm_obs.end.assert_called_once()

    tracer.end_span(root, {"response": "hi"})
    assert root.id not in tracer._observations

    tracer.end_trace(trace, {"response": "hi"})
    assert trace.id not in tracer._observations


def test_langfuse_backend_error_logged_at_debug(caplog) -> None:
    """Backend failures are swallowed but logged at DEBUG (M13)."""
    mock_client = MagicMock()
    mock_client.start_observation.side_effect = RuntimeError("Langfuse down")
    with patch("langfuse.Langfuse", return_value=mock_client):
        tracer = LangfuseTracer(public_key="pk", secret_key="sk")

    with caplog.at_level(logging.DEBUG, logger="open_agent.observability.tracer"):
        tracer.start_trace("run", {"q": "hello"})

    assert any(
        "observability backend call failed" in rec.getMessage()
        for rec in caplog.records
    )


def test_langsmith_backend_error_logged_at_debug(caplog) -> None:
    """LangSmith backend failures are logged at DEBUG instead of silent."""
    mock_client = MagicMock()
    mock_client.create_run.side_effect = RuntimeError("LangSmith down")
    with patch("langsmith.Client", return_value=mock_client):
        tracer = LangSmithTracer(api_key="fake-key")

    with caplog.at_level(logging.DEBUG, logger="open_agent.observability.tracer"):
        tracer.start_trace("run", {"q": "hello"})

    assert any(
        "observability backend call failed" in rec.getMessage()
        for rec in caplog.records
    )
