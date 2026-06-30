"""Tests for the model interface and shared pydantic schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from open_agent.models.base import (
    Message,
    ModelInterface,
    ModelResponse,
    ToolCall,
    ToolSchema,
)


def test_message_basic():
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_message_requires_role_and_content():
    with pytest.raises(ValidationError):
        Message(role="user")  # missing content
    with pytest.raises(ValidationError):
        Message(content="hi")  # missing role


def test_tool_schema_fields():
    schema = ToolSchema(
        name="shell",
        description="Run a shell command.",
        parameters={"type": "object"},
    )
    assert schema.name == "shell"
    assert schema.description == "Run a shell command."
    assert schema.parameters == {"type": "object"}


def test_tool_schema_requires_all_fields():
    with pytest.raises(ValidationError):
        ToolSchema(name="x")  # missing description and parameters
    with pytest.raises(ValidationError):
        ToolSchema(name="x", description="d")  # missing parameters


def test_tool_call_fields():
    call = ToolCall(name="shell", arguments={"command": "echo hi"})
    assert call.name == "shell"
    assert call.arguments == {"command": "echo hi"}


def test_tool_call_requires_arguments():
    with pytest.raises(ValidationError):
        ToolCall(name="shell")  # missing arguments


def test_model_response_defaults():
    resp = ModelResponse()
    assert resp.content == ""
    assert resp.tool_calls == []


def test_model_response_with_tool_calls():
    call = ToolCall(name="shell", arguments={"command": "echo hi"})
    resp = ModelResponse(content="thinking...", tool_calls=[call])
    assert resp.content == "thinking..."
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "shell"
    assert resp.tool_calls[0].arguments == {"command": "echo hi"}


def test_model_response_tool_calls_default_is_independent():
    """Each default ``tool_calls`` must be a fresh list (not a shared mutable)."""
    a = ModelResponse()
    b = ModelResponse()
    a.tool_calls.append(ToolCall(name="x", arguments={}))
    assert b.tool_calls == []


def test_model_interface_is_abstract():
    with pytest.raises(TypeError):
        ModelInterface()  # type: ignore[abstract]
