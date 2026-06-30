"""Tests for the :class:`Planner` response parsing."""
from __future__ import annotations

from open_agent.agent.planner import DirectResponse, Planner
from open_agent.models.base import ModelResponse, ToolCall


def test_parse_direct_response_with_no_tool_calls():
    planner = Planner()
    resp = ModelResponse(content="The answer is 42.")
    plan = planner.parse(resp)
    assert isinstance(plan, DirectResponse)
    assert plan.text == "The answer is 42."


def test_parse_prefers_structured_tool_calls():
    planner = Planner()
    call = ToolCall(name="shell", arguments={"command": "echo hi"})
    resp = ModelResponse(content="calling tool", tool_calls=[call])
    plan = planner.parse(resp)
    assert isinstance(plan, ToolCall)
    assert plan.name == "shell"
    assert plan.arguments == {"command": "echo hi"}


def test_parse_fenced_json_tool_call():
    planner = Planner()
    content = (
        'I will run a command.\n'
        '```json\n'
        '{"name": "shell", "arguments": {"command": "echo hi"}}\n'
        '```'
    )
    resp = ModelResponse(content=content)
    plan = planner.parse(resp)
    assert isinstance(plan, ToolCall)
    assert plan.name == "shell"
    assert plan.arguments == {"command": "echo hi"}


def test_parse_bare_json_tool_call():
    planner = Planner()
    content = '{"name": "shell", "arguments": {"command": "echo hi"}}'
    resp = ModelResponse(content=content)
    plan = planner.parse(resp)
    assert isinstance(plan, ToolCall)
    assert plan.name == "shell"
    assert plan.arguments == {"command": "echo hi"}


def test_parse_tool_call_with_string_arguments():
    """When ``arguments`` is a JSON-encoded string it should be decoded."""
    planner = Planner()
    content = '{"name": "shell", "arguments": "{\\"command\\": \\"echo hi\\"}"}'
    resp = ModelResponse(content=content)
    plan = planner.parse(resp)
    assert isinstance(plan, ToolCall)
    assert plan.name == "shell"
    assert plan.arguments == {"command": "echo hi"}


def test_parse_malformed_json_falls_back_to_direct_response():
    planner = Planner()
    content = '{"name": "shell", "arguments": '
    resp = ModelResponse(content=content)
    plan = planner.parse(resp)
    assert isinstance(plan, DirectResponse)
    assert plan.text == content


def test_parse_json_without_name_is_direct_response():
    planner = Planner()
    content = '{"foo": "bar"}'
    resp = ModelResponse(content=content)
    plan = planner.parse(resp)
    assert isinstance(plan, DirectResponse)
    assert plan.text == content


def test_parse_empty_content_is_direct_response():
    planner = Planner()
    resp = ModelResponse(content="")
    plan = planner.parse(resp)
    assert isinstance(plan, DirectResponse)
    assert plan.text == ""


def test_parse_plain_text_not_starting_with_brace_is_direct_response():
    planner = Planner()
    resp = ModelResponse(content="Just a normal sentence with {braces} inside.")
    plan = planner.parse(resp)
    assert isinstance(plan, DirectResponse)
    assert "braces" in plan.text
