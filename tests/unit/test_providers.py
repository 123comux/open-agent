"""Tests for the OpenAI, Anthropic, and Ollama model providers."""
from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

from open_agent.models.anthropic_provider import AnthropicModel
from open_agent.models.base import Message, ToolSchema
from open_agent.models.ollama_provider import OllamaModel
from open_agent.models.openai_provider import OpenAIModel


def _ok_response(payload: dict) -> MagicMock:
    """Build a fake ``httpx.Response`` whose ``json()`` returns ``payload``."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value=payload)
    return response


class _FakeStreamResponse:
    """Mimics an httpx streaming response for testing SSE handlers."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self.raise_for_status = MagicMock()

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


class _FakeStreamCM:
    """Async context manager yielding a :class:`_FakeStreamResponse`."""

    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _attach_stream(model: object, lines: list[str]) -> MagicMock:
    """Replace ``model._client`` with one whose ``stream`` yields ``lines``."""
    response = _FakeStreamResponse(lines)
    cm = _FakeStreamCM(response)
    client = MagicMock()
    client.stream = MagicMock(return_value=cm)
    model._client = client  # type: ignore[attr-defined]
    return client


# ---------------------------------------------------------------------------
# OpenAIModel
# ---------------------------------------------------------------------------


def test_openai_build_payload_basic():
    model = OpenAIModel(api_key="sk-test", model="gpt-4o-mini")
    messages = [Message(role="user", content="hi")]
    payload = model._build_payload(messages, None)
    assert payload["model"] == "gpt-4o-mini"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert "tools" not in payload


def test_openai_build_payload_with_tools():
    model = OpenAIModel(api_key="sk-test")
    messages = [Message(role="user", content="hi")]
    tools = [ToolSchema(name="shell", description="run", parameters={"type": "object"})]
    payload = model._build_payload(messages, tools)
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "run",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_openai_parse_response_text_and_tool_calls():
    model = OpenAIModel(api_key="sk-test")
    data = {
        "choices": [
            {
                "message": {
                    "content": "Hello",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "shell",
                                "arguments": '{"command": "ls"}',
                            }
                        }
                    ],
                }
            }
        ]
    }
    resp = model._parse_response(data)
    assert resp.content == "Hello"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "shell"
    assert resp.tool_calls[0].arguments == {"command": "ls"}


def test_openai_parse_response_empty_choices():
    model = OpenAIModel(api_key="sk-test")
    resp = model._parse_response({})
    assert resp.content == ""
    assert resp.tool_calls == []


def test_openai_parse_response_no_message():
    model = OpenAIModel(api_key="sk-test")
    resp = model._parse_response({"choices": [{}]})
    assert resp.content == ""
    assert resp.tool_calls == []


def test_openai_parse_arguments_string_json():
    assert OpenAIModel._parse_arguments('{"x": 1}') == {"x": 1}


def test_openai_parse_arguments_dict():
    assert OpenAIModel._parse_arguments({"x": 1}) == {"x": 1}


def test_openai_parse_arguments_invalid_string():
    assert OpenAIModel._parse_arguments("not json") == {"_raw": "not json"}


def test_openai_parse_arguments_non_dict_json():
    assert OpenAIModel._parse_arguments("[1, 2, 3]") == {"_raw": "[1, 2, 3]"}


def test_openai_parse_arguments_none():
    assert OpenAIModel._parse_arguments(None) == {}


async def test_openai_chat_calls_request_with_retry():
    model = OpenAIModel(
        api_key="sk-test", base_url="https://api.openai.com/v1", model="gpt-4o-mini"
    )
    model._client = MagicMock()  # avoid creating a real httpx client
    fake = _ok_response({"choices": [{"message": {"content": "Hi"}}]})
    with patch(
        "open_agent.models.openai_provider.request_with_retry",
        new=AsyncMock(return_value=fake),
    ) as mock_req:
        result = await model.chat([Message(role="user", content="hello")])
    assert result.content == "Hi"
    mock_req.assert_awaited_once()
    args, kwargs = mock_req.call_args
    assert args[1] == "POST"
    assert args[2] == "https://api.openai.com/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
    assert kwargs["json"]["model"] == "gpt-4o-mini"
    assert kwargs["json"]["messages"] == [{"role": "user", "content": "hello"}]


async def test_openai_stream_chat_yields_chunks():
    model = OpenAIModel(api_key="sk-test")
    lines = [
        'data: {"choices": [{"delta": {"content": "Hello"}}]}',
        'data: {"choices": [{"delta": {"content": " world"}}]}',
        "data: [DONE]",
    ]
    _attach_stream(model, lines)
    chunks = [
        chunk
        async for chunk in model.stream_chat([Message(role="user", content="hi")])
    ]
    assert chunks == ["Hello", " world"]


async def test_openai_aclose_closes_client():
    model = OpenAIModel(api_key="sk-test")
    fake_client = MagicMock()
    fake_client.aclose = AsyncMock()
    model._client = fake_client
    await model.aclose()
    fake_client.aclose.assert_awaited_once()
    assert model._client is None


async def test_openai_aclose_when_no_client():
    model = OpenAIModel(api_key="sk-test")
    await model.aclose()
    assert model._client is None


# ---------------------------------------------------------------------------
# AnthropicModel
# ---------------------------------------------------------------------------


def test_anthropic_build_payload_hoists_system():
    model = AnthropicModel(api_key="sk-ant")
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="hi"),
    ]
    payload = model._build_payload(messages, None)
    assert payload["system"] == "You are helpful."
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_build_payload_merges_consecutive_same_role():
    model = AnthropicModel(api_key="sk-ant")
    messages = [
        Message(role="user", content="part1"),
        Message(role="user", content="part2"),
        Message(role="assistant", content="ok"),
        Message(role="user", content="part3"),
    ]
    payload = model._build_payload(messages, None)
    assert payload["messages"] == [
        {"role": "user", "content": "part1\npart2"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "part3"},
    ]


def test_anthropic_build_payload_includes_max_tokens():
    model = AnthropicModel(api_key="sk-ant", max_tokens=512)
    messages = [Message(role="user", content="hi")]
    payload = model._build_payload(messages, None)
    assert payload["max_tokens"] == 512


def test_anthropic_build_payload_with_tools():
    model = AnthropicModel(api_key="sk-ant")
    messages = [Message(role="user", content="hi")]
    tools = [ToolSchema(name="shell", description="run", parameters={"type": "object"})]
    payload = model._build_payload(messages, tools)
    assert payload["tools"] == [
        {
            "name": "shell",
            "description": "run",
            "input_schema": {"type": "object"},
        }
    ]


def test_anthropic_build_payload_multiple_system_joined():
    model = AnthropicModel(api_key="sk-ant")
    messages = [
        Message(role="system", content="rule1"),
        Message(role="system", content="rule2"),
        Message(role="user", content="hi"),
    ]
    payload = model._build_payload(messages, None)
    assert payload["system"] == "rule1\n\nrule2"


def test_anthropic_parse_response_text_and_tool_use():
    model = AnthropicModel(api_key="sk-ant")
    data = {
        "content": [
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "name": "shell", "input": {"command": "ls"}},
        ]
    }
    resp = model._parse_response(data)
    assert resp.content == "Hello"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "shell"
    assert resp.tool_calls[0].arguments == {"command": "ls"}


def test_anthropic_parse_response_empty():
    model = AnthropicModel(api_key="sk-ant")
    resp = model._parse_response({})
    assert resp.content == ""
    assert resp.tool_calls == []


# ---------------------------------------------------------------------------
# OllamaModel
# ---------------------------------------------------------------------------


def test_ollama_build_payload_basic():
    model = OllamaModel(model="llama3.1")
    messages = [Message(role="user", content="hi")]
    payload = model._build_payload(messages, None)
    assert payload["model"] == "llama3.1"
    assert payload["stream"] is False
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert "tools" not in payload


def test_ollama_build_payload_with_tools():
    model = OllamaModel()
    messages = [Message(role="user", content="hi")]
    tools = [ToolSchema(name="shell", description="run", parameters={"type": "object"})]
    payload = model._build_payload(messages, tools)
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "run",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_ollama_parse_response_text_and_tool_calls():
    model = OllamaModel()
    data = {
        "message": {
            "content": "Hello",
            "tool_calls": [
                {
                    "function": {
                        "name": "shell",
                        "arguments": '{"command": "ls"}',
                    }
                }
            ],
        }
    }
    resp = model._parse_response(data)
    assert resp.content == "Hello"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "shell"
    assert resp.tool_calls[0].arguments == {"command": "ls"}


def test_ollama_parse_response_empty():
    model = OllamaModel()
    resp = model._parse_response({})
    assert resp.content == ""
    assert resp.tool_calls == []


async def test_ollama_headers_passed_through():
    model = OllamaModel(headers={"Authorization": "Bearer secret"})
    model._client = MagicMock()
    fake = _ok_response({"message": {"content": "ok"}})
    with patch(
        "open_agent.models.ollama_provider.request_with_retry",
        new=AsyncMock(return_value=fake),
    ) as mock_req:
        result = await model.chat([Message(role="user", content="hi")])
    assert result.content == "ok"
    _, kwargs = mock_req.call_args
    assert kwargs["headers"] == {"Authorization": "Bearer secret"}


async def test_ollama_no_headers_passes_none():
    model = OllamaModel()
    model._client = MagicMock()
    fake = _ok_response({"message": {"content": "ok"}})
    with patch(
        "open_agent.models.ollama_provider.request_with_retry",
        new=AsyncMock(return_value=fake),
    ) as mock_req:
        await model.chat([Message(role="user", content="hi")])
    _, kwargs = mock_req.call_args
    assert kwargs["headers"] is None
