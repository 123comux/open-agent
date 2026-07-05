"""Unit tests for the Zhipu AI provider and config integration."""
from __future__ import annotations

import os

from open_agent.config import Settings
from open_agent.models.zhipu_provider import (
    ZHIPU_DEFAULT_BASE_URL,
    ZHIPU_DEFAULT_MODEL,
    ZhipuModel,
)


def test_zhipu_model_defaults():
    """ZhipuModel uses Zhipu endpoint and glm-4-flash by default."""
    model = ZhipuModel(api_key="test-key")
    assert model.base_url == ZHIPU_DEFAULT_BASE_URL
    assert model.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert model.model == ZHIPU_DEFAULT_MODEL
    assert model.model == "glm-4-flash"
    assert model.api_key == "test-key"


def test_zhipu_model_custom_base_url():
    """ZhipuModel accepts a custom base_url override."""
    model = ZhipuModel(api_key="k", base_url="https://custom.example.com/v1")
    assert model.base_url == "https://custom.example.com/v1"


def test_zhipu_model_custom_model():
    """ZhipuModel accepts a custom model name (e.g. glm-4.7-flash)."""
    model = ZhipuModel(api_key="k", model="glm-4.7-flash")
    assert model.model == "glm-4.7-flash"


def test_zhipu_model_inherits_openai_behavior():
    """ZhipuModel is a subclass of OpenAIModel (OpenAI-compatible wire format)."""
    from open_agent.models.openai_provider import OpenAIModel

    assert issubclass(ZhipuModel, OpenAIModel)
    model = ZhipuModel(api_key="k")
    assert isinstance(model, OpenAIModel)
    # Inherited methods exist and are callable.
    assert callable(model.chat)
    assert callable(model.stream_chat)
    assert callable(model.aclose)


def test_config_loads_zhipu_provider(monkeypatch):
    """Settings.load() accepts MODEL_PROVIDER=zhipu."""
    monkeypatch.setenv("OPEN_AGENT_MODEL_PROVIDER", "zhipu")
    monkeypatch.setenv("OPEN_AGENT_API_KEY", "zhipu-key")
    monkeypatch.setenv("OPEN_AGENT_MODEL_NAME", "glm-4.7-flash")
    settings = Settings.load()
    assert settings.model_provider == "zhipu"
    assert settings.model_name == "glm-4.7-flash"


def test_config_rejects_unknown_provider(monkeypatch):
    """Unknown provider values fall back to 'openai'."""
    monkeypatch.setenv("OPEN_AGENT_MODEL_PROVIDER", "unknown-provider")
    settings = Settings.load()
    assert settings.model_provider == "openai"


def test_zhipu_build_model_branch(monkeypatch):
    """cli._build_model routes zhipu provider to ZhipuModel."""
    from open_agent.cli import _build_model

    monkeypatch.setenv("OPEN_AGENT_MODEL_PROVIDER", "zhipu")
    monkeypatch.setenv("OPEN_AGENT_API_KEY", "zhipu-key")
    monkeypatch.setenv("OPEN_AGENT_MODEL_NAME", "glm-4-flash")
    settings = Settings.load()
    model = _build_model(settings)
    assert isinstance(model, ZhipuModel)
    assert model.base_url == ZHIPU_DEFAULT_BASE_URL
    assert model.api_key == "zhipu-key"


def test_zhipu_build_model_with_custom_base_url(monkeypatch):
    """When base_url is overridden away from OpenAI default, it is respected."""
    from open_agent.cli import _build_model

    monkeypatch.setenv("OPEN_AGENT_MODEL_PROVIDER", "zhipu")
    monkeypatch.setenv("OPEN_AGENT_API_KEY", "zhipu-key")
    monkeypatch.setenv("OPEN_AGENT_BASE_URL", "https://custom.zhipu.example/v4")
    monkeypatch.setenv("OPEN_AGENT_MODEL_NAME", "glm-4-flash")
    settings = Settings.load()
    model = _build_model(settings)
    assert isinstance(model, ZhipuModel)
    assert model.base_url == "https://custom.zhipu.example/v4"
