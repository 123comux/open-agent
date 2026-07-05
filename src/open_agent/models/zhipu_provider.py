"""Zhipu AI (智谱) chat completion provider.

Zhipu AI exposes an OpenAI-compatible API at
``https://open.bigmodel.cn/api/paas/v4``. This provider subclasses
:class:`OpenAIModel` with Zhipu-specific defaults so users can simply set
``OPEN_AGENT_MODEL_PROVIDER=zhipu`` without manually configuring the base URL.

Free models include ``glm-4-flash`` and ``glm-4.7-flash``; see
https://open.bigmodel.cn/dev/api for the full model list.
"""
from __future__ import annotations

from open_agent.models.openai_provider import OpenAIModel

ZHIPU_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
ZHIPU_DEFAULT_MODEL = "glm-4-flash"


class ZhipuModel(OpenAIModel):
    """Zhipu AI provider (GLM-4-Flash, GLM-4.7-Flash, etc.).

    A thin specialization of :class:`OpenAIModel` that defaults to Zhipu's
    OpenAI-compatible endpoint and the free ``glm-4-flash`` model. All request
    / response parsing, streaming, and retry logic is inherited from
    :class:`OpenAIModel` since Zhipu's API is wire-compatible with OpenAI's
    ``/v1/chat/completions`` contract.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = ZHIPU_DEFAULT_BASE_URL,
        model: str = ZHIPU_DEFAULT_MODEL,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )
