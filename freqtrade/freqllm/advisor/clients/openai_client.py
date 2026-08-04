"""OpenAI-compatible client - supports OpenAI and custom endpoints."""

import logging
from importlib import import_module

from freqtrade.freqllm.advisor.clients.base import LLMClient, LLMResponse


logger = logging.getLogger(__name__)

OPENAI_BASE = "https://api.openai.com/v1"


class OpenAIClient(LLMClient):
    """
    OpenAI-compatible client supporting GPT series and any endpoint
    that implements the OpenAI Chat Completions API format.
    """

    def __init__(self, api_key: str, model: str, api_base_url: str = ""):
        if not api_base_url:
            api_base_url = OPENAI_BASE
        super().__init__(api_key=api_key, model=model, api_base_url=api_base_url)
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the underlying OpenAI SDK client."""
        try:
            openai = import_module("openai")
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base_url)
            logger.info(
                "OpenAI client initialized, model=%s, base_url=%s",
                self.model,
                self.api_base_url,
            )
        except ImportError as exc:
            raise ImportError("openai package is required: pip install openai") from exc

    def _do_chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
        )
        choice = resp.choices[0]
        usage = resp.usage
        return LLMResponse(
            content=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            model=resp.model or self.model,
            finish_reason=choice.finish_reason or "",
        )

    @property
    def provider_name(self) -> str:
        return "openai"
