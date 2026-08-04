"""Anthropic Claude client implementation."""

import logging
from importlib import import_module

from freqtrade.freqllm.advisor.clients.base import LLMClient, LLMResponse


logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """
    Anthropic Claude client supporting Claude 3 and later model series.
    System messages are extracted and passed via the dedicated 'system' parameter.
    """

    def __init__(self, api_key: str, model: str, api_base_url: str = ""):
        if not api_base_url:
            api_base_url = "https://api.anthropic.com"
        super().__init__(api_key=api_key, model=model, api_base_url=api_base_url)
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Anthropic SDK client."""
        try:
            anthropic = import_module("anthropic")
            self._client = anthropic.Anthropic(api_key=self.api_key)
            logger.info("Anthropic client initialized, model=%s", self.model)
        except ImportError as exc:
            raise ImportError("anthropic package is required: pip install anthropic") from exc

    def _do_chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        # Separate system message from user/assistant messages
        system_content = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": user_messages,
            "temperature": 0.3,
        }
        if system_content:
            kwargs["system"] = system_content

        resp = self._client.messages.create(**kwargs)
        content = resp.content[0].text if resp.content and hasattr(resp.content[0], "text") else ""
        usage = resp.usage
        pt = usage.input_tokens if usage else 0
        ct = usage.output_tokens if usage else 0
        return LLMResponse(
            content=content,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
            model=resp.model or self.model,
            finish_reason=resp.stop_reason or "",
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"
