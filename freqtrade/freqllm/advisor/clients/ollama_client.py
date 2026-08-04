"""Ollama local model client using the OpenAI-compatible API."""

import logging
from importlib import import_module

from freqtrade.freqllm.advisor.clients.base import LLMClient, LLMResponse


logger = logging.getLogger(__name__)

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaClient(LLMClient):
    """
    Client for locally-hosted models served by Ollama.
    Uses the OpenAI-compatible /v1 endpoint exposed by Ollama.
    """

    def __init__(self, api_key: str = "", model: str = "llama3", api_base_url: str = ""):
        if not api_base_url:
            api_base_url = OLLAMA_DEFAULT_BASE_URL
        super().__init__(api_key=api_key, model=model, api_base_url=api_base_url)
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the OpenAI SDK client pointed at the local Ollama endpoint."""
        try:
            openai = import_module("openai")
            self._client = openai.OpenAI(
                api_key=self.api_key or "ollama",  # Ollama ignores the key value
                base_url=f"{self.api_base_url}/v1",
            )
            logger.info(
                "Ollama client initialized, model=%s, endpoint=%s",
                self.model,
                self.api_base_url,
            )
        except ImportError as exc:
            raise ImportError(
                "openai package is required for Ollama client: pip install openai"
            ) from exc

    def _do_chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            model=response.model or self.model,
            finish_reason=choice.finish_reason or "",
        )

    @property
    def provider_name(self) -> str:
        return "ollama"
