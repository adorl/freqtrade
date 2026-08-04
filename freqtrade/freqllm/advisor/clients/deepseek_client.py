"""DeepSeek client - DeepSeek API via OpenAI-compatible endpoint."""

import logging

from freqtrade.freqllm.advisor.clients.openai_client import OpenAIClient


logger = logging.getLogger(__name__)

# DeepSeek official OpenAI-compatible API endpoint
DEEPSEEK_BASE = "https://api.deepseek.com/v1"


class DeepSeekClient(OpenAIClient):
    """
    Client for DeepSeek model series (DeepSeek-V3, DeepSeek-R1, etc.).
    Uses the OpenAI-compatible Chat Completions API provided by DeepSeek.
    """

    def __init__(self, api_key: str, model: str = "deepseek-chat", api_base_url: str = ""):
        if not api_base_url:
            api_base_url = DEEPSEEK_BASE
        # Pass to OpenAIClient with the DeepSeek API base URL
        super().__init__(
            api_key=api_key,
            model=model,
            api_base_url=api_base_url,
        )

    @property
    def provider_name(self) -> str:
        return "deepseek"


__all__ = ["DeepSeekClient"]
