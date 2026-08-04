"""Zai (z.ai / Zhipu AI) client - GLM-5 series via OpenAI-compatible API."""

import logging

from freqtrade.freqllm.advisor.clients.openai_client import OpenAIClient


logger = logging.getLogger(__name__)

# Zhipu AI (z.ai) official OpenAI-compatible API endpoint
ZAI_BASE = "https://api.z.ai/api/paas/v4"


class ZaiClient(OpenAIClient):
    """
    Client for Zhipu AI (z.ai) GLM model series (GLM-5, GLM-5-Turbo, etc.).
    Uses the OpenAI-compatible Chat Completions API provided by bigmodel.cn.
    """

    def __init__(self, api_key: str, model: str = "glm-5", api_base_url: str = ""):
        if not api_base_url:
            api_base_url = ZAI_BASE
        # Pass to OpenAIClient with the Zhipu API base URL
        super().__init__(
            api_key=api_key,
            model=model,
            api_base_url=api_base_url,
        )

    @property
    def provider_name(self) -> str:
        return "zai"
