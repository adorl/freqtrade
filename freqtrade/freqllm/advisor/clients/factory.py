"""
LLM client factory - instantiates the correct client based on config provider field.
"""

import logging

from freqtrade.freqllm.advisor.clients.anthropic_client import AnthropicClient
from freqtrade.freqllm.advisor.clients.base import LLMClient
from freqtrade.freqllm.advisor.clients.deepseek_client import DeepSeekClient
from freqtrade.freqllm.advisor.clients.ollama_client import OllamaClient
from freqtrade.freqllm.advisor.clients.openai_client import OpenAIClient
from freqtrade.freqllm.advisor.clients.zai_client import ZaiClient
from freqtrade.freqllm.configuration import LLMStrategyConfig


logger = logging.getLogger(__name__)


class LLMClientFactory:
    """
    Factory class for LLM clients.
    Creates the appropriate provider-specific client instance from a config object.
    """

    PROVIDERS = ("openai", "deepseek", "anthropic", "ollama", "zai")

    @classmethod
    def supported_providers(cls) -> tuple[str, ...]:
        """Return provider names supported by this factory."""
        return cls.PROVIDERS

    @staticmethod
    def create(config: LLMStrategyConfig) -> LLMClient:
        """
        Instantiate an LLM client based on the provider field in config.

        :param config: LLMStrategyConfig instance.
        :return: Concrete LLMClient for the configured provider.
        :raises ValueError: If the provider is not supported.
        """
        provider = config.llm.provider.lower()
        logger.info("Creating LLM client: provider=%s, model=%s", provider, config.llm.model)

        if provider == "openai":
            return OpenAIClient(
                api_key=config.llm.api_key,
                model=config.llm.model,
                api_base_url=config.llm.api_base_url,
            )
        if provider == "deepseek":
            return DeepSeekClient(
                api_key=config.llm.api_key,
                model=config.llm.model,
                api_base_url=config.llm.api_base_url,
            )
        if provider == "anthropic":
            return AnthropicClient(
                api_key=config.llm.api_key,
                model=config.llm.model,
                api_base_url=config.llm.api_base_url,
            )
        if provider == "ollama":
            return OllamaClient(
                api_key=config.llm.api_key,
                model=config.llm.model,
                api_base_url=config.llm.api_base_url,
            )
        if provider == "zai":
            return ZaiClient(
                api_key=config.llm.api_key,
                model=config.llm.model,
                api_base_url=config.llm.api_base_url,
            )
        supported = ", ".join(LLMClientFactory.supported_providers())
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. Supported providers: {supported}"
        )
