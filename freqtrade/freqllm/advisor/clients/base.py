"""LLM client abstract base class and response dataclass."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from operator import methodcaller


logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Represents a single LLM API call response."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    finish_reason: str = ""


class LLMClient(ABC):
    """
    Abstract base class for all LLM provider clients.
    All concrete implementations must inherit from this class.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0  # seconds between retries (multiplied by attempt number)

    def __init__(self, api_key: str, model: str, api_base_url: str = ""):
        self.api_key = api_key
        self.model = model
        self.api_base_url = api_base_url

    @abstractmethod
    def _do_chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Execute the actual LLM API call. Must be implemented by subclasses."""
        raise NotImplementedError

    def chat(self, messages: list[dict[str, str]], pair: str = "") -> LLMResponse:
        """
        Call the LLM API with automatic retry (up to MAX_RETRIES attempts).

        :param messages: List of message dicts with 'role' and 'content' keys.
        :param pair: Trading pair label used for logging.
        :return: LLMResponse on success.
        :raises RuntimeError: If all retry attempts fail.
        """
        last_error: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.debug(
                    "[%s] Calling LLM (attempt %s), model=%s, messages=%s",
                    pair,
                    attempt,
                    self.model,
                    len(messages),
                )
                resp = self._do_chat(messages)
                logger.debug("[%s] LLM call succeeded, tokens=%s", pair, resp.total_tokens)
                return resp
            except (OSError, RuntimeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "[%s] LLM call failed (attempt %s/%s): %s",
                    pair,
                    attempt,
                    self.MAX_RETRIES,
                    exc,
                )
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY * attempt)
        logger.error(
            "[%s] LLM call failed after %s retries. Last error: %s",
            pair,
            self.MAX_RETRIES,
            last_error,
        )
        raise RuntimeError(
            f"LLM API call failed after {self.MAX_RETRIES} retries: {last_error}"
        ) from last_error

    def close(self) -> None:
        """Release provider SDK resources when the underlying client supports closing."""
        provider_client = getattr(self, "_client", None)
        close = getattr(provider_client, "close", None)
        if callable(close):
            methodcaller("close")(provider_client)

    @property
    def provider_name(self) -> str:
        """Return the provider name string."""
        return self.__class__.__name__
