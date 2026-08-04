"""
Conversation context manager - maintains per-pair independent dialogue history.
"""

import logging
import threading


logger = logging.getLogger(__name__)


class ConversationContextManager:
    """
    Manages conversation history on a per-trading-pair basis.
    Supports multi-turn dialogue mode and single-turn (stateless) mode.
    """

    def __init__(self, context_enabled: bool = True, context_max_turns: int = 10):
        """
        :param context_enabled: Enable multi-turn context (True) or single-turn mode (False).
        :param context_max_turns: Maximum number of turns to retain per pair.
                                  Each turn = 1 user message + 1 assistant message.
        """
        self.context_enabled = context_enabled
        self.context_max_turns = context_max_turns
        # key: pair string, value: list of {"role": "user"|"assistant", "content": "..."}
        self._histories: dict[str, list[dict[str, str]]] = {}
        self._lock = threading.Lock()
        logger.info(
            "ConversationContextManager initialized: context_enabled=%s, context_max_turns=%s",
            context_enabled,
            context_max_turns,
        )

    def add_turn(self, pair: str, user_msg: str, assistant_msg: str) -> None:
        """
        Append one dialogue turn (user + assistant) to the pair's history.

        :param pair: Trading pair identifier.
        :param user_msg: The user-side message content.
        :param assistant_msg: The assistant reply content.
        """
        if not self.context_enabled:
            return
        with self._lock:
            if pair not in self._histories:
                self._histories[pair] = []
            history = self._histories[pair]
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_msg})
            # Truncate oldest turns when limit is exceeded (each turn = 2 messages)
            max_messages = self.context_max_turns * 2
            if len(history) > max_messages:
                removed = len(history) - max_messages
                self._histories[pair] = history[removed:]
                logger.debug(
                    "[%s] History truncated: removed %s turn(s), retained %s turn(s)",
                    pair,
                    removed // 2,
                    len(self._histories[pair]) // 2,
                )

    def get_history(self, pair: str) -> list[dict[str, str]]:
        """
        Return the conversation history for a pair.

        :param pair: Trading pair identifier.
        :return: List of message dicts; always empty when context_enabled=False.
        """
        if not self.context_enabled:
            return []
        with self._lock:
            return list(self._histories.get(pair, []))

    def clear(self, pair: str | None = None) -> None:
        """
        Clear conversation history.

        :param pair: Specific pair to clear; clears all pairs when None.
        """
        with self._lock:
            if pair is None:
                self._histories.clear()
                logger.info("Cleared conversation history for all pairs.")
            elif pair in self._histories:
                del self._histories[pair]
                logger.info("[%s] Conversation history cleared.", pair)

    def get_all_pairs(self) -> list[str]:
        """Return a list of all pairs that have conversation history."""
        with self._lock:
            return list(self._histories.keys())

    def get_stats(self) -> dict[str, int]:
        """Return a dict mapping each pair to its current turn count."""
        with self._lock:
            return {pair: len(h) // 2 for pair, h in self._histories.items()}
