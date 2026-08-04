"""Adaptive repository backed by the unified FreqLLM database lifecycle."""

from typing import TYPE_CHECKING, Any

from sqlalchemy import text


if TYPE_CHECKING:
    from freqtrade.freqllm.persistence.models import FreqLLMDatabase


class AdaptiveRepository:
    """Execute parameter-bound adaptive queries through a shared database."""

    def __init__(self, database: "FreqLLMDatabase") -> None:
        self._database = database

    def query(self, statement: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run a parameter-bound read and detach its rows from the session."""
        session = self._database.session
        try:
            rows = session.execute(text(statement), params or {}).mappings().all()
            return [dict(row) for row in rows]
        finally:
            self._database.remove_session()

    def execute(self, statement: str, params: dict[str, Any] | None = None) -> None:
        """Run a parameter-bound write in a committed session transaction."""
        session = self._database.session
        try:
            with session.begin():
                session.execute(text(statement), params or {})
        finally:
            self._database.remove_session()
