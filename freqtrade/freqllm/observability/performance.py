"""
Performance tracker - records trade results and generates feedback text for LLM context.

Backed by FreqLLMDatabase (SQLite) for persistence across bot restarts.
The in-memory deque serves as a read-through cache for fast access.
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.exc import SQLAlchemyError


logger = logging.getLogger(__name__)

# Maximum number of trade records to keep per pair in memory cache
MAX_HISTORY_PER_PAIR = 200
_TRACKER_ERRORS = (AttributeError, KeyError, TypeError, ValueError, RuntimeError, SQLAlchemyError)


@dataclass(slots=True)
class TradeRecord:
    """Represent the result of a single closed trade."""

    pair: str
    profit_ratio: float
    duration_hours: float
    entry_reason: str
    exit_reason: str
    llm_reason: str = field(default="")
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        """Set a timezone-aware timestamp when one is not supplied."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)

    @property
    def is_win(self) -> bool:
        """Return whether this trade closed at a profit."""
        return self.profit_ratio > 0


class PerformanceTracker:
    """Track historical trade performance on a per-pair basis."""

    def __init__(
        self,
        win_rate_threshold: float = 0.4,
        feedback_trades: int = 10,
        database=None,
    ):
        """Initialize thresholds, cache, and optional persistence."""
        self.win_rate_threshold = win_rate_threshold
        self.feedback_trades = feedback_trades
        self._db = database
        self._histories: dict[str, deque[TradeRecord]] = {}
        self._lock = threading.Lock()
        self._loaded_pairs: set[str] = set()
        logger.info(
            "PerformanceTracker initialised: win_rate_threshold=%s, "
            "feedback_trades=%s, db_backed=%s",
            win_rate_threshold,
            feedback_trades,
            "yes" if database else "no (in-memory only)",
        )

    def _ensure_loaded(self, pair: str) -> None:
        """Load a pair's historical trades into the cache once."""
        if pair in self._loaded_pairs or self._db is None:
            return
        try:
            rows = self._db.get_trade_history(pair, limit=MAX_HISTORY_PER_PAIR)
            if rows:
                history: deque[TradeRecord] = deque(maxlen=MAX_HISTORY_PER_PAIR)
                for row in rows:
                    timestamp = row.get("timestamp")
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.fromisoformat(timestamp)
                        except (TypeError, ValueError):
                            timestamp = None
                    history.append(
                        TradeRecord(
                            pair=row["pair"],
                            profit_ratio=row["profit_ratio"],
                            duration_hours=row["duration_hours"],
                            entry_reason=row.get("entry_reason", ""),
                            exit_reason=row.get("exit_reason", ""),
                            llm_reason=row.get("llm_reason", ""),
                            timestamp=timestamp,
                        )
                    )
                self._histories[pair] = history
                logger.info("[%s] Loaded %d historical trades from database.", pair, len(history))
        except _TRACKER_ERRORS as exc:
            logger.warning("[%s] Failed to load trade history from DB: %s", pair, exc)
            return
        self._loaded_pairs.add(pair)

    def record_trade(self, record: TradeRecord | None = None, **trade_data: Any) -> None:
        """Record a closed trade in memory and optional persistent storage."""
        if record is None:
            record = TradeRecord(**trade_data)
        elif trade_data:
            raise ValueError("trade_data cannot be supplied with a TradeRecord")

        with self._lock:
            self._ensure_loaded(record.pair)
            history = self._histories.setdefault(record.pair, deque(maxlen=MAX_HISTORY_PER_PAIR))
            history.append(record)

        if self._db is not None:
            threading.Thread(target=self._persist_trade, args=(record,), daemon=True).start()

        logger.debug(
            "[%s] Trade recorded: profit=%.4f, duration=%.1fh, exit=%s",
            record.pair,
            record.profit_ratio,
            record.duration_hours,
            record.exit_reason,
        )

    def _persist_trade(self, record: TradeRecord) -> None:
        """Write a trade record to the database in a background thread."""
        try:
            self._db.record_trade(
                pair=record.pair,
                profit_ratio=record.profit_ratio,
                duration_hours=record.duration_hours,
                entry_reason=record.entry_reason,
                exit_reason=record.exit_reason,
                llm_reason=record.llm_reason,
            )
        except _TRACKER_ERRORS as exc:
            logger.warning("[%s] Async trade persist failed: %s", record.pair, exc)

    def get_stats(self, pair: str) -> dict[str, Any]:
        """Compute performance statistics for a trading pair."""
        with self._lock:
            self._ensure_loaded(pair)
            history = list(self._histories.get(pair, []))

        if not history:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "max_consecutive_losses": 0,
            }

        total = len(history)
        wins = sum(1 for record in history if record.is_win)
        return {
            "total_trades": total,
            "win_rate": wins / total,
            "avg_profit": sum(record.profit_ratio for record in history) / total,
            "max_consecutive_losses": self._max_consecutive_losses(history),
        }

    def get_feedback_text(self, pair: str, n: int | None = None) -> str:
        """Generate recent performance feedback for the LLM prompt."""
        count = self.feedback_trades if n is None else n
        with self._lock:
            self._ensure_loaded(pair)
            history = list(self._histories.get(pair, []))

        if not history:
            return f"## Historical Performance - {pair}\nNo completed trades recorded yet.\n"

        recent = history[-count:]
        stats = self.get_stats(pair)
        lines = [
            f"## Historical Performance - {pair}",
            f"- Total trades: {stats['total_trades']}",
            f"- Win rate: {stats['win_rate']:.1%}",
            f"- Average profit: {stats['avg_profit']:.2%}",
            f"- Max consecutive losses: {stats['max_consecutive_losses']}",
            "",
            f"### Last {len(recent)} Trade(s)",
        ]

        for index, record in enumerate(recent, 1):
            result = (
                f"+{record.profit_ratio:.2%}" if record.is_win else f"{record.profit_ratio:.2%}"
            )
            timestamp = record.timestamp or datetime.now(UTC)
            lines.append(
                f"{index}. [{timestamp.strftime('%Y-%m-%d %H:%M')}] "
                f"{'WIN' if record.is_win else 'LOSS'} {result} "
                f"({record.duration_hours:.1f}h) | exit: {record.exit_reason}"
            )
            if record.llm_reason:
                reason_preview = record.llm_reason[:120].replace("\n", " ")
                lines.append(f"   LLM reason: {reason_preview}...")

        lines.append("")
        win_rate = stats["win_rate"]
        if win_rate < self.win_rate_threshold:
            lines.append(
                f"⚠️ **Risk Warning**: Win rate ({win_rate:.1%}) is below the "
                f"threshold ({self.win_rate_threshold:.1%}). "
                "Please apply stricter entry criteria and tighter risk management."
            )
        elif win_rate >= 0.6:
            lines.append(
                f"✅ **Good Performance**: Win rate is {win_rate:.1%}. "
                "Current strategy is performing well; maintain discipline."
            )
        lines.append("")
        return "\n".join(lines)

    def get_global_numeric_stats(self) -> dict[str, float]:
        """Aggregate numeric performance stats across all tracked pairs."""
        default = {
            "total_trades": 0,
            "win_rate": 0.5,
            "avg_win_ratio": 0.02,
            "avg_loss_ratio": 0.02,
        }

        try:
            if self._db is not None:
                try:
                    database_pairs = self._db.get_all_pairs_stats()
                    with self._lock:
                        for pair in database_pairs:
                            self._ensure_loaded(pair)
                except _TRACKER_ERRORS as exc:
                    logger.debug("get_global_numeric_stats: failed to enumerate DB pairs: %s", exc)

            with self._lock:
                all_records = [record for history in self._histories.values() for record in history]
            if not all_records:
                return default

            total_trades = len(all_records)
            win_profits = [record.profit_ratio for record in all_records if record.is_win]
            loss_profits = [abs(record.profit_ratio) for record in all_records if not record.is_win]
            return {
                "total_trades": total_trades,
                "win_rate": len(win_profits) / total_trades,
                "avg_win_ratio": (sum(win_profits) / len(win_profits) if win_profits else 0.02),
                "avg_loss_ratio": (sum(loss_profits) / len(loss_profits) if loss_profits else 0.02),
            }
        except _TRACKER_ERRORS as exc:
            logger.debug("Failed to compute global numeric stats: %s", exc)
            return default

    @staticmethod
    def _max_consecutive_losses(history: list[TradeRecord]) -> int:
        """Calculate the maximum number of consecutive losing trades."""
        max_loss = 0
        current = 0
        for r in history:
            if not r.is_win:
                current += 1
                max_loss = max(max_loss, current)
            else:
                current = 0
        return max_loss
