"""Record and persist LLM API token consumption statistics."""

import json
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError


logger = logging.getLogger(__name__)

DEFAULT_STATS_PATH = "user_data/freqllm/token_stats.json"
_PERSISTENCE_ERRORS = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    SQLAlchemyError,
)


@dataclass(frozen=True, slots=True)
class _TrackerConfig:
    """Hold immutable tracker configuration."""

    stats_path: Path
    cost_per_1k: float
    log_interval: int
    database: Any


@dataclass(slots=True)
class _TokenTotals:
    """Hold cumulative token counters."""

    calls: int = 0
    prompt: int = 0
    completion: int = 0
    total: int = 0
    cost: float = 0.0


@dataclass(frozen=True, slots=True)
class _TokenRecord:
    """Represent one validated token-usage record."""

    pair: str
    call_type: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class TokenTracker:
    """Track LLM token usage with database or JSON persistence."""

    def __init__(
        self,
        stats_path: str = DEFAULT_STATS_PATH,
        cost_per_1k: float = 0.0,
        log_interval: int = 10,
        database=None,
    ):
        """Initialize tracker configuration and load persisted totals."""
        if not stats_path:
            raise ValueError("stats_path must not be empty")
        if cost_per_1k < 0:
            raise ValueError("cost_per_1k must be non-negative")
        if log_interval <= 0:
            raise ValueError("log_interval must be positive")
        self._config = _TrackerConfig(Path(stats_path), cost_per_1k, log_interval, database)
        self._lock = threading.Lock()
        self._call_count = 0
        self._totals = _TokenTotals()
        self._pair_stats: dict[str, dict[str, Any]] = {}
        self._load()
        logger.info(
            "TokenTracker initialised: db_backed=%s, cost_per_1k=%s, log_interval=%s",
            "yes" if database else "no (JSON fallback)",
            cost_per_1k,
            log_interval,
        )

    @property
    def stats_path(self) -> Path:
        """Return the JSON fallback path."""
        return self._config.stats_path

    @property
    def cost_per_1k(self) -> float:
        """Return the configured cost per thousand tokens."""
        return self._config.cost_per_1k

    @property
    def log_interval(self) -> int:
        """Return the summary logging interval."""
        return self._config.log_interval

    @staticmethod
    def _validate_token_counts(token_counts: dict[str, int]) -> tuple[int, int, int]:
        """Validate and unpack token counts supplied by an API response."""
        expected = {"prompt_tokens", "completion_tokens", "total_tokens"}
        if set(token_counts) != expected:
            missing = expected - set(token_counts)
            unexpected = set(token_counts) - expected
            raise ValueError(
                f"Invalid token counts; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        values = tuple(token_counts[name] for name in sorted(expected))
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("token counts must be integers")
        if any(value < 0 for value in values):
            raise ValueError("token counts must be non-negative")
        return (
            token_counts["prompt_tokens"],
            token_counts["completion_tokens"],
            token_counts["total_tokens"],
        )

    def record(self, pair: str, call_type: str, **token_counts: int) -> None:
        """Record one validated LLM API call."""
        if not pair or not call_type:
            raise ValueError("pair and call_type must not be empty")
        prompt_tokens, completion_tokens, total_tokens = self._validate_token_counts(token_counts)
        cost = total_tokens / 1000.0 * self.cost_per_1k
        record = _TokenRecord(
            pair,
            call_type,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost,
        )

        with self._lock:
            self._update_totals(record)
            self._update_pair_stats(record)
            self._call_count += 1
            should_log_summary = self._call_count % self.log_interval == 0

        logger.debug(
            "[%s] Token recorded: type=%s, prompt=%d, completion=%d, total=%d, cost=%.6f",
            record.pair,
            record.call_type,
            record.prompt_tokens,
            record.completion_tokens,
            record.total_tokens,
            record.cost,
        )
        if should_log_summary:
            logger.info("%s", self.get_summary())
        self._persist_async(record)

    def _update_totals(self, record: _TokenRecord) -> None:
        """Add one record to cumulative counters while the lock is held."""
        self._totals.calls += 1
        self._totals.prompt += record.prompt_tokens
        self._totals.completion += record.completion_tokens
        self._totals.total += record.total_tokens
        self._totals.cost += record.cost

    def _update_pair_stats(self, record: _TokenRecord) -> None:
        """Add one record to its per-pair counters while the lock is held."""
        pair_stats = self._pair_stats.setdefault(
            record.pair,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
            },
        )
        pair_stats["calls"] += 1
        pair_stats["prompt_tokens"] += record.prompt_tokens
        pair_stats["completion_tokens"] += record.completion_tokens
        pair_stats["total_tokens"] += record.total_tokens
        pair_stats["cost"] += record.cost

    def get_summary(self) -> str:
        """Return a human-readable summary of cumulative token usage."""
        with self._lock:
            lines = [
                "=== LLM Token Usage Summary ===",
                f"Total calls       : {self._totals.calls}",
                f"Prompt tokens     : {self._totals.prompt}",
                f"Completion tokens : {self._totals.completion}",
                f"Total tokens      : {self._totals.total}",
            ]
            if self.cost_per_1k > 0:
                lines.append(f"Estimated cost    : ${self._totals.cost:.4f}")
            if self._pair_stats:
                lines.append("--- Per-pair breakdown ---")
                for pair, stats in sorted(self._pair_stats.items()):
                    cost_suffix = f", cost=${stats['cost']:.4f}" if self.cost_per_1k > 0 else ""
                    lines.append(
                        f"  {pair}: calls={stats['calls']}, "
                        f"tokens={stats['total_tokens']}{cost_suffix}"
                    )
            return "\n".join(lines)

    def get_stats_dict(self) -> dict[str, Any]:
        """Return the full stats as a JSON-serializable dictionary."""
        with self._lock:
            return {
                "total_calls": self._totals.calls,
                "total_prompt_tokens": self._totals.prompt,
                "total_completion_tokens": self._totals.completion,
                "total_tokens": self._totals.total,
                "total_cost": self._totals.cost,
                "pair_stats": dict(self._pair_stats),
                "last_updated": datetime.now(UTC).isoformat(),
            }

    def _load(self) -> None:
        """Load cumulative stats from the configured persistence backend."""
        if self._config.database is not None:
            self._load_from_db()
        else:
            self._load_from_json()

    def _apply_loaded_stats(self, stats: dict[str, Any]) -> None:
        """Apply persisted aggregate fields to in-memory counters."""
        self._totals = _TokenTotals(
            calls=stats.get("total_calls", 0),
            prompt=stats.get("total_prompt_tokens", 0),
            completion=stats.get("total_completion_tokens", 0),
            total=stats.get("total_tokens", 0),
            cost=stats.get("total_cost", 0.0),
        )

    def _load_from_db(self) -> None:
        """Load aggregate token statistics from the database."""
        try:
            stats = self._config.database.get_token_stats()
            self._apply_loaded_stats(stats)
            for pair, pair_stats in stats.get("pair_stats", {}).items():
                self._pair_stats[pair] = {
                    "calls": pair_stats.get("calls", 0),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": pair_stats.get("total_tokens", 0),
                    "cost": pair_stats.get("cost", 0.0),
                }
            logger.info(
                "Token stats loaded from database: total_calls=%d, total_tokens=%d",
                self._totals.calls,
                self._totals.total,
            )
        except _PERSISTENCE_ERRORS as exc:
            logger.warning("Failed to load token stats from database: %s", exc)

    def _load_from_json(self) -> None:
        """Load cumulative stats from the JSON fallback file when present."""
        if not self.stats_path.exists():
            logger.debug("Token stats file not found at '%s'; starting fresh.", self.stats_path)
            return
        try:
            with self.stats_path.open(encoding="utf-8") as stats_file:
                stats = json.load(stats_file)
            if not isinstance(stats, dict):
                raise TypeError("token stats root must be an object")
            self._apply_loaded_stats(stats)
            pair_stats = stats.get("pair_stats", {})
            if not isinstance(pair_stats, dict):
                raise TypeError("pair_stats must be an object")
            self._pair_stats = pair_stats
            logger.info(
                "Token stats loaded from '%s': total_calls=%d, total_tokens=%d",
                self.stats_path,
                self._totals.calls,
                self._totals.total,
            )
        except _PERSISTENCE_ERRORS as exc:
            logger.warning("Failed to load token stats from '%s': %s", self.stats_path, exc)

    def _persist_async(self, record: _TokenRecord) -> None:
        """Persist one token record in a background thread."""
        if self._config.database is not None:
            thread = threading.Thread(target=self._persist_to_db, args=(record,), daemon=True)
        else:
            thread = threading.Thread(target=self._save_json, daemon=True)
        thread.start()

    def _persist_to_db(self, record: _TokenRecord) -> None:
        """Write a single token-usage record to the database."""
        try:
            self._config.database.record_token_usage(
                pair=record.pair,
                call_type=record.call_type,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                total_tokens=record.total_tokens,
                cost=record.cost,
            )
        except _PERSISTENCE_ERRORS as exc:
            logger.warning("[%s] Async token persist to DB failed: %s", record.pair, exc)

    def _save_json(self) -> None:
        """Write current stats to the JSON fallback file."""
        try:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            with self.stats_path.open("w", encoding="utf-8") as stats_file:
                json.dump(self.get_stats_dict(), stats_file, indent=2, ensure_ascii=False)
        except _PERSISTENCE_ERRORS as exc:
            logger.warning("Failed to save token stats to '%s': %s", self.stats_path, exc)
