"""
FreqLLM persistence models — unified database for trade history and token usage.

All FreqLLM data (trade records, token statistics) is persisted in a single database
(supports SQLite, MySQL, MariaDB) so that the bot can survive restarts without losing
Kelly formula history or cost data.

Uses SQLAlchemy ORM, consistent with freqtrade's main persistence layer.
"""

import logging
import threading
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from sqlalchemy.sql.functions import count as sql_count

from freqtrade.freqllm.persistence.adaptive import AdaptiveRepository
from freqtrade.freqllm.persistence.base import FreqLLMModelBase


logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_URL = "sqlite:///user_data/freqllm/freqllm.db"
DATABASE_ERRORS = (KeyError, OSError, SQLAlchemyError, TypeError, ValueError)


def _utcnow() -> datetime:
    """Return current UTC time (used as column default)."""
    return datetime.now(UTC)


def _as_utc_naive(value: datetime | None) -> datetime | None:
    """Normalise datetimes to naive UTC for database storage/comparison."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _as_utc_aware(value: datetime | None) -> datetime | None:
    """Normalise datetimes to UTC-aware values for application use."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


# ORM Models


class TradeHistory(FreqLLMModelBase):
    """Persisted record of a closed trade for performance tracking and Kelly formula."""

    __tablename__ = "freqllm_trade_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(64), nullable=False, index=True)
    profit_ratio = Column(Float, nullable=False)  # e.g. 0.025 = +2.5%
    duration_hours = Column(Float, nullable=False)
    entry_reason = Column(String(128), default="")
    exit_reason = Column(String(128), default="")
    llm_reason = Column(Text, default="")  # LLM's original reasoning
    is_win = Column(Integer, nullable=False, default=0)  # 1 = win, 0 = loss
    timestamp = Column(DateTime, nullable=False, default=_utcnow)

    def __repr__(self) -> str:
        return (
            f"<TradeHistory pair={self.pair} profit={self.profit_ratio:.4f} "
            f"win={self.is_win} ts={self.timestamp}>"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-safe representation of this trade."""
        return {
            "pair": self.pair,
            "profit_ratio": self.profit_ratio,
            "duration_hours": self.duration_hours,
            "entry_reason": self.entry_reason or "",
            "exit_reason": self.exit_reason or "",
            "llm_reason": self.llm_reason or "",
            "is_win": bool(self.is_win),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class TokenUsage(FreqLLMModelBase):
    """Persisted record of LLM API token consumption."""

    __tablename__ = "freqllm_token_usage"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(64), nullable=False, index=True)
    call_type = Column(String(32), default="analysis")  # "analysis", "retry", etc.
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    timestamp = Column(DateTime, nullable=False, default=_utcnow)

    def __repr__(self) -> str:
        return (
            f"<TokenUsage pair={self.pair} type={self.call_type} "
            f"tokens={self.total_tokens} ts={self.timestamp}>"
        )


class LongShortRatioHistory(FreqLLMModelBase):
    """Persisted long/short ratio data point for historical analysis."""

    __tablename__ = "freqllm_ls_ratio_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(64), nullable=False, index=True)
    ratio_type = Column(
        String(32), nullable=False
    )  # top_trader_account, top_trader_position, global_account
    long_short_ratio = Column(Float, nullable=False)
    period = Column(String(8), default="5m")
    source_timestamp = Column(DateTime, nullable=True)  # original timestamp from exchange
    timestamp = Column(DateTime, nullable=False, default=_utcnow, index=True)

    def __repr__(self) -> str:
        return (
            f"<LSRatioHistory pair={self.pair} type={self.ratio_type} "
            f"ratio={self.long_short_ratio:.4f} ts={self.timestamp}>"
        )


class MarketFeatureHistory(FreqLLMModelBase):
    """Persisted external market microstructure snapshot for feature replay."""

    __tablename__ = "freqllm_market_feature_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(64), nullable=False, index=True)
    feature_type = Column(String(64), nullable=False, index=True)
    period = Column(String(16), nullable=False, default="snapshot")
    feature_key = Column(String(128), nullable=False, default="")

    funding_rate = Column(Float, nullable=True)
    basis = Column(Float, nullable=True)
    basis_pct = Column(Float, nullable=True)
    mark_price = Column(Float, nullable=True)
    index_price = Column(Float, nullable=True)
    next_funding_time = Column(DateTime, nullable=True)

    open_interest = Column(Float, nullable=True)
    open_interest_value = Column(Float, nullable=True)

    buy_vol = Column(Float, nullable=True)
    sell_vol = Column(Float, nullable=True)
    buy_sell_ratio = Column(Float, nullable=True)
    buy_sell_imbalance = Column(Float, nullable=True)
    event_count = Column(Integer, nullable=True)

    max_leverage = Column(Float, nullable=True)

    spread = Column(Float, nullable=True)
    spread_pct = Column(Float, nullable=True)
    bid_ask_imbalance = Column(Float, nullable=True)
    bid_total_qty = Column(Float, nullable=True)
    ask_total_qty = Column(Float, nullable=True)
    bid_depth_value = Column(Float, nullable=True)
    ask_depth_value = Column(Float, nullable=True)
    price_impact_buy_pct = Column(Float, nullable=True)
    price_impact_sell_pct = Column(Float, nullable=True)

    liquidation_side = Column(String(16), nullable=True)
    liquidation_price = Column(Float, nullable=True)
    liquidation_qty = Column(Float, nullable=True)

    raw_payload = Column(Text, default="")
    source_timestamp = Column(DateTime, nullable=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=_utcnow, index=True)

    def __repr__(self) -> str:
        return (
            f"<MarketFeatureHistory pair={self.pair} type={self.feature_type} "
            f"period={self.period} ts={self.source_timestamp or self.timestamp}>"
        )


class AdaptiveSignal(FreqLLMModelBase):
    """Canonical attributed signal stored for adaptive replay."""

    __tablename__ = "adaptive_signals"
    __table_args__ = (
        UniqueConstraint("pair", "reference_time", "side", name="uq_adaptive_signal_identity"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(64), nullable=False, index=True)
    reference_time = Column(String(48), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    decision = Column(String(24), nullable=False, default="")
    reason = Column(String(96), nullable=False, default="")
    payload_json = Column(Text, nullable=False)
    matured = Column(Integer, nullable=False, default=0, index=True)
    created_at = Column(String(48), nullable=False)
    updated_at = Column(String(48), nullable=False)


class AdaptiveTrade(FreqLLMModelBase):
    """Closed trade payload used by adaptive analysis."""

    __tablename__ = "adaptive_trades"
    __table_args__ = (
        UniqueConstraint(
            "trade_id", "close_time", "exit_reason", name="uq_adaptive_trade_identity"
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(96), nullable=False)
    pair = Column(String(64), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    open_time = Column(String(48), nullable=False)
    close_time = Column(String(48), nullable=False, index=True)
    exit_reason = Column(String(96), nullable=False, default="")
    profit_ratio = Column(Float, nullable=False, default=0.0)
    profit_abs = Column(Float, nullable=False, default=0.0)
    payload_json = Column(Text, nullable=False)
    created_at = Column(String(48), nullable=False)


class AdaptiveState(FreqLLMModelBase):
    """Versioned active adaptive parameter state."""

    __tablename__ = "adaptive_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scope = Column(String(64), nullable=False, index=True)
    active_from = Column(String(48), nullable=False)
    mode = Column(String(24), nullable=False)
    params_json = Column(Text, nullable=False)
    metrics_json = Column(Text, nullable=False)
    reasons_json = Column(Text, nullable=False)
    created_at = Column(String(48), nullable=False)


class AdaptiveUpdate(FreqLLMModelBase):
    """Audit row for one adaptive parameter change."""

    __tablename__ = "adaptive_updates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scope = Column(String(64), nullable=False, index=True)
    param_name = Column(String(96), nullable=False)
    old_value = Column(Float, nullable=False)
    new_value = Column(Float, nullable=False)
    reason = Column(String(160), nullable=False)
    metrics_json = Column(Text, nullable=False)
    created_at = Column(String(48), nullable=False)


# Database Manager


def _get_request_or_thread_id() -> str:
    """Return current thread id (used as scoped_session scope function)."""
    return str(threading.current_thread().ident)


class FreqLLMDatabase:
    """
    Unified database manager for all FreqLLM persistent data.

    Uses SQLAlchemy with scoped_session, following the same pattern as
    freqtrade's core persistence layer (freqtrade/persistence/models.py).
    """

    def __init__(self, db_url: str = DEFAULT_DB_URL):
        """
        :param db_url: SQLAlchemy database URL.
                       Defaults to 'sqlite:///user_data/freqllm/freqllm.db'.
        """
        self.db_url = db_url

        # Ensure directory exists for SQLite files
        if db_url.startswith("sqlite:///"):
            db_path = Path(db_url.removeprefix("sqlite:///"))
            if db_path.parent != Path():
                db_path.parent.mkdir(parents=True, exist_ok=True)

        # Build engine kwargs (consistent with freqtrade's init_db)
        kwargs: dict[str, Any] = {}
        if db_url.startswith("sqlite://"):
            kwargs["connect_args"] = {"check_same_thread": False}
        # MySQL/MariaDB connection pool configuration
        elif db_url.startswith(("mysql", "mariadb")):
            kwargs.update(
                {
                    "pool_size": 32,
                    "max_overflow": 128,
                    "pool_timeout": 60,
                    # Avoid disconnects caused by the MySQL wait_timeout setting.
                    "pool_recycle": 1800,
                    # Validate pooled connections before handing them to a caller.
                    "pool_pre_ping": True,
                }
            )

        self._engine = create_engine(db_url, future=True, **kwargs)
        self._session_factory = scoped_session(
            sessionmaker(bind=self._engine, autoflush=False),
            scopefunc=_get_request_or_thread_id,
        )

        # Create all tables
        FreqLLMModelBase.metadata.create_all(self._engine)
        self.adaptive = AdaptiveRepository(self)
        logger.info("FreqLLM database initialised using the configured connection URL.")

    @property
    def session(self) -> Session:
        """Return the scoped session for the current thread."""
        return self._session_factory()

    def remove_session(self) -> None:
        """Remove the scoped session for the current thread."""
        self._session_factory.remove()

    def _remove_session(self) -> None:
        """Remove the scoped session after a legacy database operation."""
        self.remove_session()

    def close(self) -> None:
        """Release scoped sessions and database connection-pool resources."""
        self._session_factory.remove()
        self._engine.dispose()

    # Trade History operations

    def record_trade(
        self,
        pair: str,
        profit_ratio: float,
        duration_hours: float,
        **reasons: str,
    ) -> None:
        """Insert a closed trade record into the database."""
        allowed_reasons = {"entry_reason", "exit_reason", "llm_reason"}
        unknown_reasons = set(reasons) - allowed_reasons
        if unknown_reasons:
            raise ValueError(f"Unknown trade reason fields: {sorted(unknown_reasons)}")
        entry_reason = reasons.get("entry_reason", "")
        exit_reason = reasons.get("exit_reason", "")
        llm_reason = reasons.get("llm_reason", "")
        session = self.session
        try:
            record = TradeHistory(
                pair=pair,
                profit_ratio=profit_ratio,
                duration_hours=duration_hours,
                entry_reason=entry_reason,
                exit_reason=exit_reason,
                llm_reason=llm_reason,
                is_win=1 if profit_ratio > 0 else 0,
            )
            session.add(record)
            session.commit()
            logger.debug(
                "[%s] Trade persisted: profit=%.4f, duration=%.1fh, exit=%s",
                pair,
                profit_ratio,
                duration_hours,
                exit_reason,
            )
        except DATABASE_ERRORS as e:
            session.rollback()
            logger.warning("[%s] Failed to persist trade record: %s", pair, e)
        finally:
            self._remove_session()

    def get_trade_history(self, pair: str, limit: int = 200) -> list[dict[str, Any]]:
        """
        Retrieve recent trade records for a pair, ordered chronologically.

        :param pair: Trading pair.
        :param limit: Maximum number of records to return.
        :return: List of trade record dicts (oldest first).
        """
        session = self.session
        try:
            rows = (
                session.query(TradeHistory)
                .filter(TradeHistory.pair == pair)
                .order_by(TradeHistory.timestamp.desc())
                .limit(limit)
                .all()
            )
            # Return in chronological order (oldest first) for stats computation
            return [r.to_dict() for r in reversed(rows)]
        except DATABASE_ERRORS as e:
            logger.warning("[%s] Failed to load trade history: %s", pair, e)
            return []
        finally:
            self._remove_session()

    @staticmethod
    def _compute_trade_stats(session: Session, pair: str) -> dict[str, Any]:
        """Core stats computation using an existing session (no session lifecycle management)."""
        total = session.query(TradeHistory.id).filter(TradeHistory.pair == pair).count()

        if total == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "max_consecutive_losses": 0,
            }

        wins = (
            session.query(sql_count(TradeHistory.id))
            .filter(TradeHistory.pair == pair, TradeHistory.is_win == 1)
            .scalar()
        ) or 0

        avg_profit = (
            session.query(func.avg(TradeHistory.profit_ratio))
            .filter(TradeHistory.pair == pair)
            .scalar()
        ) or 0.0

        # Max consecutive losses requires row-level scan
        rows = (
            session.query(TradeHistory.is_win)
            .filter(TradeHistory.pair == pair)
            .order_by(TradeHistory.timestamp.asc())
            .all()
        )
        max_consec_loss = 0
        current = 0
        for (is_win,) in rows:
            if not is_win:
                current += 1
                max_consec_loss = max(max_consec_loss, current)
            else:
                current = 0

        return {
            "total_trades": total,
            "win_rate": wins / total if total > 0 else 0.0,
            "avg_profit": float(avg_profit),
            "max_consecutive_losses": max_consec_loss,
        }

    def get_all_pairs_stats(self) -> dict[str, dict[str, Any]]:
        """Return trade stats for all pairs that have history."""
        session = self.session
        try:
            pairs = session.query(TradeHistory.pair).distinct().all()
            # Use the internal method to avoid nested session remove
            return {pair: self._compute_trade_stats(session, pair) for (pair,) in pairs}
        except DATABASE_ERRORS as e:
            logger.warning("Failed to load all pairs stats: %s", e)
            return {}
        finally:
            self._remove_session()

    # Token Usage operations

    def record_token_usage(self, pair: str, call_type: str, **usage: float) -> None:
        """Insert a token usage record into the database."""
        required_fields = {"prompt_tokens", "completion_tokens", "total_tokens"}
        allowed_fields = {*required_fields, "cost"}
        unknown_fields = set(usage) - allowed_fields
        missing_fields = required_fields - set(usage)
        if unknown_fields or missing_fields:
            raise ValueError(
                f"Invalid token usage fields: missing={sorted(missing_fields)}, "
                f"unknown={sorted(unknown_fields)}"
            )
        session = self.session
        try:
            record = TokenUsage(
                pair=pair,
                call_type=call_type,
                prompt_tokens=int(usage["prompt_tokens"]),
                completion_tokens=int(usage["completion_tokens"]),
                total_tokens=int(usage["total_tokens"]),
                cost=float(usage.get("cost", 0.0)),
            )
            session.add(record)
            session.commit()
        except DATABASE_ERRORS as e:
            session.rollback()
            logger.warning("[%s] Failed to persist token usage: %s", pair, e)
        finally:
            self._remove_session()

    # Long/Short Ratio History operations

    @staticmethod
    def _partition_ls_records(
        records: list[dict[str, Any]],
    ) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], list[dict[str, Any]]]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        direct_records: list[dict[str, Any]] = []
        for source in records:
            record = dict(source)
            record["source_timestamp"] = _as_utc_naive(source.get("source_timestamp"))
            timestamp = record["source_timestamp"]
            if timestamp is None:
                direct_records.append(record)
            else:
                groups[(record["pair"], record["ratio_type"])].append(record)
        return groups, direct_records

    @staticmethod
    def _existing_ls_timestamps(
        session: Session,
        groups: dict[tuple[str, str], list[dict[str, Any]]],
    ) -> dict[tuple[str, str], set[datetime]]:
        existing: dict[tuple[str, str], set[datetime]] = {}
        for (pair, ratio_type), records in groups.items():
            timestamps = [record["source_timestamp"] for record in records]
            rows = (
                session.query(LongShortRatioHistory.source_timestamp)
                .filter(
                    LongShortRatioHistory.pair == pair,
                    LongShortRatioHistory.ratio_type == ratio_type,
                    LongShortRatioHistory.source_timestamp.in_(timestamps),
                )
                .all()
            )
            existing[(pair, ratio_type)] = {row[0] for row in rows}
        return existing

    @staticmethod
    def _add_ls_record(session: Session, record: dict[str, Any]) -> None:
        session.add(
            LongShortRatioHistory(
                pair=record["pair"],
                ratio_type=record["ratio_type"],
                long_short_ratio=record["long_short_ratio"],
                period=record.get("period", "5m"),
                source_timestamp=record.get("source_timestamp"),
            )
        )

    @classmethod
    def _stage_ls_records(
        cls,
        session: Session,
        groups: dict[tuple[str, str], list[dict[str, Any]]],
        direct_records: list[dict[str, Any]],
        existing: dict[tuple[str, str], set[datetime]],
    ) -> tuple[int, int]:
        inserted = len(direct_records)
        skipped = 0
        for record in direct_records:
            cls._add_ls_record(session, record)
        for key, records in groups.items():
            known_timestamps = existing.get(key, set())
            for record in records:
                if record["source_timestamp"] in known_timestamps:
                    skipped += 1
                else:
                    cls._add_ls_record(session, record)
                    inserted += 1
        return inserted, skipped

    def record_ls_ratios_batch(self, records: list[dict[str, Any]]) -> None:
        """Insert L/S ratio points while deduplicating timestamped records."""
        if not records:
            return
        session = self.session
        try:
            groups, direct_records = self._partition_ls_records(records)
            existing = self._existing_ls_timestamps(session, groups)
            inserted, skipped = self._stage_ls_records(session, groups, direct_records, existing)
            if inserted:
                session.commit()
                logger.debug(
                    "Persisted %d L/S ratio records in batch (skipped %d duplicates).",
                    inserted,
                    skipped,
                )
            elif skipped:
                logger.debug("All %d L/S ratio records already exist, nothing to insert.", skipped)
        except DATABASE_ERRORS as exc:
            session.rollback()
            logger.warning("Failed to batch persist L/S ratios: %s", exc)
        finally:
            self._remove_session()

    def get_latest_ls_ratio_timestamp(
        self,
        pair: str,
        ratio_type: str,
    ) -> datetime | None:
        """Return the latest source_timestamp for a given pair + ratio_type.

        Used to determine how much new data needs to be fetched from the exchange.
        Returns None if no records exist.
        """
        session = self.session
        try:
            row = (
                session.query(func.max(LongShortRatioHistory.source_timestamp))
                .filter(
                    LongShortRatioHistory.pair == pair,
                    LongShortRatioHistory.ratio_type == ratio_type,
                    LongShortRatioHistory.source_timestamp.isnot(None),
                )
                .scalar()
            )
            return _as_utc_aware(row)
        except DATABASE_ERRORS as e:
            logger.warning("[%s] Failed to get latest L/S ratio timestamp: %s", pair, e)
            return None
        finally:
            self._remove_session()

    def get_ls_ratio_records(
        self, pair: str, ratio_type: str, **filters: str | datetime | int | None
    ) -> list[dict[str, Any]]:
        """Retrieve L/S ratio history within an absolute time range."""
        allowed_filters = {"period", "since", "until", "limit"}
        unknown_filters = set(filters) - allowed_filters
        if unknown_filters:
            raise ValueError(f"Unknown L/S ratio filters: {sorted(unknown_filters)}")
        period = filters.get("period")
        since = filters.get("since")
        until = filters.get("until")
        limit = filters.get("limit")
        if period is not None and not isinstance(period, str):
            raise TypeError("period must be a string")
        if since is not None and not isinstance(since, datetime):
            raise TypeError("since must be a datetime")
        if until is not None and not isinstance(until, datetime):
            raise TypeError("until must be a datetime")
        if limit is not None and not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        session = self.session
        try:
            query = session.query(LongShortRatioHistory).filter(
                LongShortRatioHistory.pair == pair,
                LongShortRatioHistory.ratio_type == ratio_type,
            )
            if period is not None:
                query = query.filter(LongShortRatioHistory.period == period)
            if since is not None:
                query = query.filter(
                    func.coalesce(
                        LongShortRatioHistory.source_timestamp, LongShortRatioHistory.timestamp
                    )
                    >= _as_utc_naive(since)
                )
            if until is not None:
                query = query.filter(
                    func.coalesce(
                        LongShortRatioHistory.source_timestamp, LongShortRatioHistory.timestamp
                    )
                    <= _as_utc_naive(until)
                )
            if limit is not None and limit > 0:
                query = query.order_by(
                    func.coalesce(
                        LongShortRatioHistory.source_timestamp,
                        LongShortRatioHistory.timestamp,
                    ).desc()
                ).limit(limit)
                rows = list(reversed(query.all()))
            else:
                rows = query.order_by(
                    func.coalesce(
                        LongShortRatioHistory.source_timestamp,
                        LongShortRatioHistory.timestamp,
                    ).asc()
                ).all()
            return [
                {
                    "long_short_ratio": r.long_short_ratio,
                    "period": r.period,
                    "timestamp": _as_utc_aware(r.timestamp).isoformat() if r.timestamp else None,
                    "source_timestamp": _as_utc_aware(r.source_timestamp).isoformat()
                    if r.source_timestamp
                    else None,
                }
                for r in rows
            ]
        except DATABASE_ERRORS as e:
            logger.warning("[%s] Failed to load ranged L/S ratio history: %s", pair, e)
            return []
        finally:
            self._remove_session()

    def get_ls_ratio_history(
        self,
        pair: str,
        ratio_type: str = "global_account",
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Retrieve historical L/S ratio data for a pair within the given time window.

        :param pair: Trading pair.
        :param ratio_type: Type of ratio to retrieve.
        :param days: Number of days of history to fetch.
        :return: List of dicts with ratio and timestamp, ordered chronologically.
        """
        session = self.session
        try:
            since = _utcnow() - timedelta(days=days)
            rows = (
                session.query(LongShortRatioHistory)
                .filter(
                    LongShortRatioHistory.pair == pair,
                    LongShortRatioHistory.ratio_type == ratio_type,
                    LongShortRatioHistory.timestamp >= since,
                )
                .order_by(LongShortRatioHistory.timestamp.asc())
                .all()
            )
            return [
                {
                    "long_short_ratio": r.long_short_ratio,
                    "timestamp": _as_utc_aware(r.timestamp).isoformat() if r.timestamp else None,
                    "source_timestamp": _as_utc_aware(r.source_timestamp).isoformat()
                    if r.source_timestamp
                    else None,
                }
                for r in rows
            ]
        except DATABASE_ERRORS as e:
            logger.warning("[%s] Failed to load L/S ratio history: %s", pair, e)
            return []
        finally:
            self._remove_session()

    def cleanup_ls_ratio_history(self, retention_days: int = 60) -> int:
        """Delete L/S ratio records older than retention_days.

        :param retention_days: Records older than this will be deleted.
        :return: Number of records deleted.
        """
        session = self.session
        try:
            cutoff = _as_utc_naive(_utcnow() - timedelta(days=retention_days))
            deleted = (
                session.query(LongShortRatioHistory)
                .filter(
                    func.coalesce(
                        LongShortRatioHistory.source_timestamp, LongShortRatioHistory.timestamp
                    )
                    < cutoff
                )
                .delete(synchronize_session=False)
            )
            session.commit()
            if deleted > 0:
                logger.info(
                    "Cleaned up %d L/S ratio records older than %d days.",
                    deleted,
                    retention_days,
                )
            return deleted
        except DATABASE_ERRORS as e:
            session.rollback()
            logger.warning("Failed to clean up L/S ratio history: %s", e)
            return 0
        finally:
            self._remove_session()

    @staticmethod
    def _market_feature_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(record.get("pair", "")),
            str(record.get("feature_type", "")),
            str(record.get("period", "snapshot")),
            str(record.get("feature_key", "")),
        )

    @classmethod
    def _partition_market_features(
        cls, records: list[dict[str, Any]]
    ) -> tuple[dict[tuple[str, str, str, str], list[dict[str, Any]]], list[dict[str, Any]]]:
        groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        direct_records: list[dict[str, Any]] = []
        for source in records:
            record = dict(source)
            record["source_timestamp"] = _as_utc_naive(source.get("source_timestamp"))
            record["next_funding_time"] = _as_utc_naive(source.get("next_funding_time"))
            if record["source_timestamp"] is None:
                direct_records.append(record)
            else:
                groups[cls._market_feature_key(record)].append(record)
        return groups, direct_records

    @staticmethod
    def _existing_market_timestamps(
        session: Session,
        groups: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    ) -> dict[tuple[str, str, str, str], set[datetime]]:
        existing: dict[tuple[str, str, str, str], set[datetime]] = {}
        for key, records in groups.items():
            pair, feature_type, period, feature_key = key
            timestamps = [record["source_timestamp"] for record in records]
            rows = (
                session.query(MarketFeatureHistory.source_timestamp)
                .filter(
                    MarketFeatureHistory.pair == pair,
                    MarketFeatureHistory.feature_type == feature_type,
                    MarketFeatureHistory.period == period,
                    MarketFeatureHistory.feature_key == feature_key,
                    MarketFeatureHistory.source_timestamp.in_(timestamps),
                )
                .all()
            )
            existing[key] = {row[0] for row in rows}
        return existing

    @staticmethod
    def _stage_market_features(
        session: Session,
        groups: dict[tuple[str, str, str, str], list[dict[str, Any]]],
        direct_records: list[dict[str, Any]],
        existing: dict[tuple[str, str, str, str], set[datetime]],
    ) -> tuple[int, int]:
        inserted = len(direct_records)
        skipped = 0
        for record in direct_records:
            session.add(MarketFeatureHistory(**record))
        for key, records in groups.items():
            known_timestamps = existing.get(key, set())
            for record in records:
                if record["source_timestamp"] in known_timestamps:
                    skipped += 1
                else:
                    session.add(MarketFeatureHistory(**record))
                    inserted += 1
        return inserted, skipped

    def record_market_features_batch(self, records: list[dict[str, Any]]) -> None:
        """Insert market snapshots while deduplicating timestamped records."""
        if not records:
            return
        session = self.session
        try:
            groups, direct_records = self._partition_market_features(records)
            existing = self._existing_market_timestamps(session, groups)
            inserted, skipped = self._stage_market_features(
                session, groups, direct_records, existing
            )
            if inserted:
                session.commit()
                logger.debug(
                    "Persisted %d market feature records in batch (skipped %d duplicates).",
                    inserted,
                    skipped,
                )
            elif skipped:
                logger.debug(
                    "All %d market feature records already exist, nothing to insert.", skipped
                )
        except DATABASE_ERRORS as exc:
            session.rollback()
            logger.warning("Failed to batch persist market features: %s", exc)
        finally:
            self._remove_session()

    def get_latest_market_feature_timestamp(
        self,
        pair: str,
        feature_type: str,
        period: str = "snapshot",
        feature_key: str = "",
    ) -> datetime | None:
        """Return the latest source_timestamp for a given pair + feature bucket."""
        session = self.session
        try:
            row = (
                session.query(func.max(MarketFeatureHistory.source_timestamp))
                .filter(
                    MarketFeatureHistory.pair == pair,
                    MarketFeatureHistory.feature_type == feature_type,
                    MarketFeatureHistory.period == period,
                    MarketFeatureHistory.feature_key == feature_key,
                    MarketFeatureHistory.source_timestamp.isnot(None),
                )
                .scalar()
            )
            return _as_utc_aware(row)
        except DATABASE_ERRORS as e:
            logger.warning(
                "[%s] Failed to get latest market feature timestamp for %s: %s",
                pair,
                feature_type,
                e,
            )
            return None
        finally:
            self._remove_session()

    @staticmethod
    def _validate_market_history_filters(
        filters: dict[str, str | datetime | int | None],
    ) -> dict[str, str | datetime | int | None]:
        allowed_types = {
            "period": str,
            "feature_key": str,
            "since": datetime,
            "until": datetime,
            "limit": int,
        }
        unknown_filters = set(filters) - set(allowed_types)
        if unknown_filters:
            raise ValueError(f"Unknown market feature filters: {sorted(unknown_filters)}")
        for name, value in filters.items():
            if value is not None and not isinstance(value, allowed_types[name]):
                raise TypeError(f"{name} has an invalid type")
        return filters

    @staticmethod
    def _market_feature_timestamp() -> Any:
        return func.coalesce(MarketFeatureHistory.source_timestamp, MarketFeatureHistory.timestamp)

    @staticmethod
    def _query_market_feature_rows(
        session: Session,
        pair: str,
        feature_type: str,
        filters: dict[str, str | datetime | int | None],
    ) -> list[MarketFeatureHistory]:
        query = session.query(MarketFeatureHistory).filter(
            MarketFeatureHistory.pair == pair,
            MarketFeatureHistory.feature_type == feature_type,
        )
        period = filters.get("period")
        feature_key = filters.get("feature_key")
        since = filters.get("since")
        until = filters.get("until")
        limit = filters.get("limit")
        if isinstance(period, str):
            query = query.filter(MarketFeatureHistory.period == period)
        if isinstance(feature_key, str):
            query = query.filter(MarketFeatureHistory.feature_key == feature_key)
        timestamp = FreqLLMDatabase._market_feature_timestamp()
        if isinstance(since, datetime):
            query = query.filter(timestamp >= _as_utc_naive(since))
        if isinstance(until, datetime):
            query = query.filter(timestamp <= _as_utc_naive(until))
        if isinstance(limit, int) and limit > 0:
            return list(reversed(query.order_by(timestamp.desc()).limit(limit).all()))
        return query.order_by(timestamp.asc()).all()

    def get_market_feature_history(
        self, pair: str, feature_type: str, **filters: str | datetime | int | None
    ) -> list[dict[str, Any]]:
        """Retrieve persisted market feature records for a pair and feature type."""
        validated_filters = self._validate_market_history_filters(filters)
        session = self.session
        try:
            rows = self._query_market_feature_rows(session, pair, feature_type, validated_filters)
            results: list[dict[str, Any]] = []
            for row in rows:
                results.append(
                    {
                        "pair": row.pair,
                        "feature_type": row.feature_type,
                        "period": row.period,
                        "feature_key": row.feature_key,
                        "funding_rate": row.funding_rate,
                        "basis": row.basis,
                        "basis_pct": row.basis_pct,
                        "mark_price": row.mark_price,
                        "index_price": row.index_price,
                        "next_funding_time": _as_utc_aware(row.next_funding_time).isoformat()
                        if row.next_funding_time
                        else None,
                        "open_interest": row.open_interest,
                        "open_interest_value": row.open_interest_value,
                        "buy_vol": row.buy_vol,
                        "sell_vol": row.sell_vol,
                        "buy_sell_ratio": row.buy_sell_ratio,
                        "buy_sell_imbalance": row.buy_sell_imbalance,
                        "event_count": row.event_count,
                        "max_leverage": row.max_leverage,
                        "spread": row.spread,
                        "spread_pct": row.spread_pct,
                        "bid_ask_imbalance": row.bid_ask_imbalance,
                        "bid_total_qty": row.bid_total_qty,
                        "ask_total_qty": row.ask_total_qty,
                        "bid_depth_value": row.bid_depth_value,
                        "ask_depth_value": row.ask_depth_value,
                        "price_impact_buy_pct": row.price_impact_buy_pct,
                        "price_impact_sell_pct": row.price_impact_sell_pct,
                        "liquidation_side": row.liquidation_side,
                        "liquidation_price": row.liquidation_price,
                        "liquidation_qty": row.liquidation_qty,
                        "raw_payload": row.raw_payload,
                        "timestamp": _as_utc_aware(row.timestamp).isoformat()
                        if row.timestamp
                        else None,
                        "source_timestamp": _as_utc_aware(row.source_timestamp).isoformat()
                        if row.source_timestamp
                        else None,
                    }
                )
            return results
        except DATABASE_ERRORS as e:
            logger.warning(
                "[%s] Failed to load market feature history for %s: %s",
                pair,
                feature_type,
                e,
            )
            return []
        finally:
            self._remove_session()

    def cleanup_market_feature_history(self, retention_days: int = 60) -> int:
        """Delete market feature records older than retention_days."""
        session = self.session
        try:
            cutoff = _as_utc_naive(_utcnow() - timedelta(days=retention_days))
            deleted = (
                session.query(MarketFeatureHistory)
                .filter(
                    func.coalesce(
                        MarketFeatureHistory.source_timestamp, MarketFeatureHistory.timestamp
                    )
                    < cutoff
                )
                .delete(synchronize_session=False)
            )
            session.commit()
            if deleted > 0:
                logger.info(
                    "Cleaned up %d market feature records older than %d days.",
                    deleted,
                    retention_days,
                )
            return deleted
        except DATABASE_ERRORS as e:
            session.rollback()
            logger.warning("Failed to clean up market feature history: %s", e)
            return 0
        finally:
            self._remove_session()

    def get_token_stats(self) -> dict[str, Any]:
        """Return aggregate token usage statistics."""
        session = self.session
        try:
            total_calls = session.query(sql_count(TokenUsage.id)).scalar() or 0
            total_prompt = session.query(func.sum(TokenUsage.prompt_tokens)).scalar() or 0
            total_completion = session.query(func.sum(TokenUsage.completion_tokens)).scalar() or 0
            total_tokens = session.query(func.sum(TokenUsage.total_tokens)).scalar() or 0
            total_cost = session.query(func.sum(TokenUsage.cost)).scalar() or 0.0

            # Per-pair breakdown
            pair_rows = (
                session.query(
                    TokenUsage.pair,
                    sql_count(TokenUsage.id),
                    func.sum(TokenUsage.total_tokens),
                    func.sum(TokenUsage.cost),
                )
                .group_by(TokenUsage.pair)
                .all()
            )
            pair_stats = {}
            for pair, calls, tokens, cost in pair_rows:
                pair_stats[pair] = {
                    "calls": calls or 0,
                    "total_tokens": int(tokens or 0),
                    "cost": float(cost or 0),
                }

            return {
                "total_calls": total_calls,
                "total_prompt_tokens": int(total_prompt),
                "total_completion_tokens": int(total_completion),
                "total_tokens": int(total_tokens),
                "total_cost": float(total_cost),
                "pair_stats": pair_stats,
            }
        except DATABASE_ERRORS as e:
            logger.warning("Failed to compute token stats: %s", e)
            return {
                "total_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "pair_stats": {},
            }
        finally:
            self._remove_session()
