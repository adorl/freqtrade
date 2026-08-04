"""
Long/short ratio data collector - fetches futures sentiment data via exchange REST APIs.
Supports Binance USDM Futures and Bybit V5, with an extensible base class for others.
"""

import logging
import math
import statistics
import threading
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any


logger = logging.getLogger(__name__)


def _ensure_utc_datetime(value: datetime | None) -> datetime | None:
    """Normalise datetime objects to UTC-aware datetimes."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _clamp_limit(value: Any, minimum: int = 1, maximum: int = 500) -> int:
    """Clamp an API limit parameter to the exchange-accepted range."""
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = minimum
    return max(minimum, min(normalized, maximum))


class LongShortRatioBase(ABC):
    """Abstract base class for exchange-specific long/short ratio collectors."""

    def __init__(self, exchange_instance):
        """Initialize the collector with a Freqtrade exchange instance."""
        self.exchange = exchange_instance
        self._cache: dict[str, Any] = {}

    @abstractmethod
    def get_top_trader_account_ratio(
        self, symbol: str, period: str = "5m", limit: int = 10
    ) -> list[dict] | None:
        """Fetch top-trader long/short ratio by account count."""
        raise NotImplementedError

    @abstractmethod
    def get_top_trader_position_ratio(
        self, symbol: str, period: str = "5m", limit: int = 10
    ) -> list[dict] | None:
        """Fetch top-trader long/short ratio by position size."""
        raise NotImplementedError

    @abstractmethod
    def get_global_account_ratio(
        self, symbol: str, period: str = "5m", limit: int = 10
    ) -> list[dict] | None:
        """Fetch global long/short ratio by account count (all traders)."""
        raise NotImplementedError

    def _api_method(self, method_name: str) -> Any | None:
        """Return an implicit ccxt method without exposing a protected access expression."""
        api = vars(self.exchange).get("_api")
        return getattr(api, method_name, None)

    def _safe_request(self, method_name: str, params: dict) -> Any | None:
        """Call a ccxt method by name, returning ``None`` when it is unavailable or fails."""
        try:
            method = self._api_method(method_name)
            if method is None:
                logger.debug("Exchange does not support method: %s", method_name)
                return None
            return method(params)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
            logger.warning("Request %s failed: %s", method_name, error)
            return None

    def _safe_request_any(self, method_names: list[str], params: dict) -> Any | None:
        """Call the first supported ccxt method from the provided candidates."""
        for method_name in method_names:
            method = self._api_method(method_name)
            if method is None:
                continue
            try:
                return method(params)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
                logger.warning("Request %s failed: %s", method_name, error)
                return None
        logger.debug("Exchange does not support methods: %s", ", ".join(method_names))
        return None

    @staticmethod
    def _to_base_symbol(pair: str) -> str:
        """Convert a Freqtrade pair such as ``BTC/USDT:USDT`` to ``BTCUSDT``."""
        symbol = pair.split(":")[0]
        return symbol.replace("/", "")


class BinanceLongShortRatio(LongShortRatioBase):
    """Binance USDM Futures long/short ratio collector."""

    def get_top_trader_account_ratio(
        self, symbol: str, period: str = "5m", limit: int = 10
    ) -> list[dict] | None:
        """Top-trader L/S ratio by account count (/futures/data/topLongShortAccountRatio)."""
        limit = _clamp_limit(limit, maximum=2880)
        data = self._safe_request(
            "fapiDataGetTopLongShortAccountRatio",
            {"symbol": self._to_base_symbol(symbol), "period": period, "limit": limit},
        )
        if not data:
            return None
        return [
            {
                "timestamp": item.get("timestamp"),
                "long_account": float(item.get("longAccount", 0)),
                "short_account": float(item.get("shortAccount", 0)),
                "long_short_ratio": float(item.get("longShortRatio", 0)),
            }
            for item in data
        ]

    def get_top_trader_position_ratio(
        self, symbol: str, period: str = "5m", limit: int = 10
    ) -> list[dict] | None:
        """Top-trader L/S ratio by position size (/futures/data/topLongShortPositionRatio)."""
        limit = _clamp_limit(limit, maximum=2880)
        data = self._safe_request(
            "fapiDataGetTopLongShortPositionRatio",
            {"symbol": self._to_base_symbol(symbol), "period": period, "limit": limit},
        )
        if not data:
            return None
        return [
            {
                "timestamp": item.get("timestamp"),
                "long_position": float(item.get("longAccount", 0)),
                "short_position": float(item.get("shortAccount", 0)),
                "long_short_ratio": float(item.get("longShortRatio", 0)),
            }
            for item in data
        ]

    def get_global_account_ratio(
        self, symbol: str, period: str = "5m", limit: int = 10
    ) -> list[dict] | None:
        """Global L/S ratio by account count (/futures/data/globalLongShortAccountRatio)."""
        limit = _clamp_limit(limit, maximum=2880)
        data = self._safe_request(
            "fapiDataGetGlobalLongShortAccountRatio",
            {"symbol": self._to_base_symbol(symbol), "period": period, "limit": limit},
        )
        if not data:
            return None
        return [
            {
                "timestamp": item.get("timestamp"),
                "long_account": float(item.get("longAccount", 0)),
                "short_account": float(item.get("shortAccount", 0)),
                "long_short_ratio": float(item.get("longShortRatio", 0)),
            }
            for item in data
        ]

    def get_taker_buy_sell_volume(
        self, symbol: str, period: str = "5m", limit: int = 10
    ) -> list[dict] | None:
        """Taker buy/sell volume ratio (/futures/data/takerlongshortRatio)."""
        limit = _clamp_limit(limit, maximum=2880)
        data = self._safe_request_any(
            [
                "fapiDataGetTakerlongshortRatio",
                "fapiDataGetTakerLongShortRatio",
                "fapiDataGetTakerBuySellVol",
            ],
            {"symbol": self._to_base_symbol(symbol), "period": period, "limit": limit},
        )
        if not data:
            return None
        return [
            {
                "timestamp": item.get("timestamp"),
                "buy_vol": float(item.get("buyVol", 0)),
                "sell_vol": float(item.get("sellVol", 0)),
                "buy_sell_ratio": float(item.get("buySellRatio", 0)),
            }
            for item in data
        ]


class BybitLongShortRatio(LongShortRatioBase):
    """Bybit V5 long/short ratio collector."""

    def get_top_trader_account_ratio(
        self, symbol: str, period: str = "5m", limit: int = 10
    ) -> list[dict] | None:
        """Account long/short ratio via /v5/market/account-ratio."""
        data = self._safe_request(
            "publicGetV5MarketAccountRatio",
            {"symbol": self._to_base_symbol(symbol), "period": period, "limit": limit},
        )
        if not data or not data.get("result", {}).get("list"):
            return None
        return [
            {
                "timestamp": item.get("timestamp"),
                "buy_ratio": float(item.get("buyRatio", 0)),
                "sell_ratio": float(item.get("sellRatio", 0)),
                "long_short_ratio": (
                    float(item.get("buyRatio", 0)) / max(float(item.get("sellRatio", 1)), 1e-10)
                ),
            }
            for item in data["result"]["list"]
        ]

    def get_top_trader_position_ratio(
        self, symbol: str, period: str = "5m", limit: int = 10
    ) -> list[dict] | None:
        """Bybit does not provide a dedicated top-trader position ratio endpoint."""
        return None

    def get_global_account_ratio(
        self, symbol: str, period: str = "5m", limit: int = 10
    ) -> list[dict] | None:
        """Reuse account ratio as global ratio for Bybit."""
        return self.get_top_trader_account_ratio(symbol, period, limit)


class LongShortRatioCollector:
    """Unified collector with exchange selection and optional historical persistence."""

    def __init__(self, exchange_instance, database=None, config=None):
        """Initialize the collector, optional database, and analysis configuration."""
        self.exchange = exchange_instance
        self._impl: LongShortRatioBase | None = None
        self._cache: dict[str, dict[str, Any]] = {}
        self._db = database
        self._config = config
        self._init_impl()

    def _init_impl(self) -> None:
        """Select the correct implementation based on the exchange name."""
        exchange_name = ""
        try:
            exchange_name = self.exchange.name.lower()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass

        if "binance" in exchange_name:
            self._impl = BinanceLongShortRatio(self.exchange)
            logger.info("LongShortRatioCollector: using Binance implementation")
        elif "bybit" in exchange_name:
            self._impl = BybitLongShortRatio(self.exchange)
            logger.info("LongShortRatioCollector: using Bybit implementation")
        else:
            logger.warning(
                "No dedicated L/S ratio implementation for '%s'; "
                "falling back to Binance-compatible interface.",
                exchange_name,
            )
            self._impl = BinanceLongShortRatio(self.exchange)

    # Mapping from data key to ratio_type
    _TYPE_KEY_MAP = {
        "top_trader_account_ratio": "top_trader_account",
        "top_trader_position_ratio": "top_trader_position",
        "global_account_ratio": "global_account",
    }

    # Mapping from period string to seconds, used to calculate fetch amount
    _PERIOD_SECONDS = {
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "6h": 21600,
        "12h": 43200,
        "1d": 86400,
    }

    def _calc_needed_limit(
        self, pair: str, ratio_type: str, period: str, default_limit: int
    ) -> int:
        """Limit fetching to records newer than the latest persisted timestamp."""
        try:
            default_limit = max(1, int(default_limit))
        except (TypeError, ValueError):
            default_limit = 1

        if self._db is None:
            return default_limit
        try:
            latest_ts = _ensure_utc_datetime(
                self._db.get_latest_ls_ratio_timestamp(pair, ratio_type)
            )
            if latest_ts is None:
                return default_limit
            now = datetime.now(UTC)
            elapsed_seconds = max(0.0, (now - latest_ts).total_seconds())
            period_seconds = self._PERIOD_SECONDS.get(period, 300)
            needed = int(elapsed_seconds / period_seconds) + 1  # +1 to ensure boundary coverage
            return max(1, min(needed, default_limit))
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
            logger.debug("[%s] Failed to calc needed limit for %s: %s", pair, ratio_type, error)
            return default_limit

    def _collect_standard_ratios(
        self, pair: str, period: str, limit: int, result: dict[str, Any]
    ) -> None:
        """Collect the three exchange-independent ratio types."""
        if self._impl is None:
            return
        requests = (
            (
                "top_trader_account_ratio",
                "top_trader_account",
                self._impl.get_top_trader_account_ratio,
            ),
            (
                "top_trader_position_ratio",
                "top_trader_position",
                self._impl.get_top_trader_position_ratio,
            ),
            ("global_account_ratio", "global_account", self._impl.get_global_account_ratio),
        )
        for result_key, ratio_type, fetch in requests:
            try:
                needed = self._calc_needed_limit(pair, ratio_type, period, limit)
                ratio_data = fetch(pair, period, needed)
                if ratio_data:
                    result[result_key] = ratio_data
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
                logger.warning("[%s] %s failed: %s", pair, result_key, error)
                cached = self._cache.get(pair, {}).get(result_key)
                if cached:
                    result[result_key] = cached

    def _collect_taker_volume(
        self, pair: str, period: str, limit: int, result: dict[str, Any]
    ) -> None:
        """Collect the Binance-specific taker buy/sell volume ratio."""
        if not isinstance(self._impl, BinanceLongShortRatio):
            return
        try:
            volume_data = self._impl.get_taker_buy_sell_volume(pair, period, limit)
            if volume_data:
                result["taker_buy_sell_volume"] = volume_data
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
            logger.warning("[%s] taker_buy_sell_volume failed: %s", pair, error)

    def collect(self, pair: str, period: str = "5m", limit: int = 10) -> dict[str, Any]:
        """Collect all available long/short ratio data for the given pair."""
        result: dict[str, Any] = {
            "pair": pair,
            "period": period,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if self._impl is None:
            return result
        self._collect_standard_ratios(pair, period, limit, result)
        self._collect_taker_volume(pair, period, limit, result)
        self._cache[pair] = result
        self._persist_collected_data(pair, result)
        return result

    @staticmethod
    def _parse_source_timestamp(ts_val) -> datetime | None:
        """Parse exchange-returned timestamp into a UTC datetime object."""
        if ts_val is None:
            return None
        try:
            if isinstance(ts_val, (int, float)):
                return datetime.fromtimestamp(ts_val / 1000.0, tz=UTC)
            if isinstance(ts_val, str):
                return _ensure_utc_datetime(datetime.fromisoformat(ts_val))
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None
        return None

    def _persist_collected_data(self, pair: str, data: dict[str, Any]) -> None:
        """Save collected L/S ratio data to database for historical analysis.

        Persists all fetched data entries (not just the latest one). Deduplication
        is handled by the database layer in record_ls_ratios_batch based on
        (pair, ratio_type, source_timestamp).
        """
        if self._db is None:
            return
        try:
            records = []
            period = data.get("period", "5m")
            for data_key, ratio_type in self._TYPE_KEY_MAP.items():
                items = data.get(data_key, [])
                if not items:
                    continue
                # Persist all entries; deduplication is handled by the database layer
                for item in items:
                    src_ts = self._parse_source_timestamp(item.get("timestamp"))
                    records.append(
                        {
                            "pair": pair,
                            "ratio_type": ratio_type,
                            "long_short_ratio": float(item.get("long_short_ratio", 0)),
                            "period": period,
                            "source_timestamp": src_ts,
                        }
                    )
            if records:
                worker = threading.Thread(
                    target=self._db.record_ls_ratios_batch,
                    args=(records,),
                    daemon=True,
                )
                worker.start()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
            logger.debug("[%s] Failed to persist L/S ratio data: %s", pair, error)

    def _historical_distribution_metrics(
        self,
        history: list[dict[str, Any]],
        current_ratio: float,
        current_timestamp: datetime | None,
        series_size: int,
    ) -> dict[str, Any]:
        """Calculate distribution metrics from persisted ratio history."""
        history_ratios = [self._coerce_ratio(item.get("long_short_ratio")) for item in history]
        valid_ratios = [ratio for ratio in history_ratios if ratio is not None]
        if len(valid_ratios) < 5:
            return {
                "percentile": None,
                "is_extreme_high": False,
                "is_extreme_low": False,
                "mean": None,
                "stdev": None,
                "z_score": None,
                "data_points": series_size,
                "history_available": False,
            }
        distribution = list(valid_ratios)
        history_ts = self._parse_source_timestamp(
            history[-1].get("source_timestamp") or history[-1].get("timestamp")
        )
        if current_timestamp is not None and (history_ts is None or current_timestamp > history_ts):
            distribution.append(current_ratio)
        else:
            distribution[-1] = current_ratio
        percentile = round(
            sum(ratio < current_ratio for ratio in distribution) / len(distribution) * 100.0,
            1,
        )
        mean_ratio = statistics.mean(distribution)
        stdev_ratio = statistics.stdev(distribution) if len(distribution) >= 2 else 0.0
        z_score = round((current_ratio - mean_ratio) / stdev_ratio, 2) if stdev_ratio else 0.0
        return {
            "percentile": percentile,
            "is_extreme_high": percentile >= 90.0,
            "is_extreme_low": percentile <= 10.0,
            "mean": round(mean_ratio, 4),
            "stdev": round(stdev_ratio, 4),
            "z_score": z_score,
            "data_points": len(distribution),
            "history_available": True,
        }

    def analyze_historical(
        self,
        pair: str,
        ratio_type: str = "global_account",
        latest_items: list[dict[str, Any]] | None = None,
        short_window: int | None = None,
    ) -> dict[str, Any] | None:
        """Combine historical positioning metrics with recent log-ratio trends."""
        window_days = 30
        medium_window = 12
        if self._config is not None:
            window_days = self._config.market_data.ls_ratio_percentile_window_days
            medium_window = self._config.market_data.ls_ratio_change_rate_periods

        if short_window is None:
            short_window = max(3, min(6, medium_window // 2 if medium_window > 1 else 3))
        short_window = max(1, short_window)

        history: list[dict[str, Any]] = []
        if self._db is not None:
            history = self._db.get_ls_ratio_history(pair, ratio_type=ratio_type, days=window_days)

        series = self._build_ratio_series(history=history, latest_items=latest_items)
        if not series:
            return None

        ratios = [point["long_short_ratio"] for point in series]
        short_metrics = self._compute_window_change(ratios, short_window)
        medium_metrics = self._compute_window_change(ratios, medium_window)
        trend_acceleration = self._classify_trend_acceleration(
            short_metrics["change"],
            medium_metrics["change"],
        )
        distribution_metrics = self._historical_distribution_metrics(
            history,
            ratios[-1],
            series[-1]["timestamp"],
            len(series),
        )
        return {
            "current_ratio": ratios[-1],
            "current_timestamp": series[-1]["timestamp"],
            "change_rate": round(medium_metrics["change"], 4),
            "change_trend": medium_metrics["trend"],
            **distribution_metrics,
            "window_days": window_days if distribution_metrics["history_available"] else None,
            "short_term_change": round(short_metrics["change"], 4),
            "short_term_trend": short_metrics["trend"],
            "short_term_bars": short_metrics["bars"],
            "medium_term_change": round(medium_metrics["change"], 4),
            "medium_term_trend": medium_metrics["trend"],
            "medium_term_bars": medium_metrics["bars"],
            "trend_acceleration": trend_acceleration,
            "reversal_flag": trend_acceleration == "reversing",
            "trend_confidence": self._compute_trend_confidence(
                ratios,
                short_metrics["bars"],
                short_metrics["change"],
            ),
        }

    def cleanup_history(self) -> int:
        """Delete old L/S ratio records based on configured retention days."""
        if self._db is None:
            return 0
        retention = 60
        if self._config is not None:
            retention = self._config.market_data.ls_ratio_history_days
        return self._db.cleanup_ls_ratio_history(retention_days=retention)

    # Mapping from internal ratio_type keys to human-readable labels
    _RATIO_TYPE_LABELS: dict[str, str] = {
        "global_account": "Global Account (retail sentiment)",
        "top_trader_account": "Top Trader Account (smart money direction)",
        "top_trader_position": "Top Trader Position (smart money sizing)",
    }

    @staticmethod
    def _coerce_ratio(value: Any) -> float | None:
        """Convert a ratio field to a positive float when possible."""
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            return None
        return ratio if ratio > 0 else None

    def _build_ratio_series(
        self,
        history: list[dict[str, Any]] | None = None,
        latest_items: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Merge stored history with freshly collected live items."""
        series: list[dict[str, Any]] = []

        for item in history or []:
            ratio = self._coerce_ratio(item.get("long_short_ratio"))
            if ratio is None:
                continue
            series.append(
                {
                    "timestamp": self._parse_source_timestamp(
                        item.get("source_timestamp") or item.get("timestamp")
                    ),
                    "long_short_ratio": ratio,
                }
            )

        for item in latest_items or []:
            ratio = self._coerce_ratio(item.get("long_short_ratio"))
            if ratio is None:
                continue
            point = {
                "timestamp": self._parse_source_timestamp(item.get("timestamp")),
                "long_short_ratio": ratio,
            }
            ts = point["timestamp"]
            if ts is not None:
                replaced = False
                for idx in range(len(series) - 1, max(-1, len(series) - 8), -1):
                    if series[idx]["timestamp"] == ts:
                        series[idx] = point
                        replaced = True
                        break
                if replaced:
                    continue
            series.append(point)

        return series

    @staticmethod
    def _mean_log_ratio(values: list[float]) -> float | None:
        """Return the arithmetic mean of log-ratios for symmetric change analysis."""
        logs = [math.log(value) for value in values if value > 0]
        if not logs:
            return None
        return statistics.mean(logs)

    @staticmethod
    def _classify_change_trend(change: float) -> str:
        """Map a relative change into a coarse trend label."""
        if change >= 0.08:
            return "rising_fast"
        if change >= 0.02:
            return "rising"
        if change <= -0.08:
            return "falling_fast"
        if change <= -0.02:
            return "falling"
        return "stable"

    def _compute_window_change(self, ratios: list[float], requested_window: int) -> dict[str, Any]:
        """Compare a recent ratio window against its baseline using log-space means."""
        if len(ratios) < 2:
            return {"change": 0.0, "trend": "stable", "bars": 0}

        window = max(1, int(requested_window or 1))
        window = min(window, len(ratios) - 1)

        recent = ratios[-window:]
        baseline = (
            ratios[-(window * 2) : -window] if len(ratios) >= window * 2 else ratios[:-window]
        )
        if not baseline:
            baseline = ratios[:-1]
            recent = ratios[-1:]
            window = 1

        baseline_log = self._mean_log_ratio(baseline)
        recent_log = self._mean_log_ratio(recent)
        if baseline_log is None or recent_log is None:
            return {"change": 0.0, "trend": "stable", "bars": 0}

        change = math.exp(recent_log - baseline_log) - 1.0
        return {
            "change": change,
            "trend": self._classify_change_trend(change),
            "bars": window,
        }

    @staticmethod
    def _compute_trend_confidence(ratios: list[float], bars: int, net_change: float) -> float:
        """Estimate whether the recent trend is consistent or noisy."""
        if len(ratios) < 2 or bars <= 0:
            return 0.0

        lookback = min(bars, len(ratios) - 1)
        if lookback <= 0:
            return 0.0

        diffs: list[float] = []
        for idx in range(len(ratios) - lookback, len(ratios)):
            prev = ratios[idx - 1]
            curr = ratios[idx]
            if prev <= 0 or curr <= 0:
                continue
            diffs.append(math.log(curr) - math.log(prev))

        if not diffs:
            return 0.0

        direction = 1 if net_change > 0 else -1 if net_change < 0 else 0
        if direction == 0:
            quiet_steps = sum(1 for diff in diffs if abs(diff) < 0.01)
            return round(quiet_steps / len(diffs), 2)

        aligned_steps = sum(1 for diff in diffs if direction * diff > 0)
        neutral_steps = sum(1 for diff in diffs if abs(diff) <= 0.004)
        confidence = (aligned_steps + 0.5 * neutral_steps) / len(diffs)
        return round(max(0.0, min(confidence, 1.0)), 2)

    @staticmethod
    def _classify_trend_acceleration(short_change: float, medium_change: float) -> str:
        """Describe whether the short-term trend is accelerating, slowing, or reversing."""
        if abs(short_change) < 0.005 and abs(medium_change) < 0.005:
            return "stable"
        if (
            short_change * medium_change < 0
            and abs(short_change) >= 0.01
            and abs(medium_change) >= 0.01
        ):
            return "reversing"
        if abs(short_change) >= max(0.01, abs(medium_change) * 1.35):
            return "accelerating"
        if abs(medium_change) >= max(0.01, abs(short_change) * 1.35) and abs(short_change) < abs(
            medium_change
        ):
            return "slowing"
        return "stable"

    @staticmethod
    def _describe_positioning(percentile: float | None, ratio: float) -> str:
        """Translate ratio location into a compact crowding label."""
        description = "balanced"
        if percentile is not None and percentile >= 90.0:
            description = "extreme long crowding"
        elif percentile is not None and percentile >= 75.0:
            description = "elevated long positioning"
        elif percentile is not None and percentile <= 10.0:
            description = "extreme short crowding"
        elif percentile is not None and percentile <= 25.0:
            description = "elevated short positioning"
        elif ratio >= 1.02:
            description = "mildly long-leaning"
        elif ratio <= 0.98:
            description = "mildly short-leaning"
        return description

    @staticmethod
    def _format_trend_summary(trend: str, change: float | None, bars: int) -> str:
        """Format a compact human-readable trend summary."""
        if bars <= 0 or change is None:
            return "insufficient trend data"
        return f"{trend.replace('_', ' ')} ({change:+.1%} / {bars} bars)"

    @staticmethod
    def _format_trend_consistency(confidence: float) -> str:
        """Map trend confidence to a compact text label."""
        if confidence >= 0.75:
            return "high trend consistency"
        if confidence >= 0.45:
            return "moderate trend consistency"
        return "low trend consistency"

    @staticmethod
    def _trend_direction(trend: str | None) -> int:
        """Convert a trend label to directional sign for divergence checks."""
        if trend in {"rising", "rising_fast"}:
            return 1
        if trend in {"falling", "falling_fast"}:
            return -1
        return 0

    @staticmethod
    def _bias_from_analysis(analysis: dict[str, Any]) -> str:
        """Infer a coarse positioning bias from level plus trend."""
        score = 0
        ratio = float(analysis.get("current_ratio") or 1.0)
        percentile = analysis.get("percentile")

        if ratio >= 1.03:
            score += 1
        elif ratio <= 0.97:
            score -= 1

        if percentile is not None:
            if percentile >= 65.0:
                score += 1
            elif percentile <= 35.0:
                score -= 1

        medium_trend = analysis.get("medium_term_trend")
        if medium_trend in {"rising", "rising_fast"}:
            score += 1
        elif medium_trend in {"falling", "falling_fast"}:
            score -= 1

        short_trend = analysis.get("short_term_trend")
        if short_trend == "rising_fast":
            score += 1
        elif short_trend == "falling_fast":
            score -= 1

        if score >= 2:
            return "long"
        if score <= -2:
            return "short"
        return "neutral"

    def _build_indicator_prompt_summary(self, label: str, analysis: dict[str, Any]) -> str:
        """Render one compact per-indicator positioning line for the LLM prompt."""
        positioning = self._describe_positioning(
            analysis.get("percentile"), analysis["current_ratio"]
        )
        parts = [label, positioning]
        short_bars = int(analysis.get("short_term_bars") or 0)
        if short_bars > 0:
            trend = self._format_trend_summary(
                analysis.get("short_term_trend", "stable"),
                analysis.get("short_term_change"),
                short_bars,
            )
            parts.append(f"short-term {trend}")
        acceleration = analysis.get("trend_acceleration")
        if acceleration and acceleration != "stable":
            parts.append(f"trend {acceleration}")
        if analysis.get("reversal_flag"):
            parts.append("recent reversal risk")
        if not analysis.get("history_available"):
            parts.append("historical percentile unavailable")
        return "- " + " | ".join(parts)

    @staticmethod
    def _bias_relationship(
        first: str | None,
        second: str | None,
        aligned: str,
        split: str,
    ) -> str | None:
        """Describe alignment between two non-neutral positioning biases."""
        if not first or not second or "neutral" in {first, second}:
            return None
        return aligned.format(bias=first) if first == second else split

    @staticmethod
    def _fallback_combined_read(biases: dict[str, str]) -> str:
        """Return a default interpretation when no stronger relationship exists."""
        long_count = sum(bias == "long" for bias in biases.values())
        short_count = sum(bias == "short" for bias in biases.values())
        if long_count >= 2 and short_count == 0:
            return (
                "Most positioning indicators lean long, but price confirmation is still needed "
                "to distinguish trend support from late crowding."
            )
        if short_count >= 2 and long_count == 0:
            return (
                "Most positioning indicators lean short, but price confirmation is still needed "
                "to distinguish trend pressure from squeeze fuel."
            )
        return (
            "Positioning signals are mixed, so long/short ratios are a weak prior rather than "
            "a standalone trade trigger."
        )

    def _build_combined_read(self, analyses: dict[str, dict[str, Any]]) -> list[str]:
        """Build a short multi-indicator interpretation block for the LLM prompt."""
        if not analyses:
            return []
        biases = {key: self._bias_from_analysis(value) for key, value in analyses.items()}
        relationships = (
            self._bias_relationship(
                biases.get("global_account"),
                biases.get("top_trader_account"),
                "Retail and top-trader account sentiment both lean {bias}.",
                "Retail and top-trader sentiment disagree, weakening the positioning prior.",
            ),
            self._bias_relationship(
                biases.get("top_trader_account"),
                biases.get("top_trader_position"),
                "Smart-money account direction and position sizing align to the {bias} side.",
                "Smart-money direction and sizing diverge, suggesting hedging or instability.",
            ),
        )
        lines = [relationship for relationship in relationships if relationship]
        reversal_labels = [
            self._RATIO_TYPE_LABELS.get(ratio_type, ratio_type)
            for ratio_type, analysis in analyses.items()
            if analysis.get("reversal_flag")
        ]
        if reversal_labels:
            lines.append(f"Recent reversal risk is showing in {', '.join(reversal_labels)}.")
        for note in self.detect_cross_indicator_divergence(analyses):
            if note not in lines:
                lines.append(note)
        if not lines:
            lines.append(self._fallback_combined_read(biases))
        return [f"- {line}" for line in lines[:3]]

    def _retail_divergence(
        self, global_analysis: dict[str, Any], top_account: dict[str, Any]
    ) -> str | None:
        """Detect retail versus top-account divergence."""
        if global_analysis["is_extreme_high"] and top_account["is_extreme_low"]:
            return "Retail is extremely long while top-trader accounts are extremely short."
        if global_analysis["is_extreme_low"] and top_account["is_extreme_high"]:
            return "Retail is extremely short while top-trader accounts are extremely long."
        global_direction = self._trend_direction(global_analysis.get("short_term_trend"))
        top_direction = self._trend_direction(top_account.get("short_term_trend"))
        if global_direction and top_direction and global_direction != top_direction:
            return "Retail and top-trader short-term sentiment move in opposite directions."
        return None

    def _smart_money_divergence(
        self, top_account: dict[str, Any], top_position: dict[str, Any]
    ) -> str | None:
        """Detect top-account versus top-position divergence."""
        if top_account["is_extreme_high"] and top_position["is_extreme_low"]:
            return "Top-trader accounts are heavily long while position sizing is heavily short."
        if top_account["is_extreme_low"] and top_position["is_extreme_high"]:
            return "Top-trader accounts are heavily short while position sizing is heavily long."
        account_bias = self._bias_from_analysis(top_account)
        position_bias = self._bias_from_analysis(top_position)
        if "neutral" not in {account_bias, position_bias} and account_bias != position_bias:
            return "Top-trader direction and sizing are not aligned; conviction looks uneven."
        return None

    def detect_cross_indicator_divergence(self, analyses: dict[str, dict[str, Any]]) -> list[str]:
        """Detect concise cross-indicator disagreement notes for prompt summaries."""
        notes: list[str] = []
        global_analysis = analyses.get("global_account")
        top_account = analyses.get("top_trader_account")
        top_position = analyses.get("top_trader_position")
        if global_analysis and top_account:
            note = self._retail_divergence(global_analysis, top_account)
            if note:
                notes.append(note)
        if top_account and top_position:
            note = self._smart_money_divergence(top_account, top_position)
            if note:
                notes.append(note)
        return notes[:2]

    @staticmethod
    def _ordered_ratio_types() -> tuple[tuple[str, str], ...]:
        """Return data keys in prompt display order."""
        return (
            ("global_account_ratio", "global_account"),
            ("top_trader_account_ratio", "top_trader_account"),
            ("top_trader_position_ratio", "top_trader_position"),
        )

    def _latest_ratio_timestamp(self, data: dict[str, Any]) -> datetime | None:
        """Find the newest timestamp across ratio sources."""
        timestamps = []
        for data_key, _ in self._ordered_ratio_types():
            items = data.get(data_key, [])
            if items:
                parsed = self._parse_source_timestamp(items[-1].get("timestamp"))
                if parsed is not None:
                    timestamps.append(parsed)
        return max(timestamps) if timestamps else None

    def _build_prompt_analyses(
        self, pair: str, data: dict[str, Any], short_window: int
    ) -> dict[str, dict[str, Any]]:
        """Analyze each available ratio source for prompt rendering."""
        analyses: dict[str, dict[str, Any]] = {}
        for data_key, ratio_type in self._ordered_ratio_types():
            analysis = self.analyze_historical(
                pair,
                ratio_type=ratio_type,
                latest_items=data.get(data_key, []),
                short_window=short_window,
            )
            if analysis is not None:
                analyses[ratio_type] = analysis
        return analyses

    def _build_indicator_lines(self, analyses: dict[str, dict[str, Any]]) -> list[str]:
        """Render analyzed sources in stable display order."""
        lines = []
        for _, ratio_type in self._ordered_ratio_types():
            analysis = analyses.get(ratio_type)
            if analysis is not None:
                label = self._RATIO_TYPE_LABELS.get(ratio_type, ratio_type)
                lines.append(self._build_indicator_prompt_summary(label, analysis))
        return lines

    def build_context_text(
        self,
        pair: str,
        data: dict[str, Any] | None = None,
        **options: Any,
    ) -> str:
        """Format long/short data as a compact positioning summary for LLM input."""
        period = options.pop("period", "5m")
        limit = options.pop("limit", 10)
        display_limit = options.pop("display_limit", 3)
        if options:
            unexpected = ", ".join(sorted(options))
            raise TypeError(f"Unexpected context options: {unexpected}")
        if data is None:
            data = self.collect(pair, period, limit)
        lines = [f"### Positioning / Crowding - {pair} ({period})"]
        latest_timestamp = self._latest_ratio_timestamp(data)
        if latest_timestamp is not None:
            lines.append(f"**As of**: {latest_timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("")
        analyses = self._build_prompt_analyses(pair, data, max(3, display_limit))
        if not analyses:
            return "\n".join([*lines, "(No long/short ratio data available)", ""])
        combined_lines = self._build_combined_read(analyses)
        if combined_lines:
            lines.extend(["**Combined read**", *combined_lines])
        indicator_lines = self._build_indicator_lines(analyses)
        if indicator_lines:
            lines.extend(["", "**Key sources**", *indicator_lines[:3]])
        return "\n".join([*lines, ""])
