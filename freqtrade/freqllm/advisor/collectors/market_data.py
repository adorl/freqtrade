"""Collect futures market data and build concise LLM context."""

import importlib
import json
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

import ccxt
import pandas as pd

from freqtrade.exceptions import ExchangeError


logger = logging.getLogger(__name__)
_PERIOD_SECONDS = dict(
    zip(
        ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"),
        (300, 900, 1800, 3600, 7200, 14400, 21600, 43200, 86400),
        strict=True,
    )
)
_API_ERRORS = (ccxt.BaseError, AttributeError, KeyError, TypeError, ValueError, ExchangeError)
_DATA_ERRORS = (AttributeError, KeyError, TypeError, ValueError)


def _clamp_limit(value: Any, minimum: int = 1, maximum: int = 500) -> int:
    """Clamp an API limit to the exchange-accepted range."""
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = minimum
    return max(minimum, min(normalized, maximum))


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: Any) -> datetime | None:
    parsed: datetime | None = None
    try:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, (int, float)):
            parsed = datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
        elif isinstance(value, str):
            parsed = datetime.fromisoformat(value)
    except (OSError, TypeError, ValueError):
        return None
    if parsed is not None and parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC) if parsed is not None else None


class _ExchangeAccess:
    """Resolve public exchange wrappers first and isolate the ccxt compatibility fallback."""

    def __init__(self, exchange: Any) -> None:
        self.exchange = exchange

    def client(self) -> Any:
        """Return the configured public API client."""
        public_client = getattr(self.exchange, "api", None)
        return public_client if public_client is not None else vars(self.exchange).get("_api")

    def method(self, name: str) -> Callable[..., Any] | None:
        """Resolve an exchange method by name."""
        public_method = getattr(self.exchange, name, None)
        if callable(public_method):
            return public_method
        method = getattr(self.client(), name, None)
        return method if callable(method) else None

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a required exchange method."""
        method = self.method(name)
        if method is None:
            raise AttributeError(f"Exchange endpoint {name!r} is unavailable")
        return method(*args, **kwargs)

    def markets(self) -> dict[str, Any]:
        """Return loaded exchange markets."""
        markets = getattr(self.exchange, "markets", {})
        return markets if isinstance(markets, dict) else {}


class _FeatureRecordBuilder:
    """Build normalized persistence records."""

    def __init__(self, pair: str, period: str, collection_ts: datetime | None) -> None:
        self.pair, self.period, self.collection_ts = pair, period, collection_ts

    def _source_time(self, item: dict[str, Any]) -> datetime | None:
        return _parse_timestamp(item.get("timestamp") or item.get("datetime")) or self.collection_ts

    def record(self, feature_type: str, item: dict[str, Any], **values: Any) -> dict[str, Any]:
        """Create one normalized feature record."""
        period = values.pop("period", self.period)
        return {
            "pair": self.pair,
            "feature_type": feature_type,
            "period": period,
            **values,
            "raw_payload": json.dumps(item, ensure_ascii=False, default=str),
            "source_timestamp": self._source_time(item),
        }

    def _funding_records(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        item = data.get("funding_rate") or {}
        if not item:
            return []
        keys = ("funding_rate", "basis", "basis_pct", "mark_price", "index_price")
        values = {key: item.get(key) for key in keys}
        values["next_funding_time"] = _parse_timestamp(item.get("next_funding_time"))
        return [self.record("funding_rate", item, **values)]

    def _basis_records(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        basis = data.get("basis") or {}
        items = basis.get("history") or ([basis] if basis else [])
        keys = ("basis", "basis_pct", "mark_price", "index_price")
        return [
            self.record("basis", item, **{key: item.get(key) for key in keys}) for item in items
        ]

    def _open_interest_records(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        current = data.get("open_interest") or {}
        keys = ("open_interest", "open_interest_value")
        records = []
        if any(current.get(key) is not None for key in keys):
            records.append(
                self.record(
                    "open_interest_current", current, **{key: current.get(key) for key in keys}
                )
            )
        records.extend(
            self.record("open_interest_history", item, **{key: item.get(key) for key in keys})
            for item in current.get("history", []) or []
        )
        return records

    def _market_flow_records(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        flow = data.get("market_flow") or {}
        records = []
        for item in flow.get("taker_buy_sell_volume", []) or []:
            buy, sell = (
                _safe_float(item.get("buy_vol")) or 0.0,
                _safe_float(item.get("sell_vol")) or 0.0,
            )
            total = buy + sell
            records.append(
                self.record(
                    "taker_buy_sell_volume",
                    item,
                    buy_vol=buy,
                    sell_vol=sell,
                    buy_sell_ratio=item.get("buy_sell_ratio"),
                    buy_sell_imbalance=(buy - sell) / total if total else None,
                )
            )
        return records

    def _reference_records(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        reference = data.get("reference_price_klines") or {}
        period, records = reference.get("timeframe") or self.period, []
        for item in reference.get("index_price_klines", []) or []:
            values = {"period": period, "index_price": item.get("close")}
            records.append(self.record("index_price_kline", item, **values))
        for item in reference.get("premium_index_klines", []) or []:
            close = _safe_float(item.get("close"))
            records.append(
                self.record(
                    "premium_index_kline",
                    item,
                    period=period,
                    basis_pct=close * 100 if close is not None else None,
                )
            )
        return records

    def _spot_flow_records(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        flow = data.get("spot_trade_flow") or {}
        keys = {
            "buy_vol": "buy_quote_vol",
            "sell_vol": "sell_quote_vol",
            "buy_sell_ratio": "buy_sell_ratio",
            "buy_sell_imbalance": "buy_sell_imbalance",
            "event_count": "trade_count",
        }
        return [
            self.record(
                "spot_trade_flow",
                item,
                period=flow.get("period") or self.period,
                **{target: item.get(source) for target, source in keys.items()},
            )
            for item in flow.get("spot_trade_flow", []) or []
        ]

    def _leverage_records(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        item = data.get("leverage_stats") or {}
        return (
            [
                self.record(
                    "max_leverage", item, period="snapshot", max_leverage=item.get("max_leverage")
                )
            ]
            if item.get("max_leverage") is not None
            else []
        )

    def build(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Build all persistence records."""
        groups = (
            self._funding_records,
            self._basis_records,
            self._open_interest_records,
            self._market_flow_records,
            self._reference_records,
            self._spot_flow_records,
            self._leverage_records,
        )
        return [record for group in groups for record in group(data)]


class MarketDataCollector:
    """Collect market, derivative, order-flow, and leverage context."""

    def __init__(self, data_provider, config, exchange=None, database=None):
        self.dp = data_provider
        self.config = config
        self._db = database
        self._exchange = exchange or getattr(data_provider, "exchange", None)
        if self._exchange is None:
            self._exchange = vars(data_provider).get("_exchange")
        self._access = _ExchangeAccess(self._exchange) if self._exchange is not None else None
        self._cache: dict[str, dict[str, Any]] = {}

    def _calc_needed_feature_limit(
        self, pair: str, feature_type: str, default_limit: int, period: str | None = None
    ) -> int:
        normalized = _clamp_limit(default_limit, maximum=max(1, int(default_limit or 1)))
        if self._db is None:
            return normalized
        try:
            bucket = period or self.config.market_data.long_short_ratio_period
            latest = self._db.get_latest_market_feature_timestamp(
                pair, feature_type, period=bucket, feature_key=""
            )
            if latest is None:
                return normalized
            elapsed = max(0.0, (datetime.now(UTC) - latest).total_seconds())
            needed = int(elapsed / _PERIOD_SECONDS.get(bucket, 300)) + 1
            return max(1, min(needed, normalized))
        except _DATA_ERRORS as exc:
            logger.debug("[%s] Failed to calculate %s limit: %s", pair, feature_type, exc)
            return normalized

    def collect(self, pair: str) -> dict[str, Any]:
        """Collect all enabled market features for one pair."""
        data: dict[str, Any] = {"pair": pair, "timestamp": datetime.now(UTC).isoformat()}
        tasks: dict[str, Callable[[], Any]] = {
            "klines": lambda: self._collect_klines(pair),
            "ticker": lambda: self._collect_ticker(pair),
            "orderbook": lambda: self._collect_orderbook(pair),
            "funding_rate": lambda: self._collect_funding_rate(pair),
        }
        if self._access is not None:
            optional = (
                ("open_interest", "open_interest_enabled", self._collect_open_interest),
                (
                    "reference_price_klines",
                    "reference_price_klines_enabled",
                    self._collect_reference_price_klines,
                ),
                ("spot_trade_flow", "spot_trade_flow_enabled", self._collect_spot_trade_flow),
                ("leverage_stats", "leverage_stats_enabled", self._collect_leverage_stats),
            )
            for key, flag, collector in optional:
                if getattr(self.config.market_data, flag):
                    tasks[key] = lambda collector=collector: collector(pair)
            tasks["market_flow"] = lambda: self._collect_market_flow(pair)
        with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as executor:
            futures = {executor.submit(task): key for key, task in tasks.items()}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    data[key] = future.result(timeout=30)
                except _API_ERRORS as exc:
                    logger.warning("[%s] Collection failed for %s: %s", pair, key, exc)
                    data[key] = None
        data["basis"] = self._collect_basis(pair)
        self._cache[pair] = data
        self._persist_collected_data(pair, data)
        return data

    def _collect_klines(self, pair: str) -> dict[str, Any]:
        result = {}
        for timeframe in self.config.market_data.kline_timeframes:
            try:
                frame = self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe)
                if frame is None or frame.empty:
                    logger.warning("[%s] Empty kline data for %s", pair, timeframe)
                    cached = self._cache.get(pair, {}).get("klines", {}).get(timeframe)
                    if cached:
                        result[timeframe] = cached
                    continue
                frame = self._add_indicators(frame.tail(self.config.market_data.kline_limit).copy())
                result[timeframe] = self._df_to_list(frame)
            except _DATA_ERRORS as exc:
                logger.warning("[%s] Failed to collect %s klines: %s", pair, timeframe, exc)
                cached = self._cache.get(pair, {}).get("klines", {}).get(timeframe)
                if cached:
                    result[timeframe] = cached
        return result

    @staticmethod
    def _add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
        try:
            ta = importlib.import_module("pandas_ta")
            frame["rsi"] = ta.rsi(frame["close"], length=14)
            macd = ta.macd(frame["close"], fast=12, slow=26, signal=9)
            if macd is not None and not macd.empty:
                columns = macd.columns.tolist()
                frame[["macd", "macd_signal", "macd_hist"]] = macd[columns[:3]]
            bands = ta.bbands(frame["close"], length=20, std=2)
            if bands is not None and not bands.empty:
                columns = bands.columns.tolist()
                lower = next((column for column in columns if "BBL_" in column), columns[0])
                middle = next((column for column in columns if "BBM_" in column), columns[1])
                upper = next((column for column in columns if "BBU_" in column), columns[2])
                frame[["bb_upper", "bb_mid", "bb_lower"]] = bands[[upper, middle, lower]]
            for length in (20, 50, 200):
                frame[f"ema_{length}"] = ta.ema(frame["close"], length=length)
        except (AttributeError, ImportError, IndexError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to compute technical indicators: %s", exc)
        return frame

    @staticmethod
    def _df_to_list(frame: pd.DataFrame) -> list[dict[str, Any]]:
        records = []
        for record in frame.to_dict("records"):
            records.append(
                {
                    key: None
                    if pd.isna(value)
                    else value.isoformat()
                    if isinstance(value, pd.Timestamp)
                    else round(value, 8)
                    if isinstance(value, float)
                    else value
                    for key, value in record.items()
                }
            )
        return records

    def _cached(self, pair: str, key: str) -> Any:
        return self._cache.get(pair, {}).get(key)

    def _collect_ticker(self, pair: str) -> dict[str, Any] | None:
        try:
            ticker = self.dp.ticker(pair)
            if ticker:
                mapping = {
                    "timestamp": "timestamp",
                    "datetime": "datetime",
                    "last": "last",
                    "bid": "bid",
                    "ask": "ask",
                    "high": "high",
                    "low": "low",
                    "volume": "baseVolume",
                    "quote_volume": "quoteVolume",
                    "change_pct": "percentage",
                    "mark_price": "markPrice",
                    "index_price": "indexPrice",
                }
                return {target: ticker.get(source) for target, source in mapping.items()}
        except _API_ERRORS as exc:
            logger.warning("[%s] Failed to collect ticker: %s", pair, exc)
            return self._cached(pair, "ticker")
        return None

    @staticmethod
    def _orderbook_metrics(bids: list, asks: list) -> dict[str, Any]:
        best_bid_price, best_bid_qty = bids[0][:2]
        best_ask_price, best_ask_qty = asks[0][:2]
        spread = best_ask_price - best_bid_price
        midpoint = (best_bid_price + best_ask_price) / 2
        bid_qty = sum(level[1] for level in bids)
        ask_qty = sum(level[1] for level in asks)
        return {
            "best_bid": [best_bid_price, best_bid_qty],
            "best_ask": [best_ask_price, best_ask_qty],
            "spread": round(spread, 8),
            "spread_pct": round(spread / midpoint * 100 if midpoint else 0, 6),
            "bid_ask_imbalance": round(bid_qty / ask_qty if ask_qty else float("inf"), 4),
            "bid_total_qty": round(bid_qty, 4),
            "ask_total_qty": round(ask_qty, 4),
            "bid_depth_value": round(sum(price * qty for price, qty, *_ in bids), 2),
            "ask_depth_value": round(sum(price * qty for price, qty, *_ in asks), 2),
            "price_impact_buy_pct": MarketDataCollector._calc_price_impact(asks, 10_000),
            "price_impact_sell_pct": MarketDataCollector._calc_price_impact(bids, 10_000),
            "depth_levels": len(bids),
        }

    def _collect_orderbook(self, pair: str) -> dict[str, Any] | None:
        try:
            book = self.dp.orderbook(pair, maximum=self.config.market_data.orderbook_depth)
            bids, asks = (book or {}).get("bids", []), (book or {}).get("asks", [])
            return self._orderbook_metrics(bids, asks) if bids and asks else None
        except _API_ERRORS as exc:
            logger.warning("[%s] Failed to collect order book: %s", pair, exc)
            return self._cached(pair, "orderbook")

    @staticmethod
    def _calc_price_impact(levels: list, notional: float) -> float | None:
        if not levels:
            return None
        filled_value = filled_qty = 0.0
        reference = levels[0][0]
        for price, qty, *_ in levels:
            level_value = price * qty
            if filled_value + level_value >= notional:
                filled_qty += (notional - filled_value) / price if price > 0 else 0
                filled_value = notional
                break
            filled_value += level_value
            filled_qty += qty
        if filled_value < notional * 0.5 or filled_qty <= 0 or reference <= 0:
            return None
        return round(abs(filled_value / filled_qty - reference) / reference * 100, 6)

    def _collect_funding_rate(self, pair: str) -> dict[str, Any] | None:
        try:
            funding = self.dp.funding_rate(pair)
            if not funding:
                return None
            result = {
                "timestamp": funding.get("timestamp"),
                "datetime": funding.get("datetime"),
                "funding_rate": funding.get("fundingRate"),
                "next_funding_time": next(
                    (
                        funding.get(key)
                        for key in (
                            "fundingTimestamp",
                            "fundingDatetime",
                            "nextFundingTime",
                            "nextFundingTimestamp",
                            "nextFundingDatetime",
                        )
                        if funding.get(key) is not None
                    ),
                    None,
                ),
                "mark_price": funding.get("markPrice"),
                "index_price": funding.get("indexPrice"),
            }
            mark, index = _safe_float(result["mark_price"]), _safe_float(result["index_price"])
            if mark is not None and index not in (None, 0.0):
                result.update(basis=mark - index, basis_pct=(mark - index) / index * 100)
            return result
        except _API_ERRORS as exc:
            logger.warning("[%s] Failed to collect funding rate: %s", pair, exc)
            return self._cached(pair, "funding_rate")

    def _collect_basis(self, pair: str) -> dict[str, Any] | None:
        if self._access is None:
            return self._cached(pair, "basis")
        try:
            method = self._access.method("fapiDataGetBasis")
            if method is None:
                return self._cached(pair, "basis")
            period = self.config.market_data.long_short_ratio_period
            limit = self._calc_needed_feature_limit(
                pair, "basis", self.config.market_data.long_short_ratio_limit, period
            )
            rows = method(
                {
                    "pair": self._to_base_symbol(pair),
                    "contractType": "PERPETUAL",
                    "period": period,
                    "limit": _clamp_limit(limit),
                }
            )
            history = [item for row in rows or [] if (item := self._basis_item_to_dict(row))]
            if history:
                return {**history[-1], "history": history}
        except _API_ERRORS as exc:
            logger.debug("[%s] Basis history failed: %s", pair, exc)
        return self._cached(pair, "basis")

    def _basis_item_to_dict(self, item: dict[str, Any]) -> dict[str, Any] | None:
        index = _safe_float(item.get("indexPrice"))
        futures = _safe_float(item.get("futuresPrice"))
        basis = _safe_float(item.get("basis"))
        rate = _safe_float(item.get("basisRate"))
        annual = _safe_float(item.get("annualizedBasisRate"))
        if basis is None and futures is not None and index is not None:
            basis = futures - index
        if rate is None and basis is not None and index not in (None, 0.0):
            rate = basis / index
        if index is None and futures is None and basis is None:
            return None
        timestamp = item.get("timestamp")
        parsed = _parse_timestamp(timestamp)
        return {
            "timestamp": timestamp,
            "datetime": parsed.isoformat() if parsed else None,
            "index_price": index,
            "futures_price": futures,
            "basis": basis,
            "basis_rate": rate,
            "basis_pct": rate * 100.0 if rate is not None else None,
            "annualized_basis_rate": annual,
            "annualized_basis_pct": annual * 100.0 if annual is not None else None,
            "source": "binance_basis_history",
        }

    def _collect_open_interest(self, pair: str) -> dict[str, Any] | None:
        if self._access is None:
            return None
        result: dict[str, Any] = {}
        symbol = self._to_ccxt_symbol(pair)
        try:
            current = self._access.call("fetch_open_interest", symbol)
            if current:
                result.update(
                    open_interest=current.get("openInterest") or current.get("openInterestAmount"),
                    open_interest_value=current.get("openInterestValue"),
                    timestamp=current.get("timestamp"),
                )
        except _API_ERRORS as exc:
            logger.debug("[%s] Current open interest failed: %s", pair, exc)
        try:
            limit = self._calc_needed_feature_limit(
                pair, "open_interest_history", self.config.market_data.open_interest_limit
            )
            rows = self._access.call(
                "fetch_open_interest_history",
                symbol,
                timeframe=self.config.market_data.long_short_ratio_period,
                limit=limit,
            )
            result["history"] = [
                {
                    "timestamp": item.get("timestamp"),
                    "open_interest": item.get("openInterest") or item.get("openInterestAmount"),
                    "open_interest_value": item.get("openInterestValue"),
                }
                for item in rows or []
            ]
        except _API_ERRORS as exc:
            logger.debug("[%s] Open interest history failed: %s", pair, exc)
        return result or self._cached(pair, "open_interest")

    def _collect_market_flow(self, pair: str) -> dict[str, Any] | None:
        if self._access is None:
            return None
        names = (
            "fapiDataGetTakerlongshortRatio",
            "fapiDataGetTakerLongShortRatio",
            "fapiDataGetTakerBuySellVol",
        )
        method = next(
            (self._access.method(name) for name in names if self._access.method(name)), None
        )
        if method is None:
            return self._cached(pair, "market_flow")
        try:
            limit = self._calc_needed_feature_limit(
                pair, "taker_buy_sell_volume", self.config.market_data.open_interest_limit
            )
            rows = method(
                {
                    "symbol": self._to_base_symbol(pair),
                    "period": self.config.market_data.long_short_ratio_period,
                    "limit": _clamp_limit(limit),
                }
            )
            items = [
                {
                    "timestamp": item.get("timestamp"),
                    "buy_vol": _safe_float(item.get("buyVol")) or 0.0,
                    "sell_vol": _safe_float(item.get("sellVol")) or 0.0,
                    "buy_sell_ratio": _safe_float(item.get("buySellRatio")) or 0.0,
                }
                for item in rows or []
            ]
            return {"source": "binance_history", "taker_buy_sell_volume": items} if items else None
        except _API_ERRORS as exc:
            logger.debug("[%s] Taker flow failed: %s", pair, exc)
            return self._cached(pair, "market_flow")

    def _ohlcv_to_records(self, rows: list[list[Any]]) -> list[dict[str, Any]]:
        records = []
        for row in rows or []:
            if len(row) < 6:
                continue
            parsed = _parse_timestamp(row[0])
            records.append(
                {
                    "timestamp": row[0],
                    "datetime": parsed.isoformat() if parsed else None,
                    "open": _safe_float(row[1]),
                    "high": _safe_float(row[2]),
                    "low": _safe_float(row[3]),
                    "close": _safe_float(row[4]),
                    "volume": _safe_float(row[5]),
                }
            )
        return records

    def _collect_reference_price_klines(self, pair: str) -> dict[str, Any] | None:
        if self._access is None:
            return None
        timeframe = self.config.market_data.long_short_ratio_period
        result: dict[str, Any] = {"timeframe": timeframe}
        endpoints = (
            ("fetch_index_ohlcv", "index_price_kline", "index_price_klines"),
            ("fetch_premium_index_ohlcv", "premium_index_kline", "premium_index_klines"),
        )
        for endpoint, feature_type, result_key in endpoints:
            try:
                limit = self._calc_needed_feature_limit(
                    pair,
                    feature_type,
                    self.config.market_data.reference_price_kline_limit,
                    timeframe,
                )
                rows = self._access.call(
                    endpoint, self._to_ccxt_symbol(pair), timeframe=timeframe, limit=limit
                )
                candles = self._ohlcv_to_records(rows)
                if candles:
                    result[result_key] = candles
            except _API_ERRORS as exc:
                logger.debug("[%s] %s failed: %s", pair, endpoint, exc)
        return result if len(result) > 1 else self._cached(pair, "reference_price_klines")

    def _normalize_spot_trades(self, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for item in trades or []:
            is_binance = "p" in item or "q" in item
            price = _safe_float(item.get("p") if is_binance else item.get("price"))
            amount = _safe_float(item.get("q") if is_binance else item.get("amount"))
            timestamp = item.get("T") if is_binance else item.get("timestamp")
            side = ("sell" if item.get("m") else "buy") if is_binance else item.get("side")
            cost = _safe_float(item.get("cost")) or (
                price * amount if price is not None and amount is not None else None
            )
            if timestamp is not None and side in {"buy", "sell"} and amount is not None and cost:
                normalized.append(
                    {
                        "timestamp": int(timestamp),
                        "side": side,
                        "base_amount": amount,
                        "quote_amount": cost,
                    }
                )
        return normalized

    def _normalize_binance_spot_agg_trades(
        self, trades: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return self._normalize_spot_trades(trades)

    def _normalize_ccxt_trades(self, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._normalize_spot_trades(trades)

    def _aggregate_spot_trade_buckets(
        self, trades: list[dict[str, Any]], period: str
    ) -> list[dict[str, Any]]:
        bucket_ms = _PERIOD_SECONDS.get(period, 300) * 1000
        buckets: dict[int, dict[str, Any]] = {}
        for trade in trades or []:
            timestamp = trade.get("timestamp")
            side = trade.get("side")
            base, quote = (
                _safe_float(trade.get("base_amount")),
                _safe_float(trade.get("quote_amount")),
            )
            if timestamp is None or side not in {"buy", "sell"} or base is None or quote is None:
                continue
            bucket_time = int(timestamp) - int(timestamp) % bucket_ms
            bucket = buckets.setdefault(
                bucket_time,
                {
                    "timestamp": bucket_time,
                    "datetime": _parse_timestamp(bucket_time).isoformat(),
                    "buy_base_vol": 0.0,
                    "sell_base_vol": 0.0,
                    "buy_quote_vol": 0.0,
                    "sell_quote_vol": 0.0,
                    "buy_trade_count": 0,
                    "sell_trade_count": 0,
                },
            )
            bucket[f"{side}_base_vol"] += base
            bucket[f"{side}_quote_vol"] += quote
            bucket[f"{side}_trade_count"] += 1
        return [self._finalize_spot_bucket(buckets[key]) for key in sorted(buckets)]

    @staticmethod
    def _finalize_spot_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
        buy, sell = bucket["buy_quote_vol"], bucket["sell_quote_vol"]
        total = buy + sell
        count = bucket["buy_trade_count"] + bucket["sell_trade_count"]
        return {
            **bucket,
            "buy_base_vol": round(bucket["buy_base_vol"], 8),
            "sell_base_vol": round(bucket["sell_base_vol"], 8),
            "buy_quote_vol": round(buy, 8),
            "sell_quote_vol": round(sell, 8),
            "trade_count": count,
            "avg_trade_notional": round(total / count, 8) if count else None,
            "buy_sell_ratio": round(buy / sell, 8) if sell else None,
            "buy_sell_imbalance": round((buy - sell) / total, 8) if total else None,
        }

    def _collect_spot_trade_flow(self, pair: str) -> dict[str, Any] | None:
        if self._access is None:
            return None
        period = self.config.market_data.long_short_ratio_period
        limit = _clamp_limit(self.config.market_data.spot_trade_limit, maximum=1000)
        try:
            raw_method = self._access.method("publicGetAggTrades")
            rows = (
                raw_method({"symbol": self._to_base_symbol(pair), "limit": limit})
                if raw_method
                else []
            )
            normalized = self._normalize_spot_trades(rows)
            if not normalized:
                rows = self._access.call("fetch_trades", self._to_spot_symbol(pair), limit=limit)
                normalized = self._normalize_spot_trades(rows)
            buckets = self._aggregate_spot_trade_buckets(normalized, period)
            if buckets:
                return {
                    "source": "binance_spot_agg_trades",
                    "period": period,
                    "symbol": self._to_spot_symbol(pair),
                    "spot_trade_flow": buckets,
                }
        except _API_ERRORS as exc:
            logger.debug("[%s] Spot trade flow failed: %s", pair, exc)
        return self._cached(pair, "spot_trade_flow")

    def _collect_leverage_stats(self, pair: str) -> dict[str, Any] | None:
        if self._access is None:
            return None
        try:
            market = self._access.markets().get(self._to_ccxt_symbol(pair), {})
            maximum = market.get("limits", {}).get("leverage", {}).get("max")
            info = market.get("info", {})
            if not maximum:
                maximum = next(
                    (
                        info.get(key)
                        for key in ("maxLeverage", "leverageFilter", "maxLev")
                        if info.get(key)
                    ),
                    None,
                )
            return (
                {"max_leverage": maximum}
                if maximum is not None
                else self._cached(pair, "leverage_stats")
            )
        except _DATA_ERRORS as exc:
            logger.debug("[%s] Max leverage failed: %s", pair, exc)
            return self._cached(pair, "leverage_stats")

    def _persist_collected_data(self, pair: str, data: dict[str, Any]) -> None:
        if self._db is None:
            return
        try:
            builder = _FeatureRecordBuilder(
                pair,
                self.config.market_data.long_short_ratio_period,
                _parse_timestamp(data.get("timestamp")),
            )
            records = builder.build(data)
            if records:
                thread = threading.Thread(
                    target=self._db.record_market_features_batch, args=(records,), daemon=True
                )
                thread.start()
        except _DATA_ERRORS as exc:
            logger.debug("[%s] Failed to persist market data: %s", pair, exc)

    @staticmethod
    def _to_ccxt_symbol(pair: str) -> str:
        return pair

    @staticmethod
    def _to_base_symbol(pair: str) -> str:
        return pair.split(":")[0].replace("/", "")

    @staticmethod
    def _to_spot_symbol(pair: str) -> str:
        return pair.split(":")[0]

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        return _safe_float(value)

    @staticmethod
    def _format_number(value: Any, decimals: int = 2, compact: bool = False) -> str:
        number = _safe_float(value)
        if number is None:
            return "N/A" if value in (None, "") else str(value)
        suffix = ""
        if compact:
            for threshold, label in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
                if abs(number) >= threshold:
                    number, suffix = number / threshold, label
                    break
        text = f"{number:.{decimals}f}".rstrip("0").rstrip(".")
        return f"{'0' if text in {'-0', '-0.0', ''} else text}{suffix}"

    def _format_prompt_timestamp(self, value: Any) -> str:
        parsed = _parse_timestamp(value)
        return parsed.strftime("%Y-%m-%d %H:%M UTC") if parsed else str(value or "N/A")

    @staticmethod
    def _describe_change(
        change: float | None,
        threshold: float = 0.001,
        positive: str = "rising",
        negative: str = "falling",
        flat: str = "stable",
    ) -> str:
        if change is None:
            return flat
        return positive if change > threshold else negative if change < -threshold else flat

    @staticmethod
    def _pressure_label(imbalance: float | None) -> str | None:
        if imbalance is None:
            return None
        if imbalance > 1.5:
            return "strong buy pressure"
        if imbalance > 1.2:
            return "moderate buy pressure"
        if imbalance < 0.67:
            return "strong sell pressure"
        return "moderate sell pressure" if imbalance < 0.83 else "balanced"

    def _summarize_ticker(self, ticker: dict[str, Any], orderbook: dict[str, Any]) -> list[str]:
        ticker, orderbook = ticker or {}, orderbook or {}
        last, bid, ask = (_safe_float(ticker.get(key)) for key in ("last", "bid", "ask"))
        bid = bid or _safe_float((orderbook.get("best_bid") or [None])[0])
        ask = ask or _safe_float((orderbook.get("best_ask") or [None])[0])
        parts = [f"last {self._format_number(last, 4)}"] if last is not None else []
        if bid is not None and ask is not None:
            parts.append(f"bid/ask {self._format_number(bid, 4)} / {self._format_number(ask, 4)}")
        lines = [f"- Price: {' | '.join(parts)}"] if parts else []
        context = self._ticker_context_parts(ticker, last)
        return lines + ([f"- 24h Context: {' | '.join(context)}"] if context else [])

    def _ticker_context_parts(self, ticker: dict[str, Any], last: float | None) -> list[str]:
        high, low, change = (_safe_float(ticker.get(key)) for key in ("high", "low", "change_pct"))
        parts = [f"24h change {change:+.2f}%"] if change is not None else []
        if low is not None and high is not None:
            parts.append(f"24h range {self._format_number(low, 4)}-{self._format_number(high, 4)}")
            if last is not None and high > low:
                parts.append(f"range location {(last - low) / (high - low):.0%}")
        return parts

    @staticmethod
    def _time_to_timestamp_text(value: Any) -> str | None:
        parsed = _parse_timestamp(value)
        if parsed is None:
            return None
        seconds = (parsed - datetime.now(UTC)).total_seconds()
        if seconds < -60:
            return "settlement time passed / stale"
        minutes = round(max(seconds, 0) / 60)
        return f"in {minutes}m" if minutes < 90 else f"in {seconds / 3600:.1f}h"

    @staticmethod
    def _classify_squeeze_risk(
        price_change: float | None, oi_change: float | None, funding_rate: float | None
    ) -> str | None:
        if price_change is None or oi_change is None:
            return None
        if oi_change > 0.002 and price_change > 0.001:
            return (
                "short squeeze risk"
                if funding_rate is not None and funding_rate < 0
                else "new longs"
            )
        if oi_change > 0.002 and price_change < -0.001:
            return (
                "long squeeze risk"
                if funding_rate is not None and funding_rate > 0
                else "new shorts"
            )
        if oi_change < -0.002:
            return "short covering" if price_change > 0 else "long liquidation"
        return "no clear squeeze signal"

    def _funding_summary(self, data: dict[str, Any]) -> list[str]:
        funding, basis = data.get("funding_rate") or {}, data.get("basis") or {}
        rate, basis_pct = (
            _safe_float(funding.get("funding_rate")),
            _safe_float(basis.get("basis_pct")),
        )
        parts = [f"funding {rate * 100:+.4f}%"] if rate is not None else []
        if timing := self._time_to_timestamp_text(funding.get("next_funding_time")):
            parts.append(f"time_to_funding {timing}")
        if basis_pct is not None:
            parts.append(f"basis {basis_pct:+.4f}%")
        return parts

    def _positioning_summary(self, data: dict[str, Any]) -> list[str]:
        interest = data.get("open_interest") or {}
        current = _safe_float(interest.get("open_interest"))
        history = [_safe_float(item.get("open_interest")) for item in interest.get("history", [])]
        previous = next((value for value in reversed(history) if value not in (None, 0)), None)
        change = (current - previous) / previous if current is not None and previous else None
        parts = [f"OI {self._format_number(current, 2, True)}"] if current is not None else []
        if change is not None:
            parts.append(f"OI change {change:+.2%} vs recent")
        return parts

    def _summarize_derivatives(self, data: dict[str, Any]) -> list[str]:
        funding, position = self._funding_summary(data), self._positioning_summary(data)
        lines = [f"- Funding / Basis: {' | '.join(funding)}"] if funding else []
        return lines + ([f"- Positioning: {' | '.join(position)}"] if position else [])

    def _latest_price_change(self, data: dict[str, Any], last: float | None) -> float | None:
        if last is None:
            return None
        for timeframe in self.config.market_data.kline_timeframes:
            candles = (data.get("klines") or {}).get(timeframe) or []
            close = _safe_float(candles[-2].get("close")) if len(candles) > 1 else None
            if close not in (None, 0):
                return (last - close) / close
        return None

    def _summarize_order_flow(self, data: dict[str, Any]) -> list[str]:
        futures = (data.get("market_flow") or {}).get("taker_buy_sell_volume") or []
        spot = (data.get("spot_trade_flow") or {}).get("spot_trade_flow") or []
        ratio = _safe_float(futures[-1].get("buy_sell_ratio")) if futures else None
        imbalance = _safe_float(spot[-1].get("buy_sell_imbalance")) if spot else None
        parts = [f"futures buy/sell {ratio:.2f}"] if ratio is not None else []
        if imbalance is not None:
            parts.append(f"spot imbalance {imbalance:+.2f}")
        return [f"- Order Flow: {' | '.join(parts)}"] if parts else []

    def _summarize_reference_prices(self, reference: dict[str, Any]) -> list[str]:
        index = reference.get("index_price_klines", [])
        premium, parts = reference.get("premium_index_klines", []), []
        if len(index) > 1:
            start, end = _safe_float(index[0].get("close")), _safe_float(index[-1].get("close"))
            if start not in (None, 0) and end is not None:
                parts.append(f"index {(end - start) / start:+.2%}")
        if premium and (value := _safe_float(premium[-1].get("close"))) is not None:
            parts.append(f"premium {value * 100:+.4f}%")
        return [f"- Reference Prices: {' | '.join(parts)}"] if parts else []

    def _summarize_orderbook(self, orderbook: dict[str, Any]) -> list[str]:
        imbalance = _safe_float(orderbook.get("bid_ask_imbalance"))
        label = self._pressure_label(imbalance)
        if label is None or imbalance is None:
            return []
        return [f"- Order Book: {label} (imbalance {imbalance:.2f})"]

    @staticmethod
    def _percentile_rank(value: float, samples: list[float]) -> float | None:
        clean = [sample for sample in samples if sample >= 0]
        return sum(sample <= value for sample in clean) / len(clean) * 100 if clean else None

    def _realized_volatility_state(
        self, candles: list[dict[str, Any]], window: int = 12
    ) -> str | None:
        closes = [_safe_float(item.get("close")) for item in candles]
        closes = [value for value in closes if value not in (None, 0)]
        returns = [
            (closes[index] - closes[index - 1]) / closes[index - 1]
            for index in range(1, len(closes))
        ]
        if len(returns) < window:
            return None
        sample = returns[-window:]
        mean = sum(sample) / window
        volatility = (sum((value - mean) ** 2 for value in sample) / window) ** 0.5
        return f"realized vol {volatility:.2%}"

    def _build_timeframe_summary(self, timeframe: str, candles: list[dict[str, Any]]) -> str | None:
        if not candles or (close := _safe_float(candles[-1].get("close"))) is None:
            return None
        parts = [f"close {self._format_number(close, 4)}"]
        if state := self._realized_volatility_state(candles):
            parts.append(state)
        if volume := self._volume_summary(candles):
            parts.append(volume)
        return f"- {timeframe}: {' | '.join(parts)}"

    @staticmethod
    def _volume_summary(candles: list[dict[str, Any]]) -> str | None:
        volume = _safe_float(candles[-1].get("volume"))
        recent = [_safe_float(item.get("volume")) for item in candles[-6:-1]]
        recent = [value for value in recent if value not in (None, 0)]
        if volume is None or not recent:
            return None
        return f"volume activity {volume / (sum(recent) / len(recent)):.1f}x recent"

    @staticmethod
    def _window_return(candles: list[dict[str, Any]], bars: int) -> float | None:
        if len(candles) <= bars:
            return None
        end = _safe_float(candles[-1].get("close"))
        start = _safe_float(candles[-bars - 1].get("close"))
        return (end - start) / start if end is not None and start not in (None, 0) else None

    def _extract_beta_candles(
        self, *, target_pair: str, current_pair: str, data: dict[str, Any], timeframe: str
    ) -> list[dict[str, Any]]:
        if target_pair == current_pair:
            return (data.get("klines") or {}).get(timeframe) or []
        try:
            frame = self.dp.get_pair_dataframe(pair=target_pair, timeframe=timeframe)
            return [] if frame is None or frame.empty else frame.tail(64).to_dict("records")
        except _DATA_ERRORS:
            return []

    def _summarize_market_beta(self, pair: str, data: dict[str, Any]) -> list[str]:
        timeframes = self.config.market_data.kline_timeframes
        timeframe, parts = timeframes[0] if timeframes else "5m", []
        for asset in ("BTC", "ETH"):
            candles = self._extract_beta_candles(
                target_pair=f"{asset}/USDT:USDT",
                current_pair=data.get("pair") or pair,
                data=data,
                timeframe=timeframe,
            )
            if (value := self._window_return(candles, 1)) is not None:
                parts.append(f"{asset} {value:+.2%}")
        return [f"- {' | '.join(parts)}"] if parts else []

    def build_context_text(self, pair: str, data: dict[str, Any] | None = None) -> str:
        """Build a compact prompt section from collected market data."""
        data = self.collect(pair) if data is None else data
        lines = [
            f"## Market Data - {pair}",
            f"**Timestamp**: {self._format_prompt_timestamp(data.get('timestamp'))}",
            "",
        ]
        flow = self._summarize_derivatives(data) + self._summarize_order_flow(data)
        flow += self._summarize_reference_prices(data.get("reference_price_klines") or {})
        flow += self._summarize_orderbook(data.get("orderbook") or {})
        states = [
            summary
            for timeframe in self.config.market_data.kline_timeframes
            if (
                summary := self._build_timeframe_summary(
                    timeframe, (data.get("klines") or {}).get(timeframe) or []
                )
            )
        ]
        sections = (
            (
                "Market Snapshot",
                self._summarize_ticker(data.get("ticker") or {}, data.get("orderbook") or {}),
            ),
            ("Market Beta", self._summarize_market_beta(pair, data)),
            ("Derivatives & Flow", flow),
            ("Market State", states),
        )
        for title, content in sections:
            if content:
                lines.extend((f"### {title}", *content, ""))
        return "\n".join(lines)
