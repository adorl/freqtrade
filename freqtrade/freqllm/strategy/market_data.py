"""Runtime integration and external-market-data mixin."""

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from pandas import DataFrame

from freqtrade.enums import RunMode
from freqtrade.freqllm.persistence import FreqLLMDatabase
from freqtrade.freqllm.strategy.contracts import StrategyCollaborators


logger = logging.getLogger(__name__)

_MARKET_DATA_ERRORS = (AttributeError, KeyError, TypeError, ValueError, RuntimeError, OSError)
_FILL_COLUMNS = (
    "%-funding_rate",
    "%-funding_rate_change",
    "%-basis",
    "%-basis_pct",
    "%-basis_change",
    "%-mark_price_deviation",
    "%-index_price_deviation",
    "%-oi_change_1",
    "%-oi_change_3",
    "%-oi_change_12",
    "%-buy_sell_imbalance",
    "%-buy_sell_ratio",
    "%-taker_flow_total",
    "%-premium_index",
    "%-premium_index_change",
    "%-index_return_1",
    "%-index_return_3",
    "%-spot_trade_flow_ratio",
    "%-spot_trade_flow_imbalance",
    "%-spot_trade_flow_total",
    "%-spot_trade_count",
    "%-max_leverage",
    "%-top_trader_account_ratio",
    "%-top_trader_position_ratio",
    "%-global_account_ratio",
)


@dataclass(frozen=True)
class _HistoryQuery:
    """Time boundaries and period shared by persisted-history queries."""

    period: str
    start_time: datetime | None
    end_time: datetime | None


@dataclass(frozen=True)
class _ReferenceKlineSpec:
    """Source feature and destination key for a reference-price history."""

    feature: str
    target: str


@dataclass(frozen=True)
class _Enrichment:
    """One independently recoverable dataframe enrichment operation."""

    label: str
    payload: Any
    apply: Callable[[DataFrame, Any], DataFrame]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, pd.NA)).replace([float("inf"), float("-inf")], pd.NA)


def _decode_raw_payload(value: Any) -> dict[str, Any]:
    """Decode a persisted JSON object without accepting non-object payloads."""
    if not value:
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("Unable to decode persisted market payload: %s", exc)
        return {}
    return decoded if isinstance(decoded, dict) else {}


class StrategyMarketDataMixin:
    """Load and enrich causal external-market features for the strategy."""

    strategy_collaborators: StrategyCollaborators

    @staticmethod
    def external_market_feature_columns() -> tuple[str, ...]:
        """Return the stable external feature schema used by FreqAI."""
        return _FILL_COLUMNS

    @staticmethod
    def zscore_series(series: pd.Series, period: int) -> pd.Series:
        """Calculate a rolling z-score with safe zero-standard-deviation handling."""
        mean = series.rolling(period).mean()
        std = series.rolling(period).std().replace(0, pd.NA)
        return ((series - mean) / std).replace([float("inf"), float("-inf")], pd.NA)

    def _zscore_series(self, series: pd.Series, period: int) -> pd.Series:
        return self.zscore_series(series, period)

    @staticmethod
    def _parse_external_feature_timestamp(value: Any) -> pd.Timestamp:
        """Parse persisted/live external feature timestamps into UTC timestamps."""
        if value is None or value == "":
            return pd.NaT
        try:
            if isinstance(value, (int, float)):
                return pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
            return pd.to_datetime(value, utc=True, errors="coerce")
        except _MARKET_DATA_ERRORS:
            return pd.NaT

    def _external_market_period(self) -> str:
        cfg = self.strategy_collaborators.runtime.advisor_config
        if cfg is not None:
            return str(getattr(cfg.market_data, "long_short_ratio_period", "15m") or "15m")
        raw = self.config.get("llm_strategy", {}) if isinstance(self.config, dict) else {}
        market_data = raw.get("market_data", {}) if isinstance(raw, dict) else {}
        return str(market_data.get("long_short_ratio_period", "15m") or "15m")

    def _external_market_limit(self) -> int:
        cfg = self.strategy_collaborators.runtime.advisor_config
        if cfg is not None:
            return int(getattr(cfg.market_data, "long_short_ratio_limit", 48) or 48)
        raw = self.config.get("llm_strategy", {}) if isinstance(self.config, dict) else {}
        market_data = raw.get("market_data", {}) if isinstance(raw, dict) else {}
        return int(market_data.get("long_short_ratio_limit", 48) or 48)

    def _ensure_freqllm_db(self) -> FreqLLMDatabase | None:
        """Return the database owned by the strategy composition root."""
        return self.strategy_collaborators.runtime.database

    @staticmethod
    def _market_history(db, pair: str, feature: str, query: _HistoryQuery, **options):
        return db.get_market_feature_history(
            pair,
            feature,
            period=options.get("period", query.period),
            since=query.start_time,
            until=query.end_time,
            **({"limit": options["limit"]} if "limit" in options else {}),
        )

    def _load_funding_history(self, db, pair: str, query: _HistoryQuery, market: dict) -> None:
        rows = self._market_history(db, pair, "funding_rate", query)
        if not rows:
            return
        keys = (
            "funding_rate",
            "basis",
            "basis_pct",
            "mark_price",
            "index_price",
            "next_funding_time",
        )
        history = [
            {
                "timestamp": row.get("source_timestamp") or row.get("timestamp"),
                **{key: row.get(key) for key in keys},
            }
            for row in rows
        ]
        market["funding_rate_history"] = history
        market["funding_rate"] = {key: rows[-1].get(key) for key in keys}

    def _load_basis_history(self, db, pair: str, query: _HistoryQuery, market: dict) -> None:
        rows = self._market_history(db, pair, "basis", query)
        if not rows:
            return
        history = []
        for row in rows:
            decoded = _decode_raw_payload(row.get("raw_payload"))
            history.append(
                {
                    "timestamp": row.get("source_timestamp") or row.get("timestamp"),
                    "basis": row.get("basis"),
                    "basis_pct": row.get("basis_pct"),
                    "mark_price": row.get("mark_price"),
                    "index_price": row.get("index_price"),
                    "futures_price": decoded.get("futures_price"),
                    "annualized_basis_pct": decoded.get("annualized_basis_pct"),
                }
            )
        market["basis_history"] = history
        market["basis"] = {key: history[-1].get(key) for key in history[-1] if key != "timestamp"}

    def _load_open_interest(self, db, pair: str, query: _HistoryQuery, market: dict) -> None:
        rows = self._market_history(db, pair, "open_interest_history", query)
        if not rows:
            return
        history = [
            {
                "timestamp": row.get("source_timestamp") or row.get("timestamp"),
                "open_interest": row.get("open_interest"),
                "open_interest_value": row.get("open_interest_value"),
            }
            for row in rows
        ]
        market["open_interest"] = {"history": history, **history[-1]}

    def _load_taker_flow(self, db, pair: str, query: _HistoryQuery, market: dict) -> None:
        rows = self._market_history(db, pair, "taker_buy_sell_volume", query)
        if rows:
            market["market_flow"] = {
                "taker_buy_sell_volume": [
                    {
                        "timestamp": row.get("source_timestamp") or row.get("timestamp"),
                        "buy_vol": row.get("buy_vol"),
                        "sell_vol": row.get("sell_vol"),
                        "buy_sell_ratio": row.get("buy_sell_ratio"),
                        "buy_sell_imbalance": row.get("buy_sell_imbalance"),
                    }
                    for row in rows
                ]
            }

    def _load_reference_kline(
        self,
        db,
        pair: str,
        query: _HistoryQuery,
        destination: tuple[dict, _ReferenceKlineSpec],
    ) -> None:
        market, spec = destination
        rows = self._market_history(db, pair, spec.feature, query)
        if not rows:
            return
        history = []
        for row in rows:
            decoded = _decode_raw_payload(row.get("raw_payload"))
            close = row.get("index_price")
            if spec.feature == "premium_index_kline":
                basis_pct = row.get("basis_pct")
                close = basis_pct / 100.0 if basis_pct is not None else None
            history.append(
                {
                    "timestamp": row.get("source_timestamp") or row.get("timestamp"),
                    "close": close,
                    **{key: decoded.get(key) for key in ("open", "high", "low", "volume")},
                }
            )
        reference = market.setdefault("reference_price_klines", {})
        reference["timeframe"] = query.period
        reference[spec.target] = history

    def _load_spot_flow(self, db, pair: str, query: _HistoryQuery, market: dict) -> None:
        rows = self._market_history(db, pair, "spot_trade_flow", query)
        if not rows:
            return
        history = []
        for row in rows:
            decoded = _decode_raw_payload(row.get("raw_payload"))
            history.append(
                {
                    "timestamp": row.get("source_timestamp") or row.get("timestamp"),
                    "buy_quote_vol": row.get("buy_vol"),
                    "sell_quote_vol": row.get("sell_vol"),
                    "buy_sell_ratio": row.get("buy_sell_ratio"),
                    "buy_sell_imbalance": row.get("buy_sell_imbalance"),
                    "trade_count": row.get("event_count"),
                    "buy_trade_count": decoded.get("buy_trade_count"),
                    "sell_trade_count": decoded.get("sell_trade_count"),
                    "avg_trade_notional": decoded.get("avg_trade_notional"),
                }
            )
        market["spot_trade_flow"] = {"period": query.period, "spot_trade_flow": history}

    def _load_leverage(self, db, pair: str, query: _HistoryQuery, market: dict) -> None:
        rows = self._market_history(db, pair, "max_leverage", query, period="snapshot", limit=1)
        if rows:
            latest = rows[-1]
            market["leverage_stats"] = {
                "max_leverage": latest.get("max_leverage"),
                "timestamp": latest.get("source_timestamp") or latest.get("timestamp"),
            }

    @staticmethod
    def _load_ratios(db, pair: str, query: _HistoryQuery, ratios: dict) -> None:
        mapping = {
            "top_trader_account": "top_trader_account_ratio",
            "top_trader_position": "top_trader_position_ratio",
            "global_account": "global_account_ratio",
        }
        for ratio_type, target in mapping.items():
            rows = db.get_ls_ratio_records(
                pair,
                ratio_type,
                period=query.period,
                since=query.start_time,
                until=query.end_time,
            )
            if rows:
                ratios[target] = [
                    {
                        "timestamp": row.get("source_timestamp") or row.get("timestamp"),
                        "long_short_ratio": row.get("long_short_ratio"),
                    }
                    for row in rows
                ]

    def _load_persisted_external_market_data(
        self,
        pair: str,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> dict[str, Any]:
        """Load persisted market features and long/short ratios for replay."""
        db = self._ensure_freqllm_db()
        if db is None:
            return {}
        payload: dict[str, Any] = {"market": {}, "ls": {}}
        query = _HistoryQuery(self._external_market_period(), start_time, end_time)
        loaders = (
            self._load_funding_history,
            self._load_basis_history,
            self._load_open_interest,
            self._load_taker_flow,
            self._load_spot_flow,
            self._load_leverage,
        )
        try:
            for loader in loaders:
                loader(db, pair, query, payload["market"])
            self._load_reference_kline(
                db,
                pair,
                query,
                (
                    payload["market"],
                    _ReferenceKlineSpec("index_price_kline", "index_price_klines"),
                ),
            )
            self._load_reference_kline(
                db,
                pair,
                query,
                (
                    payload["market"],
                    _ReferenceKlineSpec("premium_index_kline", "premium_index_klines"),
                ),
            )
            self._load_ratios(db, pair, query, payload["ls"])
        except _MARKET_DATA_ERRORS as exc:
            logger.debug("[%s] Failed to load persisted external market data: %s", pair, exc)
        return payload

    def _get_cached_external_market_data(
        self,
        pair: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Return cached external data, preferring persisted history."""
        start_key = start_time.isoformat() if start_time else ""
        end_key = end_time.isoformat() if end_time else ""
        cache_key = f"{pair}|{start_key}|{end_key}"
        cached = self.strategy_collaborators.runtime.external_feature_cache.get(cache_key)
        if cached is not None:
            return cached
        payload = self._load_persisted_external_market_data(pair, start_time, end_time)
        has_persisted_data = bool(payload.get("market") or payload.get("ls"))
        live_mode = self.dp is not None and self.dp.runmode in (RunMode.DRY_RUN, RunMode.LIVE)
        if not has_persisted_data and live_mode:
            self._collect_live_external_data(pair, payload)
        self.strategy_collaborators.runtime.external_feature_cache[cache_key] = payload
        return payload

    def _collect_live_external_data(self, pair: str, payload: dict[str, Any]) -> None:
        advisor = self.strategy_collaborators.runtime.advisor
        market_collector = getattr(advisor, "market_collector", None)
        ls_collector = getattr(advisor, "ls_collector", None)
        if market_collector is not None:
            try:
                payload["market"] = market_collector.collect(pair)
            except _MARKET_DATA_ERRORS as exc:
                logger.debug("[%s] Failed to collect cached market features: %s", pair, exc)
        if ls_collector is not None:
            try:
                payload["ls"] = ls_collector.collect(
                    pair,
                    period=self._external_market_period(),
                    limit=self._external_market_limit(),
                )
            except _MARKET_DATA_ERRORS as exc:
                logger.debug("[%s] Failed to collect cached long-short features: %s", pair, exc)

    def _history_frame(self, rows: Iterable[Mapping[str, Any]]) -> DataFrame:
        frame = DataFrame(rows)
        if frame.empty or "timestamp" not in frame:
            return DataFrame()
        frame["date"] = frame["timestamp"].apply(self._parse_external_feature_timestamp)
        return frame.dropna(subset=["date"]).sort_values("date")

    @staticmethod
    def _merge_features(base: DataFrame, feature: DataFrame, columns: list[str]) -> DataFrame:
        if feature.empty:
            return base
        return pd.merge_asof(base, feature[["date", *columns]], on="date", direction="backward")

    @staticmethod
    def _apply_enricher(
        pair: str,
        dataframe: DataFrame,
        enrichment: _Enrichment,
    ) -> DataFrame:
        if not enrichment.payload:
            return dataframe
        try:
            return enrichment.apply(dataframe, enrichment.payload)
        except _MARKET_DATA_ERRORS as exc:
            logger.debug("[%s] Failed to enrich %s features: %s", pair, enrichment.label, exc)
            return dataframe

    def _enrich_funding(self, base: DataFrame, rows: list[dict]) -> DataFrame:
        frame = self._history_frame(rows)
        if frame.empty:
            return base
        frame["%-funding_rate"] = pd.to_numeric(frame.get("funding_rate"), errors="coerce")
        frame["%-funding_rate_change"] = frame["%-funding_rate"].pct_change()
        return self._merge_features(base, frame, ["%-funding_rate", "%-funding_rate_change"])

    def _enrich_basis(self, base: DataFrame, rows: list[dict]) -> DataFrame:
        frame = self._history_frame(rows)
        if frame.empty:
            return base
        mapping = {
            "basis": "%-basis",
            "basis_pct": "%-basis_pct",
        }
        for source, target in mapping.items():
            frame[target] = pd.to_numeric(frame.get(source), errors="coerce")
        frame["%-basis_change"] = frame["%-basis_pct"].diff()
        frame["__mark_price"] = pd.to_numeric(frame.get("mark_price"), errors="coerce")
        frame["__index_price"] = pd.to_numeric(frame.get("index_price"), errors="coerce")
        enriched = self._merge_features(
            base,
            frame,
            [*mapping.values(), "%-basis_change", "__mark_price", "__index_price"],
        )
        close = pd.to_numeric(enriched["close"], errors="coerce")
        enriched["%-mark_price_deviation"] = _safe_ratio(enriched["__mark_price"] - close, close)
        enriched["%-index_price_deviation"] = _safe_ratio(
            close - enriched["__index_price"], enriched["__index_price"]
        )
        return enriched.drop(columns=["__mark_price", "__index_price"])

    def _enrich_open_interest(self, base: DataFrame, data: dict) -> DataFrame:
        frame = self._history_frame(data.get("history") or [])
        if frame.empty or "open_interest" not in frame:
            return base
        frame["oi_value"] = pd.to_numeric(frame["open_interest"], errors="coerce")
        frame = frame.dropna(subset=["oi_value"])
        columns = []
        for period in (1, 3, 12):
            column = f"%-oi_change_{period}"
            frame[column] = frame["oi_value"].pct_change(period)
            columns.append(column)
        return self._merge_features(base, frame, columns)

    def _enrich_taker_flow(self, base: DataFrame, rows: list[dict]) -> DataFrame:
        frame = self._history_frame(rows)
        if frame.empty:
            return base
        buy = pd.to_numeric(frame.get("buy_vol"), errors="coerce").fillna(0.0)
        sell = pd.to_numeric(frame.get("sell_vol"), errors="coerce").fillna(0.0)
        frame["%-buy_sell_ratio"] = pd.to_numeric(
            frame.get("buy_sell_ratio"), errors="coerce"
        ).fillna(0.0)
        frame["%-buy_sell_imbalance"] = pd.to_numeric(
            frame.get("buy_sell_imbalance"), errors="coerce"
        )
        frame["%-taker_flow_total"] = buy + sell
        return self._merge_features(
            base,
            frame,
            ["%-buy_sell_ratio", "%-buy_sell_imbalance", "%-taker_flow_total"],
        )

    def _enrich_index_kline(self, base: DataFrame, rows: list[dict]) -> DataFrame:
        frame = self._history_frame(rows)
        if frame.empty:
            return base
        close = pd.to_numeric(frame.get("close"), errors="coerce")
        frame["%-index_return_1"] = close.pct_change(1)
        frame["%-index_return_3"] = close.pct_change(3)
        return self._merge_features(base, frame, ["%-index_return_1", "%-index_return_3"])

    def _enrich_premium_kline(self, base: DataFrame, rows: list[dict]) -> DataFrame:
        frame = self._history_frame(rows)
        if frame.empty:
            return base
        frame["%-premium_index"] = pd.to_numeric(frame.get("close"), errors="coerce")
        frame["%-premium_index_change"] = frame["%-premium_index"].diff()
        return self._merge_features(base, frame, ["%-premium_index", "%-premium_index_change"])

    def _enrich_spot_flow(self, base: DataFrame, data: dict) -> DataFrame:
        frame = self._history_frame(data.get("spot_trade_flow") or [])
        if frame.empty:
            return base
        buy = pd.to_numeric(frame.get("buy_quote_vol"), errors="coerce").fillna(0.0)
        sell = pd.to_numeric(frame.get("sell_quote_vol"), errors="coerce").fillna(0.0)
        mapping = {
            "buy_sell_ratio": "%-spot_trade_flow_ratio",
            "buy_sell_imbalance": "%-spot_trade_flow_imbalance",
            "trade_count": "%-spot_trade_count",
        }
        for source, target in mapping.items():
            frame[target] = pd.to_numeric(frame.get(source), errors="coerce").fillna(0.0)
        frame["%-spot_trade_flow_total"] = buy + sell
        return self._merge_features(base, frame, [*mapping.values(), "%-spot_trade_flow_total"])

    def _enrich_ls_ratios(self, pair: str, base: DataFrame, data: dict) -> DataFrame:
        mapping = {
            "top_trader_account_ratio": "%-top_trader_account_ratio",
            "top_trader_position_ratio": "%-top_trader_position_ratio",
            "global_account_ratio": "%-global_account_ratio",
        }
        for source, target in mapping.items():
            enrichment = _Enrichment(
                source,
                data.get(source) or [],
                lambda frame, rows, column=target: self._enrich_ratio(frame, rows, column),
            )
            base = self._apply_enricher(pair, base, enrichment)
        return base

    def _enrich_ratio(self, base: DataFrame, rows: list[dict], target: str) -> DataFrame:
        frame = self._history_frame(rows)
        if frame.empty or "long_short_ratio" not in frame:
            return base
        frame[target] = pd.to_numeric(frame["long_short_ratio"], errors="coerce")
        return self._merge_features(base, frame.dropna(subset=[target]), [target])

    def _feature_time_bounds(self, dataframe: DataFrame) -> tuple[datetime | None, datetime | None]:
        if dataframe.empty or "date" not in dataframe:
            return None, None
        try:
            return (
                pd.Timestamp(dataframe["date"].min()).to_pydatetime(),
                pd.Timestamp(dataframe["date"].max()).to_pydatetime(),
            )
        except _MARKET_DATA_ERRORS:
            return None, None

    @staticmethod
    def _finalize_external_features(base: DataFrame) -> DataFrame:
        for column in _FILL_COLUMNS:
            if column not in base:
                base[column] = 0.0
            base[column] = (
                pd.to_numeric(base[column], errors="coerce")
                .replace([float("inf"), float("-inf")], pd.NA)
                .fillna(0.0)
            )
        interactions = {
            "%-ret_x_oi_change_1": "%-oi_change_1",
            "%-ret_x_oi_change_3": "%-oi_change_3",
            "%-ret_x_oi_change_12": "%-oi_change_12",
            "%-ret_x_basis_pct": "%-basis_pct",
            "%-ret_x_premium_index": "%-premium_index",
            "%-ret_x_spot_flow_imbalance": "%-spot_trade_flow_imbalance",
        }
        for target, source in interactions.items():
            base[target] = base["%-pct_change"] * base[source]
        return base.sort_index()

    def _market_enrichments(self, market: dict[str, Any]) -> tuple[_Enrichment, ...]:
        basis = market.get("basis") or {}
        basis_rows = (
            market.get("basis_history") or basis.get("history") or ([basis] if basis else [])
        )
        reference = market.get("reference_price_klines") or {}
        return (
            _Enrichment(
                "funding history",
                market.get("funding_rate_history") or [],
                self._enrich_funding,
            ),
            _Enrichment("basis history", basis_rows, self._enrich_basis),
            _Enrichment(
                "open-interest",
                market.get("open_interest") or {},
                self._enrich_open_interest,
            ),
            _Enrichment(
                "taker flow",
                (market.get("market_flow") or {}).get("taker_buy_sell_volume") or [],
                self._enrich_taker_flow,
            ),
            _Enrichment(
                "index-price kline",
                reference.get("index_price_klines") or [],
                self._enrich_index_kline,
            ),
            _Enrichment(
                "premium-index kline",
                reference.get("premium_index_klines") or [],
                self._enrich_premium_kline,
            ),
            _Enrichment(
                "spot-trade-flow",
                market.get("spot_trade_flow") or {},
                self._enrich_spot_flow,
            ),
        )

    def _inject_external_market_features(
        self,
        dataframe: DataFrame,
        metadata: dict,
    ) -> DataFrame:
        """Attach external futures microstructure features using causal merges."""
        pair = str(metadata.get("pair") or "")
        if not pair or self.dp is None or self._external_features_disabled():
            return dataframe
        base = dataframe.sort_values("date").copy()
        time_bounds = self._feature_time_bounds(base)
        payload = self._get_cached_external_market_data(pair, *time_bounds)
        market = payload.get("market") or {}
        for enrichment in self._market_enrichments(market):
            base = self._apply_enricher(pair, base, enrichment)
        leverage = pd.to_numeric(
            (market.get("leverage_stats") or {}).get("max_leverage"), errors="coerce"
        )
        if pd.notna(leverage):
            base["%-max_leverage"] = float(leverage)
        base = self._enrich_ls_ratios(pair, base, payload.get("ls") or {})
        return self._finalize_external_features(base)
