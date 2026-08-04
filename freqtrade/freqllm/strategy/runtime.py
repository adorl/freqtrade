"""Runtime integration and external-market-data mixin."""

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.freqllm.attribution import SimpleAttributionWriter
from freqtrade.freqllm.strategy.contracts import StrategyCollaborators


logger = logging.getLogger(__name__)


class StrategyRuntimeMixin:
    """Coordinate adaptive outcomes, advisor refreshes, and shared runtime helpers."""

    strategy_collaborators: StrategyCollaborators

    _RUNTIME_ERRORS = (AttributeError, KeyError, TypeError, ValueError, RuntimeError, OSError)

    def _build_advisor_config(self) -> dict[str, Any]:
        raw = self.config.get("llm_strategy", {}) if isinstance(self.config, dict) else {}
        raw = raw if isinstance(raw, dict) else {}
        allowed = {
            "db_url",
            "llm_bypass_enabled",
            "llm",
            "context",
            "schedule",
            "market_data",
            "performance",
        }
        return {"llm_strategy": {key: value for key, value in raw.items() if key in allowed}}

    def _refresh_adaptive_params(self, pair: str | None = None) -> None:
        collaborators = self.strategy_collaborators
        if collaborators.adaptive_manager.enabled:
            collaborators.decision_engine.set_adaptive_params(
                collaborators.adaptive_manager.current_parameters_dict(pair)
            )

    @staticmethod
    def _utc_timestamp(value: datetime) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    @staticmethod
    def _signals_by_pair(pending: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in pending:
            grouped.setdefault(str(item.get("pair") or ""), []).append(item)
        return grouped

    def _pair_outcome_data(self, pair: str):
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        except self._RUNTIME_ERRORS as exc:
            logger.debug("[%s] Unable to load adaptive outcome data: %s", pair, exc)
            return None
        if dataframe is None or dataframe.empty or "date" not in dataframe:
            return None
        feature_params = self.config.get("freqai", {}).get("feature_parameters", {})
        detail_tf = self._label_detail_timeframe(feature_params)
        detail_df = self._get_detail_dataframe(pair, detail_tf) if detail_tf else None
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        return dataframe, dates, detail_tf, detail_df

    def _mature_signal_position(
        self,
        item: dict[str, Any],
        timing: tuple,
        current_ts: pd.Timestamp,
    ) -> tuple[pd.Timestamp, int, int] | None:
        dataframe, dates, timeframe_minutes = timing
        reference = pd.to_datetime(item.get("reference_time"), utc=True, errors="coerce")
        if pd.isna(reference):
            return None
        default_hold = self.strategy_collaborators.simple_config.exit.max_hold_candles
        max_hold = int(self._safe_float(item.get("max_hold_candles"), default_hold) or default_hold)
        mature_time = reference + pd.Timedelta(minutes=timeframe_minutes * (max(1, max_hold) + 1))
        positions = dataframe.index[dates == reference].tolist()
        if current_ts < mature_time or not positions:
            return None
        return reference, max_hold, int(positions[0])

    def _realized_signal_metrics(
        self, item: dict[str, Any], outcome_data: tuple, position: tuple
    ) -> tuple[str, dict]:
        dataframe, _, detail_tf, detail_df = outcome_data
        _, max_hold, row_index = position
        side = str(item.get("candidate_side") or item.get("side") or "long")
        timeframe_minutes = max(1, timeframe_to_minutes(self.timeframe))
        metrics = SimpleAttributionWriter.realized_metrics(
            dataframe,
            row_index,
            self._safe_float(item.get("price"), 0.0),
            side,
            max_hold_candles=max_hold,
            detail_dataframe=detail_df,
            timeframe_minutes=timeframe_minutes,
            detail_timeframe=detail_tf,
            detail_timeframe_minutes=timeframe_to_minutes(detail_tf) if detail_tf else 0,
        )
        return side, metrics

    def _update_mature_signal(
        self, pair: str, item: dict[str, Any], outcome_data: tuple, current_ts: pd.Timestamp
    ) -> None:
        dataframe, dates, _, _ = outcome_data
        timing = (dataframe, dates, max(1, timeframe_to_minutes(self.timeframe)))
        position = self._mature_signal_position(item, timing, current_ts)
        if position is None:
            return
        reference, _, _ = position
        side, updates = self._realized_signal_metrics(item, outcome_data, position)
        self.strategy_collaborators.adaptive_manager.update_signal_outcome(
            pair, reference.isoformat(), side, updates
        )

    def _update_adaptive_live_outcomes(self, current_time: datetime) -> None:
        collaborators = self.strategy_collaborators
        manager = collaborators.adaptive_manager
        if collaborators.runtime.is_backtest_mode or not manager.enabled:
            return
        pending = manager.pending_signals()
        if not pending:
            return
        current_ts = self._utc_timestamp(current_time)
        for pair, items in self._signals_by_pair(pending).items():
            if not pair or self.dp is None:
                continue
            outcome_data = self._pair_outcome_data(pair)
            if outcome_data is None:
                continue
            for item in items:
                self._update_mature_signal(pair, item, outcome_data, current_ts)

    def _level_line(self, row: pd.Series, kind: str) -> str | None:
        price = self._safe_float(row.get(f"%%-strongest_{kind}_price"), 0.0)
        distance = self._safe_float(row.get(f"%%-strongest_{kind}_room_atr"), float("nan"))
        strength = self._safe_float(row.get(f"%-strongest_{kind}_strength"), float("nan"))
        if not np.isfinite(distance) or price <= 0:
            return None
        strength_text = f", strength {strength:.2f}" if np.isfinite(strength) else ""
        relation = "above" if kind == "resistance" else "below"
        return f"- Strongest {kind}: {price:.8g} ({distance:.2f} ATR {relation}{strength_text})"

    def _support_resistance_context(self, pair: str) -> str | None:
        """Build the same causal key-level snapshot used by FreqAI for the LLM."""
        if self.dp is None:
            return None
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        except self._RUNTIME_ERRORS as exc:
            logger.debug("[%s] Unable to build support/resistance context: %s", pair, exc)
            return None
        if dataframe is None or dataframe.empty:
            return None
        row = dataframe.iloc[-1]
        close = self._safe_float(row.get("close"), 0.0)
        resistance_atr = self._safe_float(row.get("%%-nearest_resistance_room_atr"), float("nan"))
        support_atr = self._safe_float(row.get("%%-nearest_support_room_atr"), float("nan"))
        if close <= 0 or not np.isfinite(resistance_atr) or not np.isfinite(support_atr):
            return None
        resistance = self._safe_float(row.get("%%-nearest_resistance_price"), 0.0)
        support = self._safe_float(row.get("%%-nearest_support_price"), 0.0)
        lines = [
            "## Causal Support / Resistance",
            f"- Current close: {close:.8g}",
            f"- Nearest resistance: {resistance:.8g} ({resistance_atr:.2f} ATR above)",
            f"- Nearest support: {support:.8g} ({support_atr:.2f} ATR below)",
        ]
        strong_lines = (
            self._level_line(row, "resistance"),
            self._level_line(row, "support"),
        )
        lines.extend(filter(None, strong_lines))
        lines.append(
            "- Levels combine adaptive psychological round numbers, prior rolling highs/lows, "
            "completed day/week levels and confirmed pivots; strongest levels weight source "
            "strength with distance decay; do not infer unconfirmed future pivots."
        )
        return "\n".join(lines)

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """Refresh adaptive outcomes and periodically request pair advice."""
        runtime = self.strategy_collaborators.runtime
        if kwargs.get("force_advisor_refresh", False):
            runtime.llm_advice_time.clear()
        self._update_adaptive_live_outcomes(current_time)
        self._refresh_adaptive_params()
        if runtime.is_backtest_mode or runtime.advisor is None or runtime.advisor_config is None:
            return
        interval = max(
            1,
            int(getattr(runtime.advisor_config.schedule, "analysis_interval_minutes", 15) or 15),
        )
        for pair in self._get_pair_list():
            last_time = runtime.llm_advice_time.get(pair)
            if last_time is not None and (current_time - last_time).total_seconds() < interval * 60:
                continue
            try:
                advice = runtime.advisor.analyze(
                    pair,
                    self._get_pair_list(),
                    sr_context=self._support_resistance_context(pair),
                    timeframe=self.timeframe,
                    prediction_horizon=self.label_horizon(),
                )
                if advice.get("available", True) is False:
                    runtime.llm_advice.pop(pair, None)
                    runtime.llm_advice_time.pop(pair, None)
                    continue
                runtime.llm_advice[pair] = advice
                runtime.llm_advice_time[pair] = current_time
            except self._RUNTIME_ERRORS as exc:
                runtime.llm_advice.pop(pair, None)
                runtime.llm_advice_time.pop(pair, None)
                logger.warning("[%s] LLM analysis failed; advice marked unavailable: %s", pair, exc)

    def informative_pairs(self):
        """Return configured informative pair/timeframe combinations."""
        pairs = self._get_pair_list()
        feature_params = self.config.get("freqai", {}).get("feature_parameters", {})
        timeframes = set(feature_params.get("include_timeframes", []))
        detail_tf = self._label_detail_timeframe(feature_params)
        if detail_tf:
            timeframes.add(detail_tf)
        return [(pair, tf) for pair in pairs for tf in sorted(timeframes) if tf != self.timeframe]

    def _get_pair_list(self) -> list[str]:
        exchange = self.config.get("exchange", {}) if isinstance(self.config, dict) else {}
        pairs = exchange.get("pair_whitelist", []) if isinstance(exchange, dict) else []
        return list(pairs) if pairs else []

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or pd.isna(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Safely divide two aligned series and replace infinities with NaN."""
        denominator = denominator.replace(0, pd.NA)
        return numerator.div(denominator).replace([float("inf"), float("-inf")], pd.NA)

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _as_utc(value: Any) -> datetime | None:
        if value is None or value is pd.NaT:
            return None
        try:
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            return timestamp.to_pydatetime()
        except (TypeError, ValueError, OverflowError):
            if isinstance(value, datetime):
                return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return None
