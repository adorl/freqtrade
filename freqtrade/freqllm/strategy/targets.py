"""FreqAI feature engineering and target-generation mixin."""

import hashlib
import json
import logging
import re
from typing import Any

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.enums import RunMode
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.freqllm.strategy.contracts import (
    BarrierSettings,
    DecaySettings,
    DetailBuffers,
    EarlySettings,
    EarlyWindow,
    Excursions,
    FuturePrices,
    LevelContext,
    PolicyArrays,
    PolicyResults,
    PolicySettings,
    PolicyState,
    PolicyTargets,
    StrategyCollaborators,
    TargetWindows,
    WindowTiming,
)


logger = logging.getLogger(__name__)

_TARGET_ERRORS = (AttributeError, KeyError, TypeError, ValueError, RuntimeError, OSError)
_POLICY_TARGET_FIELDS = (
    "peak_profit",
    "mae",
    "pre_profit_drawdown",
    "post_profit_drawdown",
)


class StrategyTargetsMixin:
    """Generate policy-aligned FreqAI targets from causal future windows."""

    strategy_collaborators: StrategyCollaborators

    def label_horizon(self) -> int:
        """Return the configured target horizon for strategy integrations."""
        params = self.config.get("freqai", {}).get("feature_parameters", {})
        return self._label_horizon(params)

    def _get_live_fee_rate(self, pair: str, default: float) -> float:
        """Return one-way trading fee, using exchange API in live/dry-run modes when available."""
        if not pair:
            return default
        runmode = getattr(self.dp, "runmode", None) if self.dp is not None else None
        if runmode not in (RunMode.DRY_RUN, RunMode.LIVE):
            return default
        cached = self.strategy_collaborators.runtime.fee_rate_cache.get(pair)
        if cached is not None:
            return cached
        try:
            exchange = getattr(self.dp, "_exchange", None)
            if exchange is None:
                return default
            order_type = str(self.config.get("order_types", {}).get("entry", "market") or "market")
            taker_or_maker = "taker" if order_type == "market" else "maker"
            fee_rate = self._safe_float(
                exchange.get_fee(symbol=pair, order_type=order_type, taker_or_maker=taker_or_maker),
                default,
            )
            if fee_rate > 0:
                self.strategy_collaborators.runtime.fee_rate_cache[pair] = fee_rate
                return fee_rate
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            logger.debug(
                "[%s] Failed to fetch live trading fee, fallback=%s: %s",
                pair,
                default,
                exc,
            )
        return default

    def _configured_round_trip_cost(self, pair: str, feature_params: dict[str, Any]) -> float:
        default_fee = self._safe_float(feature_params.get("label_fee_rate", 0.0005), 0.0005)
        fee_rate = self._get_live_fee_rate(pair, default_fee)
        slippage_rate = self._safe_float(feature_params.get("label_slippage_rate", 0.0001), 0.0001)
        return max(0.0, (fee_rate + slippage_rate) * 2.0)

    def _external_features_disabled(self) -> bool:
        """True when external microstructure features should be skipped (backtest)."""
        runtime = self.strategy_collaborators.runtime
        if not runtime.disable_external_in_backtest:
            return False
        if runtime.is_backtest_mode:
            return True
        runmode = getattr(self.dp, "runmode", None) if self.dp is not None else None
        return runmode in (RunMode.BACKTEST, RunMode.HYPEROPT)

    def _label_horizon(
        self,
        feature_params: dict[str, Any],
        default_max_hold_candles: int | None = None,
    ) -> int:
        """Label horizon in strategy candles, configurable but capped at 16 (=4h on 15m)."""
        if default_max_hold_candles is None:
            default_max_hold_candles = (
                self.strategy_collaborators.simple_config.exit.max_hold_candles
            )
        raw = feature_params.get("label_period_candles", default_max_hold_candles)
        try:
            value = int(raw or 1)
        except (TypeError, ValueError):
            value = 1
        return int(min(16, max(1, value)))

    def _apply_policy_retrain_identifier(
        self,
        config: dict[str, Any],
        simple_config: Any,
    ) -> None:
        """Fingerprint labels, feature settings, backend and estimator parameters."""

        if not isinstance(config, dict):
            return
        freqai = config.get("freqai")
        if not isinstance(freqai, dict):
            return
        fp = (
            freqai.get("feature_parameters", {})
            if isinstance(freqai.get("feature_parameters", {}), dict)
            else {}
        )
        exit_cfg = simple_config.exit
        risk = simple_config.risk
        policy_keys = {
            "label_period_candles": fp.get("label_period_candles"),
            "label_fee_rate": fp.get("label_fee_rate"),
            "label_slippage_rate": fp.get("label_slippage_rate"),
            "label_class_margin": fp.get("label_class_margin"),
            "label_class_edge_gap": fp.get("label_class_edge_gap"),
            "label_level_break_buffer_atr": fp.get("label_level_break_buffer_atr"),
            "label_detail_timeframe": fp.get("label_detail_timeframe"),
            "target_schema": "policy_quantile_v5",
            "feature_parameters": fp,
            "data_split_parameters": freqai.get("data_split_parameters", {}),
            "model_training_parameters": freqai.get("model_training_parameters", {}),
            "freqaimodel": config.get("freqaimodel", "FreqLLMPolicyModel"),
            "stop_loss_pct": risk.stop_loss_pct,
            "take_profit_pct": risk.take_profit_pct,
            "trailing_enabled": exit_cfg.trailing_enabled,
            "trailing_activation_pct": exit_cfg.trailing_activation_pct,
            "trailing_distance_pct": exit_cfg.trailing_distance_pct,
            "edge_decay_profit_retention": exit_cfg.edge_decay_profit_retention,
            "edge_decay_min_profit_pct": exit_cfg.edge_decay_min_profit_pct,
            "label_early_fail_candles": fp.get("label_early_fail_candles"),
            "label_early_fail_mae_mfe_ratio": fp.get("label_early_fail_mae_mfe_ratio"),
            "label_early_fail_ret_mfe_ratio": fp.get("label_early_fail_ret_mfe_ratio"),
        }
        digest = hashlib.sha256(
            json.dumps(policy_keys, sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
        base_id = str(freqai.get("identifier", "freqllm") or "freqllm")
        base_id = re.sub(r"-ph[0-9a-f]{8}$", "", base_id)
        freqai["identifier"] = f"{base_id}-ph{digest}"

    def _label_detail_timeframe(self, feature_params: dict[str, Any]) -> str:
        detail_tf = str(feature_params.get("label_detail_timeframe") or "").strip()
        if not detail_tf:
            return ""
        try:
            if timeframe_to_minutes(detail_tf) >= timeframe_to_minutes(self.timeframe):
                return ""
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, OSError):
            return ""
        return detail_tf

    def _get_detail_dataframe(self, pair: str, detail_tf: str) -> DataFrame | None:
        if not pair or not detail_tf or self.dp is None:
            return None
        cache_key = (pair, detail_tf)
        if cache_key in self.strategy_collaborators.runtime.detail_dataframe_cache:
            return self.strategy_collaborators.runtime.detail_dataframe_cache[cache_key]
        try:
            detail = self.dp.get_pair_dataframe(pair=pair, timeframe=detail_tf)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            logger.debug("[%s] Failed to load detail timeframe %s: %s", pair, detail_tf, exc)
            return None
        if detail is None or detail.empty or "date" not in detail:
            return None
        detail = detail.copy()
        detail["date"] = pd.to_datetime(detail["date"], utc=True, errors="coerce")
        detail = detail.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
        self.strategy_collaborators.runtime.detail_dataframe_cache[cache_key] = detail
        return detail

    @staticmethod
    def _indexed_row(indexed: DataFrame, timestamp: pd.Timestamp):
        if timestamp not in indexed.index:
            return None
        row = indexed.loc[timestamp]
        return row.iloc[0] if isinstance(row, DataFrame) else row

    def _record_detail_step(
        self,
        buffers: DetailBuffers,
        position: tuple[int, int],
        row: pd.Series,
    ) -> float:
        row_idx, step_idx = position
        values = {
            "high": self._safe_float(row.get("high"), np.nan),
            "low": self._safe_float(row.get("low"), np.nan),
            "open": self._safe_float(row.get("open"), np.nan),
            "close": self._safe_float(row.get("close"), np.nan),
        }
        for name, value in values.items():
            if np.isfinite(value):
                getattr(buffers, name)[row_idx, step_idx] = value
        return values["close"]

    def _populate_detail_path(
        self,
        source: tuple[DataFrame, DetailBuffers],
        position: tuple[int, pd.Timestamp],
        step_delta: pd.Timedelta,
    ) -> None:
        indexed, buffers = source
        row_idx, entry_time = position
        entry_row = self._indexed_row(indexed, entry_time)
        if entry_row is None:
            return
        buffers.entry[row_idx] = self._safe_float(entry_row.get("open"), np.nan)
        last_close = np.nan
        for step_idx in range(buffers.high.shape[1]):
            step_row = self._indexed_row(indexed, entry_time + step_delta * step_idx)
            if step_row is not None:
                last_close = self._record_detail_step(buffers, (row_idx, step_idx), step_row)
        buffers.terminal[row_idx] = last_close

    def _build_detail_policy_windows(
        self,
        dataframe: DataFrame,
        pair: str,
        detail_tf: str,
        horizon: int,
    ) -> tuple[pd.Series, DataFrame, DataFrame, pd.Series, DataFrame, DataFrame] | None:
        detail = self._get_detail_dataframe(pair, detail_tf)
        if detail is None or detail.empty or "date" not in dataframe:
            return None
        timeframe_minutes = timeframe_to_minutes(self.timeframe)
        detail_minutes = timeframe_to_minutes(detail_tf)
        if detail_minutes <= 0 or timeframe_minutes <= detail_minutes:
            return None
        steps = max(1, round(timeframe_minutes / detail_minutes) * max(1, horizon))
        buffers = DetailBuffers.allocate(len(dataframe), steps)
        indexed = detail.set_index("date", drop=False)
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        for row_idx, date_value in enumerate(dates):
            if pd.notna(date_value):
                entry_time = date_value + pd.Timedelta(minutes=timeframe_minutes)
                self._populate_detail_path(
                    (indexed, buffers),
                    (row_idx, entry_time),
                    pd.Timedelta(minutes=detail_minutes),
                )
        if not np.isfinite(buffers.entry).any():
            return None
        return (
            pd.Series(buffers.entry, index=dataframe.index),
            DataFrame(buffers.high, index=dataframe.index),
            DataFrame(buffers.low, index=dataframe.index),
            pd.Series(buffers.terminal, index=dataframe.index),
            DataFrame(buffers.open, index=dataframe.index),
            DataFrame(buffers.close, index=dataframe.index),
        )

    def _policy_settings(self, options: dict[str, Any]) -> PolicySettings:
        risk = self.strategy_collaborators.simple_config.risk
        exit_cfg = self.strategy_collaborators.simple_config.exit
        retention = float(getattr(exit_cfg, "edge_decay_profit_retention", 0.75) or 0.75)
        early = EarlySettings(
            max(0, int(options.get("early_steps", 0) or 0)),
            max(float(options.get("early_mae_mfe_ratio", 2.0) or 2.0), 1e-6),
            max(float(options.get("early_ret_mfe_ratio", 0.25) or 0.25), 0.0),
        )
        barrier = BarrierSettings(
            max(float(risk.stop_loss_pct or 0.0), 1e-6),
            max(float(risk.take_profit_pct or 0.0), 1e-6),
            bool(getattr(exit_cfg, "trailing_enabled", False)),
            max(float(getattr(exit_cfg, "trailing_activation_pct", 0.0) or 0.0), 1e-6),
            max(float(getattr(exit_cfg, "trailing_distance_pct", 0.0) or 0.0), 1e-6),
        )
        decay = DecaySettings(
            min(max(retention, 0.1), 1.0),
            max(float(getattr(exit_cfg, "edge_decay_min_profit_pct", 0.0) or 0.0), 0.0),
            float(exit_cfg.recovery_band_mult),
            float(exit_cfg.recovery_sl_floor_ratio),
            float(exit_cfg.post_drawdown_band_mult),
            int(exit_cfg.max_hold_candles),
        )
        return PolicySettings(
            str(options["direction"]),
            float(options["round_trip_cost"]),
            max(1, int(options.get("steps_per_candle", 1) or 1)),
            early,
            barrier,
            decay,
        )

    @staticmethod
    def _policy_arrays(entry_price: pd.Series, windows: tuple, options: dict) -> PolicyArrays:
        high, low, terminal = windows
        open_window = options.get("future_open_window")
        close_window = options.get("future_close_window")
        return PolicyArrays(
            entry_price.to_numpy(dtype=float),
            high.to_numpy(dtype=float),
            low.to_numpy(dtype=float),
            terminal.to_numpy(dtype=float),
            open_window.to_numpy(dtype=float) if open_window is not None else None,
            close_window.to_numpy(dtype=float) if close_window is not None else None,
        )

    @staticmethod
    def _step_profits(
        settings: PolicySettings, entry: float, high: float, low: float
    ) -> tuple[float, float]:
        if settings.direction == "short":
            return entry / high - 1.0, entry / low - 1.0
        return low / entry - 1.0, high / entry - 1.0

    @staticmethod
    def _intrabar_events(
        arrays: PolicyArrays,
        settings: PolicySettings,
        position: tuple[int, int],
    ) -> tuple[str, str]:
        if arrays.opens is None or arrays.closes is None:
            return "adverse", "favorable"
        step_open = arrays.opens[position]
        step_close = arrays.closes[position]
        if not np.isfinite(step_open) or not np.isfinite(step_close):
            return "adverse", "favorable"
        bullish = step_close >= step_open
        adverse_first = bullish if settings.direction != "short" else not bullish
        return ("adverse", "favorable") if adverse_first else ("favorable", "adverse")

    @staticmethod
    def _adverse_exit(
        settings: PolicySettings,
        state: PolicyState,
        adverse_profit: float,
    ) -> float | None:
        if state.has_positive_profit:
            state.max_post_profit_drawdown = max(
                state.max_post_profit_drawdown,
                0.0,
                state.peak_profit - adverse_profit,
            )
        else:
            state.min_profit_before_positive = min(state.min_profit_before_positive, adverse_profit)
        if adverse_profit <= -settings.barrier.stop_loss:
            return -settings.barrier.stop_loss
        trailing_inactive = not settings.barrier.trailing_enabled
        activation_pending = state.peak_profit < settings.barrier.trailing_activation
        if trailing_inactive or activation_pending:
            return None
        fixed_stop = state.peak_profit - settings.barrier.trailing_distance
        retention_stop = (
            state.peak_profit * settings.decay.retention
            if state.peak_profit >= settings.decay.min_profit
            else 0.0
        )
        trailing_stop = max(fixed_stop, retention_stop, 0.0)
        return trailing_stop if adverse_profit <= trailing_stop else None

    def _handle_intrabar_event(
        self,
        settings: PolicySettings,
        state: PolicyState,
        event: str,
        profits: tuple[float, float],
    ) -> float | None:
        adverse_profit, favorable_profit = profits
        if event == "adverse":
            return self._adverse_exit(settings, state, adverse_profit)
        state.peak_profit = max(state.peak_profit, favorable_profit)
        state.has_positive_profit = state.has_positive_profit or state.peak_profit > 0
        reached_take_profit = favorable_profit >= settings.barrier.take_profit
        if not settings.barrier.trailing_enabled and reached_take_profit:
            return settings.barrier.take_profit
        return None

    @staticmethod
    def _current_profit(
        arrays: PolicyArrays,
        settings: PolicySettings,
        position: tuple[int, int],
        entry: float,
    ) -> float | None:
        if arrays.closes is None:
            return None
        current_close = arrays.closes[position]
        if not np.isfinite(current_close) or current_close <= 0:
            return None
        if settings.direction == "short":
            return entry / current_close - 1.0
        return current_close / entry - 1.0

    @staticmethod
    def _recovery_exit(
        settings: PolicySettings,
        state: PolicyState,
        step_number: int,
        current_profit: float,
    ) -> float | None:
        if step_number < settings.steps_per_candle or state.min_profit_before_positive >= 0:
            return None
        allowed = max(
            abs(state.min_profit_before_positive) * settings.decay.recovery_band_mult,
            settings.barrier.stop_loss * settings.decay.recovery_sl_floor_ratio,
        )
        return current_profit if current_profit <= -allowed else None

    @staticmethod
    def _post_drawdown_exit(
        settings: PolicySettings,
        state: PolicyState,
        current_profit: float,
    ) -> float | None:
        if not state.has_positive_profit or current_profit <= 0:
            return None
        actual = max(0.0, state.peak_profit - current_profit)
        expected = max(
            state.max_post_profit_drawdown * settings.decay.post_drawdown_band_mult,
            settings.barrier.trailing_distance * settings.decay.post_drawdown_band_mult,
        )
        return current_profit if actual >= expected else None

    @staticmethod
    def _early_failure_exit(
        settings: PolicySettings,
        state: PolicyState,
        step_number: int,
        current_profit: float,
    ) -> float | None:
        if settings.early.steps <= 0 or step_number < settings.early.steps:
            return None
        adverse = abs(min(state.max_adverse_excursion, 0.0))
        favorable = max(state.peak_profit, 0.0)
        failed = (
            adverse > favorable * settings.early.mae_mfe_ratio
            and current_profit < -favorable * settings.early.return_mfe_ratio
        )
        return current_profit if failed and current_profit <= -0.0005 else None

    @staticmethod
    def _no_progress_exit(
        settings: PolicySettings,
        state: PolicyState,
        step_number: int,
        current_profit: float,
    ) -> float | None:
        age = max(3, min(settings.decay.max_hold_candles, 6))
        threshold = age * settings.steps_per_candle
        stalled = state.peak_profit < settings.round_trip_cost + settings.decay.min_profit
        expired = step_number >= threshold and stalled and current_profit <= 0
        return current_profit if expired else None

    def _close_based_exit(
        self,
        settings: PolicySettings,
        state: PolicyState,
        step_number: int,
        current_profit: float | None,
    ) -> float | None:
        if current_profit is None:
            return None
        candidates = (
            self._recovery_exit(settings, state, step_number, current_profit),
            self._post_drawdown_exit(settings, state, current_profit),
            self._early_failure_exit(settings, state, step_number, current_profit),
            self._no_progress_exit(settings, state, step_number, current_profit),
        )
        return next((result for result in candidates if result is not None), None)

    def _simulate_policy_row(
        self, arrays: PolicyArrays, settings: PolicySettings, row_idx: int
    ) -> PolicyState:
        state = PolicyState()
        entry = arrays.entry[row_idx]
        for step_idx in range(arrays.highs.shape[1]):
            high = arrays.highs[row_idx, step_idx]
            low = arrays.lows[row_idx, step_idx]
            if not np.isfinite(high) or not np.isfinite(low) or high <= 0 or low <= 0:
                continue
            profits = self._step_profits(settings, entry, high, low)
            state.max_adverse_excursion = min(state.max_adverse_excursion, profits[0])
            position = (row_idx, step_idx)
            for event in self._intrabar_events(arrays, settings, position):
                state.exit_profit = self._handle_intrabar_event(settings, state, event, profits)
                if state.exit_profit is not None:
                    break
            if state.exit_profit is not None:
                break
            current = self._current_profit(arrays, settings, position, entry)
            state.exit_profit = self._close_based_exit(settings, state, step_idx + 1, current)
            if state.exit_profit is not None:
                break
        return state

    @staticmethod
    def _terminal_profit(arrays: PolicyArrays, settings: PolicySettings, row_idx: int) -> float:
        terminal = arrays.terminal[row_idx]
        entry = arrays.entry[row_idx]
        if not np.isfinite(terminal) or terminal <= 0:
            return np.nan
        if settings.direction == "short":
            return entry / terminal - 1.0
        return terminal / entry - 1.0

    def _store_policy_result(
        self,
        results: PolicyResults,
        state: PolicyState,
        settings: PolicySettings,
        row_idx: int,
    ) -> None:
        exit_profit = state.exit_profit
        if exit_profit is not None and np.isfinite(exit_profit):
            results.returns[row_idx] = exit_profit - settings.round_trip_cost
        results.peaks[row_idx] = state.peak_profit
        results.pre_drawdowns[row_idx] = state.min_profit_before_positive
        results.post_drawdowns[row_idx] = (
            state.max_post_profit_drawdown if state.has_positive_profit else 0.0
        )
        results.adverse_excursions[row_idx] = state.max_adverse_excursion

    def _simulate_policy_return_targets(
        self,
        entry_price: pd.Series,
        *policy_windows,
        **options,
    ) -> DataFrame:
        """Simulate the exit policy and produce path-quality labels."""
        settings = self._policy_settings(options)
        arrays = self._policy_arrays(entry_price, policy_windows, options)
        results = PolicyResults.allocate(len(arrays.entry))
        for row_idx, entry in enumerate(arrays.entry):
            if not np.isfinite(entry) or entry <= 0:
                continue
            state = self._simulate_policy_row(arrays, settings, row_idx)
            if state.exit_profit is None:
                state.exit_profit = self._terminal_profit(arrays, settings, row_idx)
            self._store_policy_result(results, state, settings, row_idx)
        return DataFrame(
            {
                "policy_return": results.returns,
                "peak_profit": results.peaks,
                "mae": results.adverse_excursions,
                "pre_profit_drawdown": results.pre_drawdowns,
                "post_profit_drawdown": results.post_drawdowns,
            },
            index=entry_price.index,
        )

    @staticmethod
    def _early_window(entry_price: pd.Series, windows: tuple, early_steps: int) -> EarlyWindow:
        high_window, low_window, close_window = windows
        steps = max(1, min(int(early_steps or 1), high_window.shape[1]))
        return EarlyWindow(
            entry_price.replace(0, np.nan),
            high_window.iloc[:, :steps].max(axis=1).replace(0, np.nan),
            low_window.iloc[:, :steps].min(axis=1).replace(0, np.nan),
            close_window.iloc[:, steps - 1].replace(0, np.nan),
        )

    @staticmethod
    def _directional_excursions(window: EarlyWindow, direction: str) -> Excursions:
        if direction == "short":
            return Excursions(
                (window.high / window.entry - 1.0).clip(lower=0.0),
                (window.entry / window.low - 1.0).clip(lower=0.0),
                window.entry / window.close - 1.0,
            )
        return Excursions(
            (1.0 - window.low / window.entry).clip(lower=0.0),
            (window.high / window.entry - 1.0).clip(lower=0.0),
            window.close / window.entry - 1.0,
        )

    def _early_fail_risk_targets(
        self,
        entry_price: pd.Series,
        *future_windows,
        **options,
    ) -> pd.Series:
        """Classify early failures using relative adverse/favorable path shape."""
        window = self._early_window(entry_price, future_windows, int(options["early_steps"]))
        excursions = self._directional_excursions(window, str(options["direction"]))
        mae_ratio = max(float(options["mae_mfe_ratio"] or 1.0), 1e-6)
        return_ratio = max(float(options["ret_mfe_ratio"] or 0.0), 0.0)
        failed = (excursions.adverse > excursions.favorable * mae_ratio) & (
            excursions.close < -excursions.favorable * return_ratio
        )
        result = pd.Series(np.where(failed, 1.0, 0.0), index=entry_price.index)
        valid = (
            entry_price.notna()
            & window.close.notna()
            & excursions.adverse.notna()
            & excursions.favorable.notna()
            & excursions.close.notna()
        )
        return result.where(valid).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _future_price_windows(dataframe: DataFrame, horizon: int) -> FuturePrices:
        shifts = range(1, horizon + 1)
        return FuturePrices(
            pd.concat([dataframe["close"].shift(-step) for step in shifts], axis=1),
            pd.concat([dataframe["high"].shift(-step) for step in shifts], axis=1),
            pd.concat([dataframe["low"].shift(-step) for step in shifts], axis=1),
        )

    def _detail_early_steps(self, detail_tf: str, early_candles: int) -> int:
        if not detail_tf:
            return early_candles
        try:
            scale = round(timeframe_to_minutes(self.timeframe) / timeframe_to_minutes(detail_tf))
        except _TARGET_ERRORS:
            return early_candles
        return early_candles * max(1, scale)

    def _target_windows(
        self,
        dataframe: DataFrame,
        pair: str,
        params: dict[str, Any],
        horizon: int,
    ) -> TargetWindows:
        future = self._future_price_windows(dataframe, horizon)
        open_window = pd.concat(
            [dataframe["open"].shift(-step) for step in range(1, horizon + 1)],
            axis=1,
        )
        entry = dataframe["open"].shift(-1).replace(0, np.nan)
        terminal = dataframe["close"].shift(-horizon)
        detail_tf = self._label_detail_timeframe(params)
        detail = (
            self._build_detail_policy_windows(dataframe, pair, detail_tf, horizon)
            if detail_tf
            else None
        )
        if detail is not None:
            entry, future_high, future_low, terminal, open_window, step_close = detail
        else:
            future_high, future_low, step_close = future.high, future.low, future.close
        timing = WindowTiming(
            self._detail_early_steps(
                detail_tf,
                max(
                    1,
                    min(
                        horizon,
                        int(self._safe_float(params.get("label_early_fail_candles"), 4) or 4),
                    ),
                ),
            ),
            max(1, int(future_high.shape[1] / max(horizon, 1))),
        )
        return TargetWindows(
            entry, future_high, future_low, terminal, open_window, step_close, timing
        )

    def _direction_policy_targets(
        self,
        windows: TargetWindows,
        direction: str,
        round_trip_cost: float,
        risk_ratios: tuple[float, float],
    ) -> tuple[DataFrame, pd.Series]:
        mae_ratio, return_ratio = risk_ratios
        early_fail = self._early_fail_risk_targets(
            windows.entry,
            windows.high,
            windows.low,
            windows.step_close,
            direction=direction,
            early_steps=windows.timing.early_steps,
            mae_mfe_ratio=mae_ratio,
            ret_mfe_ratio=return_ratio,
        )
        policy = self._simulate_policy_return_targets(
            windows.entry,
            windows.high,
            windows.low,
            windows.close,
            direction=direction,
            round_trip_cost=round_trip_cost,
            future_open_window=windows.open,
            future_close_window=windows.step_close,
            steps_per_candle=windows.timing.steps_per_candle,
            early_steps=windows.timing.early_steps,
            early_mae_mfe_ratio=mae_ratio,
            early_ret_mfe_ratio=return_ratio,
        )
        return policy, early_fail

    def _policy_targets(
        self,
        windows: TargetWindows,
        round_trip_cost: float,
        params: dict[str, Any],
    ) -> PolicyTargets:
        ratios = (
            self._safe_float(params.get("label_early_fail_mae_mfe_ratio"), 2.0),
            self._safe_float(params.get("label_early_fail_ret_mfe_ratio"), 0.25),
        )
        long_policy, long_fail = self._direction_policy_targets(
            windows, "long", round_trip_cost, ratios
        )
        short_policy, short_fail = self._direction_policy_targets(
            windows, "short", round_trip_cost, ratios
        )
        return PolicyTargets(long_policy, short_policy, long_fail, short_fail)

    def _direction_target(
        self,
        index: pd.Index,
        policies: PolicyTargets,
        params: dict[str, Any],
    ) -> pd.Series:
        long_return = policies.long["policy_return"].replace([np.inf, -np.inf], np.nan)
        short_return = policies.short["policy_return"].replace([np.inf, -np.inf], np.nan)
        margin = max(0.0, self._safe_float(params.get("label_class_margin"), 0.0))
        gap = max(
            0.0,
            self._safe_float(
                params.get("label_class_edge_gap"),
                self.strategy_collaborators.simple_config.signal.min_edge_gap,
            ),
        )
        target = pd.Series(0.0, index=index)
        target = target.mask((long_return > margin) & (long_return - short_return >= gap), 1.0)
        target = target.mask((short_return > margin) & (short_return - long_return >= gap), -1.0)
        return target.where(long_return.notna() & short_return.notna(), np.nan)

    def _level_context(self, dataframe: DataFrame, params: dict[str, Any]) -> LevelContext:
        missing = pd.Series(np.nan, index=dataframe.index)
        nearest_resistance = dataframe.get(
            "%%-nearest_resistance_price",
            dataframe.get("%-nearest_resistance_price", missing),
        )
        nearest_support = dataframe.get(
            "%%-nearest_support_price",
            dataframe.get("%-nearest_support_price", missing),
        )
        strong_resistance = dataframe.get(
            "%%-strongest_resistance_price", nearest_resistance
        ).combine_first(nearest_resistance)
        strong_support = dataframe.get("%%-strongest_support_price", nearest_support).combine_first(
            nearest_support
        )
        atr = ta.ATR(dataframe, timeperiod=14)
        buffer_ratio = max(
            0.0,
            self._safe_float(params.get("label_level_break_buffer_atr"), 0.10),
        )
        return LevelContext(
            nearest_resistance,
            nearest_support,
            strong_resistance,
            strong_support,
            atr * buffer_ratio,
            self._safe_ratio(strong_resistance - dataframe["close"], atr) <= 0.75,
            self._safe_ratio(dataframe["close"] - strong_support, atr) <= 0.75,
        )

    @staticmethod
    def _level_masks(future: FuturePrices, levels: LevelContext) -> dict[str, pd.Series]:
        terminal_close = future.close.iloc[:, -1]
        close_max = future.close.max(axis=1)
        close_min = future.close.min(axis=1)
        touched_support = future.low.min(axis=1) <= levels.strong_support + levels.buffer
        touched_resistance = future.high.max(axis=1) >= levels.strong_resistance - levels.buffer
        return {
            "broke_up": (close_max > levels.strong_resistance + levels.buffer)
            & (terminal_close > levels.strong_resistance),
            "broke_down": (close_min < levels.strong_support - levels.buffer)
            & (terminal_close < levels.strong_support),
            "false_up": (future.high.max(axis=1) > levels.strong_resistance + levels.buffer)
            & (terminal_close <= levels.strong_resistance),
            "false_down": (future.low.min(axis=1) < levels.strong_support - levels.buffer)
            & (terminal_close >= levels.strong_support),
            "bounce": touched_support & (terminal_close > levels.strong_support + levels.buffer),
            "reject": touched_resistance
            & (terminal_close < levels.strong_resistance - levels.buffer),
        }

    def _level_event_target(
        self, dataframe: DataFrame, params: dict[str, Any], horizon: int
    ) -> pd.Series:
        future = self._future_price_windows(dataframe, horizon)
        levels = self._level_context(dataframe, params)
        masks = self._level_masks(future, levels)
        event = pd.Series(0.0, index=dataframe.index)
        event = event.mask(masks["broke_up"] & ~masks["broke_down"], 1.0)
        event = event.mask(masks["broke_down"] & ~masks["broke_up"], -1.0)
        event = event.mask(masks["false_up"] & levels.near_resistance, -2.0)
        event = event.mask(masks["false_down"] & levels.near_support, 2.0)
        event = event.mask(masks["bounce"] & levels.near_support, 3.0)
        event = event.mask(masks["reject"] & levels.near_resistance, -3.0)
        valid = (
            future.close.iloc[:, -1].notna()
            & levels.nearest_resistance.notna()
            & levels.nearest_support.notna()
            & levels.strong_resistance.notna()
            & levels.strong_support.notna()
        )
        return event.where(valid, np.nan)

    def _assign_policy_targets(
        self,
        dataframe: DataFrame,
        policies: PolicyTargets,
        direction: pd.Series,
        level_event: pd.Series,
    ) -> DataFrame:
        columns = self.freqai_target_columns
        dataframe[columns["dir_class"]] = direction
        for name, value in (("long", 1.0), ("flat", 0.0), ("short", -1.0)):
            dataframe[columns[f"{name}_probability"]] = (
                (direction == value).astype(float).where(direction.notna())
            )
        for side, policy, early_fail in (
            ("long", policies.long, policies.long_early_fail),
            ("short", policies.short, policies.short_early_fail),
        ):
            policy_return = policy["policy_return"].replace([np.inf, -np.inf], np.nan)
            for quantile in ("q20", "q50", "q80"):
                dataframe[columns[f"{side}_edge_{quantile}"]] = policy_return
            for field in _POLICY_TARGET_FIELDS:
                dataframe[columns[f"{side}_{field}"]] = policy[field].replace(
                    [np.inf, -np.inf], np.nan
                )
            dataframe[columns[f"{side}_early_fail_risk"]] = early_fail.replace(
                [np.inf, -np.inf], np.nan
            )
        dataframe[columns["level_event_class"]] = level_event
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """Create policy-aligned targets with barrier-consistent regression returns."""
        callback_metadata = kwargs.get("metadata")
        if isinstance(callback_metadata, dict):
            metadata = {**metadata, **callback_metadata}
        params = self.config.get("freqai", {}).get("feature_parameters", {})
        horizon = self._label_horizon(params)
        pair = str(metadata.get("pair", "") or "")
        windows = self._target_windows(dataframe, pair, params, horizon)
        cost = self._configured_round_trip_cost(pair, params)
        policies = self._policy_targets(windows, cost, params)
        direction = self._direction_target(dataframe.index, policies, params)
        level_event = self._level_event_target(dataframe, params, horizon)
        return self._assign_policy_targets(dataframe, policies, direction, level_event)
