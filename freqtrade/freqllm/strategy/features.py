"""FreqAI feature engineering and target-generation mixin."""

import logging
from collections import namedtuple

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.freqllm.strategy.contracts import StrategyCollaborators


logger = logging.getLogger(__name__)


_LevelInputs = namedtuple("_LevelInputs", "close high low previous_close atr magnitude")
_PsychologicalLevel = namedtuple("_PsychologicalLevel", "lower upper reference nearest")
_LevelFrames = namedtuple("_LevelFrames", "resistance support nearest_resistance nearest_support")
_StrengthFrames = namedtuple("_StrengthFrames", "resistance support")


class StrategyFreqaiMixin:
    """Build causal FreqAI features from candles and optional market context."""

    strategy_collaborators: StrategyCollaborators

    def _add_trend_features(self, dataframe: DataFrame, period: int) -> None:
        close = dataframe["close"]
        volume = dataframe["volume"]
        dataframe[f"%-rsi-{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-ema-{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe[f"%-adx-{period}"] = ta.ADX(dataframe, timeperiod=period)
        dataframe[f"%-atr_ratio-{period}"] = ta.ATR(dataframe, timeperiod=period) / close
        dataframe[f"%-volume_rel-{period}"] = self._safe_ratio(
            volume, volume.rolling(period).mean()
        )
        dataframe[f"%-mfi-{period}"] = ta.MFI(dataframe, timeperiod=period)
        obv = dataframe["%-obv"] if "%-obv" in dataframe.columns else ta.OBV(dataframe)
        dataframe[f"%-obv_slope-{period}"] = self._safe_ratio(
            obv.diff(period), volume.rolling(period).mean()
        )
        ema = dataframe[f"%-ema-{period}"]
        dataframe[f"%-price_to_ema-{period}"] = self._safe_ratio(close - ema, ema)
        dataframe[f"%-ema_slope-{period}"] = self._safe_ratio(ema.diff(period), close)
        dataframe[f"%-adx_slope-{period}"] = dataframe[f"%-adx-{period}"].diff(period)

    def _add_range_features(self, dataframe: DataFrame, period: int) -> None:
        close = dataframe["close"]
        prior_high = dataframe["high"].shift(1).rolling(period).max()
        prior_low = dataframe["low"].shift(1).rolling(period).min()
        dataframe[f"%-distance_to_high-{period}"] = self._safe_ratio(close - prior_high, close)
        dataframe[f"%-distance_to_low-{period}"] = self._safe_ratio(close - prior_low, close)
        dataframe[f"%-price_percentile-{period}"] = self._safe_ratio(
            close - prior_low, prior_high - prior_low
        )
        bands = ta.BBANDS(dataframe, timeperiod=period, nbdevup=2.0, nbdevdn=2.0)
        dataframe[f"%-bb_width-{period}"] = self._safe_ratio(
            bands["upperband"] - bands["lowerband"], bands["middleband"]
        )
        dataframe[f"%-bb_position-{period}"] = self._safe_ratio(
            close - bands["lowerband"], bands["upperband"] - bands["lowerband"]
        )
        channel = dataframe["high"].rolling(period).max() - dataframe["low"].rolling(period).min()
        dataframe[f"%-donchian_width-{period}"] = self._safe_ratio(channel, close)

    def _add_volatility_features(self, dataframe: DataFrame, period: int) -> None:
        close = dataframe["close"]
        changes = dataframe["%-pct_change"] if "%-pct_change" in dataframe else close.pct_change()
        dataframe[f"%-realized_vol-{period}"] = changes.rolling(period).std()
        atr_ratio = dataframe[f"%-atr_ratio-{period}"]
        dataframe[f"%-atr_percentile-{period}"] = self._safe_ratio(
            atr_ratio - atr_ratio.rolling(period).min(),
            atr_ratio.rolling(period).max() - atr_ratio.rolling(period).min(),
        )
        dataframe[f"%-volume_zscore-{period}"] = self._zscore_series(dataframe["volume"], period)

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """FreqAI auto-expanded features for multi-timeframe training."""
        callback_metadata = kwargs.get("metadata")
        if isinstance(callback_metadata, dict):
            metadata = {**metadata, **callback_metadata}
        if metadata.get("pair"):
            logger.debug("Expanding period %s features for %s", period, metadata["pair"])
        self._add_trend_features(dataframe, period)
        self._add_range_features(dataframe, period)
        self._add_volatility_features(dataframe, period)
        return dataframe

    def _add_basic_candle_features(self, dataframe: DataFrame) -> None:
        candle_range = (dataframe["high"] - dataframe["low"]).replace(0, pd.NA)
        dataframe["%-pct_change"] = dataframe["close"].pct_change()
        dataframe["%-hl_spread"] = candle_range / dataframe["close"]
        dataframe["%-body_to_range"] = (dataframe["close"] - dataframe["open"]) / candle_range
        dataframe["%-close_location"] = self._safe_ratio(
            dataframe["close"] - dataframe["low"], candle_range
        )
        dataframe["%-upper_wick_to_range"] = self._safe_ratio(
            dataframe[["open", "close"]].max(axis=1) - dataframe["high"], candle_range
        ).abs()
        dataframe["%-lower_wick_to_range"] = self._safe_ratio(
            dataframe[["open", "close"]].min(axis=1) - dataframe["low"], candle_range
        ).abs()
        for lag in (1, 2, 4):
            dataframe[f"%-impulse_ret_{lag}"] = dataframe["close"].pct_change(lag)

    def _add_basic_breakout_features(self, dataframe: DataFrame) -> None:
        for window in (4, 8, 16):
            previous_high = dataframe["high"].shift(1).rolling(window).max()
            previous_low = dataframe["low"].shift(1).rolling(window).min()
            dataframe[f"%-breakout_high_distance_{window}"] = self._safe_ratio(
                dataframe["close"] - previous_high, dataframe["close"]
            )
            dataframe[f"%-breakout_low_distance_{window}"] = self._safe_ratio(
                previous_low - dataframe["close"], dataframe["close"]
            )

    def _add_basic_volume_features(self, dataframe: DataFrame) -> None:
        typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3.0
        money_flow = typical_price * dataframe["volume"]
        flow_multiplier = (
            dataframe["close"] - dataframe["low"] - dataframe["high"] + dataframe["close"]
        )
        cmf_numerator = (flow_multiplier * dataframe["volume"]).rolling(20).sum()
        volume_sum = dataframe["volume"].rolling(20).sum()
        volume_quantile = dataframe["volume"].rolling(20).quantile(0.8)
        dataframe["%-volume_breakout_pct"] = self._safe_ratio(
            dataframe["volume"] - volume_quantile, volume_quantile
        )
        dataframe["%-obv"] = ta.OBV(dataframe["close"], dataframe["volume"])
        dataframe["%-cmf"] = self._safe_ratio(cmf_numerator, volume_sum)
        rolling_vwap = self._safe_ratio(money_flow.rolling(20).sum(), volume_sum)
        dataframe["%-vwap_deviation"] = self._safe_ratio(
            dataframe["close"] - rolling_vwap, dataframe["close"]
        )
        for period in (3, 12):
            dataframe[f"%-roc_{period}"] = dataframe["close"].pct_change(period)
            dataframe[f"%-volume_trend_{period}"] = (
                self._safe_ratio(dataframe["volume"], dataframe["volume"].shift(period)) - 1.0
            )

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """Base FreqAI features shared across configured timeframe expansions."""
        callback_metadata = kwargs.get("metadata")
        if isinstance(callback_metadata, dict):
            metadata = {**metadata, **callback_metadata}
        self._add_basic_candle_features(dataframe)
        self._add_basic_breakout_features(dataframe)
        self._add_basic_volume_features(dataframe)
        return self._inject_external_market_features(dataframe, metadata)

    @staticmethod
    def _detail_feature_names() -> tuple[str, ...]:
        return (
            "%-detail_net_ret",
            "%-detail_first_ret",
            "%-detail_last_ret",
            "%-detail_direction_consistency",
            "%-detail_late_reversal",
            "%-detail_close_location",
            "%-detail_range_ratio",
            "%-detail_up_move",
            "%-detail_down_move",
        )

    def _detail_candle_return(self, row: pd.Series) -> float:
        open_value = self._safe_float(row.get("open"), np.nan)
        close_value = self._safe_float(row.get("close"), np.nan)
        if not np.isfinite(open_value) or open_value == 0 or not np.isfinite(close_value):
            return np.nan
        return close_value / open_value - 1.0

    def _detail_window_values(self, window: DataFrame) -> dict[str, float]:
        values = {name: np.nan for name in self._detail_feature_names()}
        if window.empty:
            return values
        first_open = self._safe_float(window.iloc[0].get("open"), np.nan)
        last_close = self._safe_float(window.iloc[-1].get("close"), np.nan)
        high_value = self._safe_float(window["high"].max(), np.nan)
        low_value = self._safe_float(window["low"].min(), np.nan)
        first_return = self._detail_candle_return(window.iloc[0])
        last_return = self._detail_candle_return(window.iloc[-1])
        if np.isfinite(first_open) and first_open > 0:
            values["%-detail_net_ret"] = last_close / first_open - 1.0
            values["%-detail_range_ratio"] = (high_value - low_value) / first_open
            values["%-detail_up_move"] = high_value / first_open - 1.0
            values["%-detail_down_move"] = first_open / low_value - 1.0
        if high_value > low_value and np.isfinite(last_close):
            values["%-detail_close_location"] = (last_close - low_value) / (high_value - low_value)
        returns = self._safe_ratio(window["close"] - window["open"], window["open"])
        values["%-detail_first_ret"] = first_return
        values["%-detail_last_ret"] = last_return
        values["%-detail_direction_consistency"] = np.sign(returns.fillna(0.0)).mean()
        if np.isfinite(first_return) and np.isfinite(last_return):
            values["%-detail_late_reversal"] = last_return - first_return
        return values

    def _detail_values_for_date(
        self, detail: DataFrame, date_value: pd.Timestamp, timeframe_delta: pd.Timedelta
    ) -> dict[str, float]:
        if pd.isna(date_value):
            return {name: np.nan for name in self._detail_feature_names()}
        window = detail.loc[
            (detail.index >= date_value) & (detail.index < date_value + timeframe_delta)
        ]
        return self._detail_window_values(window)

    def _inject_detail_timeframe_features(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        feature_params = self.config.get("freqai", {}).get("feature_parameters", {})
        detail_tf = self._label_detail_timeframe(feature_params)
        pair = str(metadata.get("pair", "") or "") if isinstance(metadata, dict) else ""
        detail = self._get_detail_dataframe(pair, detail_tf) if detail_tf else None
        if detail is None or detail.empty or "date" not in dataframe:
            return dataframe
        timeframe_minutes = timeframe_to_minutes(self.timeframe)
        detail_minutes = timeframe_to_minutes(detail_tf)
        if detail_minutes <= 0 or timeframe_minutes <= detail_minutes:
            return dataframe
        prepared = detail.copy()
        prepared["date"] = pd.to_datetime(prepared["date"], utc=True, errors="coerce")
        prepared = prepared.dropna(subset=["date"]).sort_values("date").set_index("date")
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        delta = pd.Timedelta(minutes=timeframe_minutes)
        rows = [self._detail_values_for_date(prepared, date_value, delta) for date_value in dates]
        for name in self._detail_feature_names():
            dataframe[name] = pd.Series((row[name] for row in rows), index=dataframe.index)
        return dataframe

    @staticmethod
    def _previous_period_levels(dataframe: DataFrame, frequency: str) -> DataFrame:
        """Return completed previous-period OHLC levels without using the active period."""
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        indexed = dataframe.assign(_level_date=dates).dropna(subset=["_level_date"])
        indexed = indexed.set_index("_level_date")
        result = DataFrame(
            np.nan,
            index=dataframe.index,
            columns=["high", "low", "close"],
            dtype=float,
        )
        if indexed.empty:
            return result
        completed = (
            indexed[["high", "low", "close"]]
            .resample(frequency, label="left", closed="left")
            .agg({"high": "max", "low": "min", "close": "last"})
            .shift(1)
        )
        valid = dates.notna()
        if valid.any() and not completed.empty:
            aligned = completed.reindex(pd.DatetimeIndex(dates[valid]), method="ffill")
            result.loc[valid, ["high", "low", "close"]] = aligned.to_numpy()
        return result

    @staticmethod
    def _level_inputs(dataframe: DataFrame) -> _LevelInputs:
        close = dataframe["close"].replace(0, np.nan)
        atr = ta.ATR(dataframe, timeperiod=14).replace(0, np.nan)
        magnitude = pd.Series(
            np.power(10.0, np.floor(np.log10(close.where(close > 0)))),
            index=dataframe.index,
        )
        return _LevelInputs(
            close,
            dataframe["high"],
            dataframe["low"],
            close.shift(1),
            atr,
            magnitude,
        )

    @staticmethod
    def _psychological_geometry(fraction: float, inputs: _LevelInputs) -> _PsychologicalLevel:
        step = (inputs.magnitude * fraction).replace(0, np.nan)
        scaled_close = inputs.close / step
        rounded_close = np.round(scaled_close)
        exactly_on_level = (scaled_close - rounded_close).abs() < 1e-9
        lower = (np.floor(scaled_close) - exactly_on_level.astype(float)) * step
        upper = (np.ceil(scaled_close) + exactly_on_level.astype(float)) * step
        reference = np.round(inputs.previous_close / step) * step
        return _PsychologicalLevel(lower, upper, reference, rounded_close * step)

    def _add_psychological_features(
        self,
        dataframe: DataFrame,
        name: str,
        inputs: _LevelInputs,
        level: _PsychologicalLevel,
    ) -> None:
        break_up = (inputs.previous_close <= level.reference) & (inputs.close > level.reference)
        break_down = (inputs.previous_close >= level.reference) & (inputs.close < level.reference)
        touch = (inputs.low <= level.reference) & (inputs.high >= level.reference)
        recent_break_up = break_up.shift(1).rolling(8, min_periods=1).max().fillna(0).astype(bool)
        recent_break_down = (
            break_down.shift(1).rolling(8, min_periods=1).max().fillna(0).astype(bool)
        )
        dataframe[f"%-psych_{name}_nearest_signed_atr"] = self._safe_ratio(
            inputs.close - level.nearest, inputs.atr
        )
        dataframe[f"%-psych_{name}_support_atr"] = self._safe_ratio(
            inputs.close - level.lower, inputs.atr
        )
        dataframe[f"%-psych_{name}_resistance_atr"] = self._safe_ratio(
            level.upper - inputs.close, inputs.atr
        )
        dataframe[f"%-psych_{name}_proximity"] = np.exp(
            -self._safe_ratio((inputs.close - level.nearest).abs(), inputs.atr).clip(lower=0.0)
        )
        dataframe[f"%-psych_{name}_break_up"] = break_up.astype(float)
        dataframe[f"%-psych_{name}_break_down"] = break_down.astype(float)
        dataframe[f"%-psych_{name}_touch"] = touch.astype(float)
        dataframe[f"%-psych_{name}_touch_count_32"] = (
            touch.shift(1).rolling(32, min_periods=1).sum()
        )
        dataframe[f"%-psych_{name}_retest_from_above"] = (
            recent_break_up & (inputs.low <= level.reference) & (inputs.close > level.reference)
        ).astype(float)
        dataframe[f"%-psych_{name}_retest_from_below"] = (
            recent_break_down & (inputs.high >= level.reference) & (inputs.close < level.reference)
        ).astype(float)
        dataframe[f"%-psych_{name}_upper_rejection"] = self._safe_ratio(
            (level.upper - inputs.close).clip(lower=0.0), inputs.atr
        ).where(inputs.high >= level.upper, 0.0)
        dataframe[f"%-psych_{name}_lower_rejection"] = self._safe_ratio(
            (inputs.close - level.lower).clip(lower=0.0), inputs.atr
        ).where(inputs.low <= level.lower, 0.0)

    def _add_psychological_levels(
        self,
        dataframe: DataFrame,
        inputs: _LevelInputs,
        support_candidates: list[pd.Series],
        resistance_candidates: list[pd.Series],
    ) -> None:
        for name, fraction in (("fine", 0.01), ("medium", 0.05), ("major", 0.10)):
            level = self._psychological_geometry(fraction, inputs)
            self._add_psychological_features(dataframe, name, inputs, level)
            support_candidates.append(level.lower)
            resistance_candidates.append(level.upper)

    def _add_rolling_levels(
        self,
        dataframe: DataFrame,
        inputs: _LevelInputs,
        support_candidates: list[pd.Series],
        resistance_candidates: list[pd.Series],
    ) -> None:
        for window in (16, 32, 96, 192):
            prior_high = inputs.high.shift(1).rolling(window).max()
            prior_low = inputs.low.shift(1).rolling(window).min()
            dataframe[f"%-prior_high_distance_atr_{window}"] = self._safe_ratio(
                prior_high - inputs.close, inputs.atr
            )
            dataframe[f"%-prior_low_distance_atr_{window}"] = self._safe_ratio(
                inputs.close - prior_low, inputs.atr
            )
            resistance_candidates.append(prior_high.where(prior_high >= inputs.close))
            support_candidates.append(prior_low.where(prior_low <= inputs.close))

    def _add_previous_period_features(
        self,
        dataframe: DataFrame,
        inputs: _LevelInputs,
        support_candidates: list[pd.Series],
        resistance_candidates: list[pd.Series],
    ) -> None:
        periods = (
            ("previous_day", self._previous_period_levels(dataframe, "1D")),
            ("previous_week", self._previous_period_levels(dataframe, "W-MON")),
        )
        for name, levels in periods:
            pivot = (levels["high"] + levels["low"] + levels["close"]) / 3.0
            dataframe[f"%-{name}_high_distance_atr"] = self._safe_ratio(
                levels["high"] - inputs.close, inputs.atr
            )
            dataframe[f"%-{name}_low_distance_atr"] = self._safe_ratio(
                inputs.close - levels["low"], inputs.atr
            )
            dataframe[f"%-{name}_pivot_distance_atr"] = self._safe_ratio(
                inputs.close - pivot, inputs.atr
            )
            resistance_candidates.extend(
                [
                    levels["high"].where(levels["high"] >= inputs.close),
                    pivot.where(pivot >= inputs.close),
                ]
            )
            support_candidates.extend(
                [
                    levels["low"].where(levels["low"] <= inputs.close),
                    pivot.where(pivot <= inputs.close),
                ]
            )

    def _add_confirmed_pivot_features(
        self,
        dataframe: DataFrame,
        inputs: _LevelInputs,
        support_candidates: list[pd.Series],
        resistance_candidates: list[pd.Series],
    ) -> None:
        pivot_radius = 2
        confirmed_high = (
            inputs.high.shift(pivot_radius)
            .where(
                inputs.high.shift(pivot_radius) == inputs.high.rolling(2 * pivot_radius + 1).max()
            )
            .ffill()
        )
        confirmed_low = (
            inputs.low.shift(pivot_radius)
            .where(inputs.low.shift(pivot_radius) == inputs.low.rolling(2 * pivot_radius + 1).min())
            .ffill()
        )
        dataframe["%-confirmed_pivot_high_distance_atr"] = self._safe_ratio(
            confirmed_high - inputs.close, inputs.atr
        )
        dataframe["%-confirmed_pivot_low_distance_atr"] = self._safe_ratio(
            inputs.close - confirmed_low, inputs.atr
        )
        resistance_candidates.append(confirmed_high.where(confirmed_high >= inputs.close))
        support_candidates.append(confirmed_low.where(confirmed_low <= inputs.close))

    def _add_nearest_level_features(
        self,
        dataframe: DataFrame,
        inputs: _LevelInputs,
        support_candidates: list[pd.Series],
        resistance_candidates: list[pd.Series],
    ) -> _LevelFrames:
        resistance_frame = pd.concat(resistance_candidates, axis=1, ignore_index=True)
        support_frame = pd.concat(support_candidates, axis=1, ignore_index=True)
        nearest_resistance = resistance_frame.min(axis=1, skipna=True)
        nearest_support = support_frame.max(axis=1, skipna=True)
        resistance_room = self._safe_ratio(nearest_resistance - inputs.close, inputs.atr).clip(
            lower=0.0
        )
        support_room = self._safe_ratio(inputs.close - nearest_support, inputs.atr).clip(lower=0.0)
        dataframe["%%-nearest_resistance_price"] = nearest_resistance
        dataframe["%%-nearest_support_price"] = nearest_support
        dataframe["%%-nearest_resistance_room_atr"] = resistance_room
        dataframe["%%-nearest_support_room_atr"] = support_room
        dataframe["%%-nearest_resistance_room_pct"] = self._safe_ratio(
            nearest_resistance - inputs.close, inputs.close
        ).clip(lower=0.0)
        dataframe["%%-nearest_support_room_pct"] = self._safe_ratio(
            inputs.close - nearest_support, inputs.close
        ).clip(lower=0.0)
        dataframe["%-nearest_resistance_room_atr_model"] = resistance_room
        dataframe["%-nearest_support_room_atr_model"] = support_room
        dataframe["%-nearest_resistance_room_pct_model"] = dataframe[
            "%%-nearest_resistance_room_pct"
        ]
        dataframe["%-nearest_support_room_pct_model"] = dataframe["%%-nearest_support_room_pct"]
        tolerance = inputs.atr * 0.25
        dataframe["%-resistance_confluence"] = (
            resistance_frame.sub(nearest_resistance, axis=0).abs().le(tolerance, axis=0).sum(axis=1)
        )
        dataframe["%-support_confluence"] = (
            support_frame.sub(nearest_support, axis=0).abs().le(tolerance, axis=0).sum(axis=1)
        )
        dataframe["%-nearest_resistance_rejection"] = self._safe_ratio(
            (nearest_resistance - inputs.close).clip(lower=0.0), inputs.atr
        ).where(inputs.high >= nearest_resistance, 0.0)
        dataframe["%-nearest_support_rejection"] = self._safe_ratio(
            (inputs.close - nearest_support).clip(lower=0.0), inputs.atr
        ).where(inputs.low <= nearest_support, 0.0)
        return _LevelFrames(resistance_frame, support_frame, nearest_resistance, nearest_support)

    @staticmethod
    def _strong_level_scores(inputs: _LevelInputs, frames: _LevelFrames) -> _StrengthFrames:
        source_weights = [1.0] * 3 + [2.0] * 4 + [3.0] * 4 + [4.0]
        resistance_distance = (
            frames.resistance.sub(inputs.close, axis=0).div(inputs.atr, axis=0).abs()
        )
        support_distance = frames.support.rsub(inputs.close, axis=0).div(inputs.atr, axis=0).abs()
        resistance_scores = np.exp(-resistance_distance.clip(lower=0.0)).mul(source_weights, axis=1)
        support_scores = np.exp(-support_distance.clip(lower=0.0)).mul(source_weights, axis=1)
        resistance_scores = resistance_scores.where(
            frames.resistance.notna()
            & frames.resistance.ge(inputs.close, axis=0)
            & (resistance_distance <= 2.0),
            np.nan,
        )
        support_scores = support_scores.where(
            frames.support.notna()
            & frames.support.le(inputs.close, axis=0)
            & (support_distance <= 2.0),
            np.nan,
        )
        return _StrengthFrames(resistance_scores, support_scores)

    @staticmethod
    def _strongest_level_series(
        index: pd.Index, candidates: DataFrame, scores: DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        rank = scores.replace(np.nan, -np.inf)
        columns = rank.idxmax(axis=1).where(rank.max(axis=1) > -np.inf)
        strongest = pd.Series(np.nan, index=index)
        strength = pd.Series(np.nan, index=index)
        for row_index in index:
            column = columns.loc[row_index]
            if pd.notna(column):
                strongest.loc[row_index] = candidates.loc[row_index, column]
                strength.loc[row_index] = scores.loc[row_index, column]
        return strongest, strength

    def _add_strong_level_features(
        self, dataframe: DataFrame, inputs: _LevelInputs, frames: _LevelFrames
    ) -> None:
        scores = self._strong_level_scores(inputs, frames)
        strongest_resistance, resistance_strength = self._strongest_level_series(
            dataframe.index, frames.resistance, scores.resistance
        )
        strongest_support, support_strength = self._strongest_level_series(
            dataframe.index, frames.support, scores.support
        )
        resistance_room = self._safe_ratio(strongest_resistance - inputs.close, inputs.atr).clip(
            lower=0.0
        )
        support_room = self._safe_ratio(inputs.close - strongest_support, inputs.atr).clip(
            lower=0.0
        )
        dataframe["%%-strongest_resistance_price"] = strongest_resistance
        dataframe["%%-strongest_support_price"] = strongest_support
        dataframe["%%-strongest_resistance_room_atr"] = resistance_room
        dataframe["%%-strongest_support_room_atr"] = support_room
        dataframe["%-strongest_resistance_room_atr_model"] = resistance_room
        dataframe["%-strongest_support_room_atr_model"] = support_room
        dataframe["%-strongest_resistance_strength"] = resistance_strength
        dataframe["%-strongest_support_strength"] = support_strength
        source_weights = [1.0] * 3 + [2.0] * 4 + [3.0] * 4 + [4.0]
        dataframe["%-resistance_strength_confluence"] = (
            frames.resistance.sub(strongest_resistance, axis=0)
            .abs()
            .le(inputs.atr * 0.25, axis=0)
            .mul(source_weights, axis=1)
            .sum(axis=1)
        )
        dataframe["%-support_strength_confluence"] = (
            frames.support.sub(strongest_support, axis=0)
            .abs()
            .le(inputs.atr * 0.25, axis=0)
            .mul(source_weights, axis=1)
            .sum(axis=1)
        )
        dataframe["%-strongest_resistance_pressure"] = (
            resistance_strength * np.exp(-resistance_room)
        ).where(resistance_room <= 2.0, 0.0)
        dataframe["%-strongest_support_pressure"] = (
            support_strength * np.exp(-support_room)
        ).where(support_room <= 2.0, 0.0)

    def _inject_key_level_features(self, dataframe: DataFrame) -> DataFrame:
        """Add causal psychological, rolling and confirmed-pivot level features."""
        inputs = self._level_inputs(dataframe)
        support_candidates: list[pd.Series] = []
        resistance_candidates: list[pd.Series] = []
        self._add_psychological_levels(dataframe, inputs, support_candidates, resistance_candidates)
        self._add_rolling_levels(dataframe, inputs, support_candidates, resistance_candidates)
        self._add_previous_period_features(
            dataframe, inputs, support_candidates, resistance_candidates
        )
        self._add_confirmed_pivot_features(
            dataframe, inputs, support_candidates, resistance_candidates
        )
        frames = self._add_nearest_level_features(
            dataframe, inputs, support_candidates, resistance_candidates
        )
        self._add_strong_level_features(dataframe, inputs, frames)
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """Non-expanded temporal, candle-path and causal key-level features."""
        callback_metadata = kwargs.get("metadata")
        if isinstance(callback_metadata, dict):
            metadata = {**metadata, **callback_metadata}
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        dataframe["%-candle_body"] = (dataframe["close"] - dataframe["open"]) / dataframe["open"]
        dataframe["%-upper_wick"] = (
            dataframe[["open", "close"]].max(axis=1) - dataframe["high"]
        ).abs() / dataframe["close"]
        dataframe["%-lower_wick"] = (
            dataframe[["open", "close"]].min(axis=1) - dataframe["low"]
        ).abs() / dataframe["close"]
        dataframe["%-candle_range"] = self._safe_ratio(
            dataframe["high"] - dataframe["low"], dataframe["close"]
        )
        dataframe["%-close_to_open_gap"] = self._safe_ratio(
            dataframe["open"] - dataframe["close"].shift(1), dataframe["close"].shift(1)
        )
        dataframe["%-return_sign"] = (
            dataframe["close"]
            .pct_change()
            .apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
        )
        dataframe = self._inject_key_level_features(dataframe)
        return self._inject_detail_timeframe_features(dataframe, metadata)
