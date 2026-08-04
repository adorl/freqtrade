"""Prediction, attribution, and trade-execution mixin."""

import logging
import re
from collections import namedtuple
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.freqllm.attribution import AttributionRecord, SimpleAttributionWriter
from freqtrade.freqllm.decision import FreqAIPrediction, LLMView, SimpleDecision
from freqtrade.freqllm.observability import TradeRecord
from freqtrade.freqllm.strategy.contracts import StrategyCollaborators
from freqtrade.strategy import stoploss_from_open


logger = logging.getLogger(__name__)


_AttributionContext = namedtuple(
    "_AttributionContext", "dataframe row_position row freqai llm decision price"
)


class StrategyExecutionMixin:
    """Convert model output into Freqtrade entry, exit, and sizing callbacks."""

    strategy_collaborators: StrategyCollaborators

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Run FreqAI when enabled and sanitize non-finite indicator values."""
        if self.config.get("freqai", {}).get("enabled", False):
            dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe.replace([np.inf, -np.inf], np.nan)

    # ---------------------------------------------------------------------
    # Prediction extraction and decisions
    # ---------------------------------------------------------------------

    def _prediction_targets(self, row: pd.Series) -> tuple[dict[str, float], list[str]]:
        values: dict[str, float] = {}
        missing: list[str] = []
        for key, column in self.freqai_target_columns.items():
            if column not in row or pd.isna(row.get(column)):
                missing.append(column)
            else:
                values[key] = self._safe_float(row.get(column), 0.0)
        return values, missing

    def _normalized_probabilities(
        self, values: dict[str, float]
    ) -> tuple[float, float, float] | None:
        probabilities = (
            self._clip(values.get("long_probability", 0.0), 0.0, 1.0),
            self._clip(values.get("flat_probability", 0.0), 0.0, 1.0),
            self._clip(values.get("short_probability", 0.0), 0.0, 1.0),
        )
        total = sum(probabilities)
        if total <= 0:
            return None
        return tuple(value / total for value in probabilities)

    def _prediction_room_values(self, row: pd.Series) -> dict[str, float]:
        columns = {
            "upside_room_pct": ("%%-nearest_resistance_room_pct", float("inf")),
            "downside_room_pct": ("%%-nearest_support_room_pct", float("inf")),
            "upside_room_atr": ("%%-nearest_resistance_room_atr", float("inf")),
            "downside_room_atr": ("%%-nearest_support_room_atr", float("inf")),
            "strong_upside_room_atr": ("%%-strongest_resistance_room_atr", float("inf")),
            "strong_downside_room_atr": ("%%-strongest_support_room_atr", float("inf")),
            "strong_resistance_strength": ("%-strongest_resistance_strength", 0.0),
            "strong_support_strength": ("%-strongest_support_strength", 0.0),
        }
        return {
            name: self._safe_float(row.get(column), default)
            for name, (column, default) in columns.items()
        }

    def _extract_freqai_prediction(self, row: pd.Series, pair: str = "") -> FreqAIPrediction:
        do_predict = self._safe_float(row.get("do_predict"), 0.0)
        if do_predict != 1.0:
            return FreqAIPrediction(False, reason="do_predict_not_ready", do_predict=do_predict)
        values, missing = self._prediction_targets(row)
        if missing:
            return FreqAIPrediction(
                False, reason="missing_targets", do_predict=do_predict, raw={"missing": missing}
            )
        probabilities = self._normalized_probabilities(values)
        if probabilities is None:
            return FreqAIPrediction(
                False,
                reason="invalid_direction_probabilities",
                do_predict=do_predict,
            )
        long_probability, flat_probability, short_probability = probabilities
        room = self._prediction_room_values(row)
        return FreqAIPrediction(
            available=True,
            reason="ok",
            confidence=self._clip(max(long_probability, short_probability), 0.0, 1.0),
            dir_class=self._clip(long_probability - short_probability, -1.0, 1.0),
            long_probability=long_probability,
            flat_probability=flat_probability,
            short_probability=short_probability,
            long_edge=values["long_edge_q50"],
            short_edge=values["short_edge_q50"],
            long_edge_lower=values["long_edge_q20"],
            short_edge_lower=values["short_edge_q20"],
            long_edge_upper=values["long_edge_q80"],
            short_edge_upper=values["short_edge_q80"],
            long_peak_profit=values.get("long_peak_profit", 0.0),
            short_peak_profit=values.get("short_peak_profit", 0.0),
            long_pre_profit_drawdown=min(values.get("long_pre_profit_drawdown", 0.0), 0.0),
            short_pre_profit_drawdown=min(values.get("short_pre_profit_drawdown", 0.0), 0.0),
            long_post_profit_drawdown=max(values.get("long_post_profit_drawdown", 0.0), 0.0),
            short_post_profit_drawdown=max(values.get("short_post_profit_drawdown", 0.0), 0.0),
            long_early_fail_risk=self._clip(values.get("long_early_fail_risk", 0.0), 0.0, 1.0),
            short_early_fail_risk=self._clip(values.get("short_early_fail_risk", 0.0), 0.0, 1.0),
            level_event_class=self._clip(values.get("level_event_class", 0.0), -3.0, 3.0),
            **room,
            round_trip_cost=self._configured_round_trip_cost(
                pair, self.strategy_collaborators.runtime.feature_params
            ),
            do_predict=do_predict,
            di_value=self._safe_float(row.get("DI_values"), 0.0),
            raw={**values, **room},
        )

    def _neutral_llm_advice(self) -> dict[str, Any]:
        return {
            "action": "hold",
            "direction_bias": "neutral",
            "confidence": 0.0,
            "event_risk": 0.0,
            "avoid_trade": False,
            "leverage_cap_multiplier": 1.0,
            "reason": "neutral_bypass",
        }

    def _llm_view(self, pair: str) -> LLMView:
        if self.strategy_collaborators.runtime.is_backtest_mode:
            return LLMView.from_advice(self._neutral_llm_advice())
        return LLMView.from_advice(self.strategy_collaborators.runtime.llm_advice.get(pair))

    def _decision_from_row(
        self, row: pd.Series, pair: str
    ) -> tuple[FreqAIPrediction, LLMView, SimpleDecision]:
        freqai = self._extract_freqai_prediction(row, pair)
        llm = self._llm_view(pair)
        return freqai, llm, self.strategy_collaborators.decision_engine.decide_entry(freqai, llm)

    def _entry_tag(self, decision: SimpleDecision) -> str:
        side = decision.direction or "none"
        pre_dd_abs = abs(decision.pre_profit_drawdown)
        return (
            f"simple_{side}_e{decision.edge:.4f}_ps{decision.path_score:.4f}_"
            f"g{decision.path_score_gap:.4f}_sr{decision.stake_ratio:.3f}_"
            f"lv{decision.leverage:.2f}_pk{decision.peak_profit:.4f}_prd{pre_dd_abs:.4f}_"
            f"pod{decision.post_profit_drawdown:.4f}_llm{decision.llm_alignment}"
        )[:240]

    @staticmethod
    def _tag_float(entry_tag: str | None, key: str, default: float) -> float:
        if not entry_tag:
            return default
        match = re.search(rf"{re.escape(key)}(-?\d+(?:\.\d+)?)", entry_tag)
        if not match:
            return default
        try:
            return float(match.group(1))
        except ValueError:
            return default

    # ---------------------------------------------------------------------
    # Entries and attribution
    # ---------------------------------------------------------------------

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Emit directional entries and persist attribution for usable predictions."""
        pair = metadata.get("pair", "")
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""
        if dataframe.empty or not pair:
            return dataframe

        if self.strategy_collaborators.runtime.is_backtest_mode:
            self.strategy_collaborators.attribution_writer.reset_pair_once(pair)
            row_positions = range(len(dataframe))
        else:
            row_positions = [len(dataframe) - 1]

        for row_position in row_positions:
            row = dataframe.iloc[row_position]
            price = self._safe_float(row.get("close"), 0.0)
            if price <= 0 or self._safe_float(row.get("volume"), 0.0) <= 0:
                continue
            self._refresh_adaptive_params(pair)
            freqai, llm, decision = self._decision_from_row(row, pair)
            if not freqai.available and not self.strategy_collaborators.runtime.is_backtest_mode:
                continue

            if freqai.available:
                self._append_attribution(
                    pair, dataframe, row_position, row, freqai, llm, decision, price
                )

            if not decision.should_enter:
                continue
            tag = self._entry_tag(decision)
            if decision.direction == "long":
                dataframe.at[dataframe.index[row_position], "enter_long"] = 1
            else:
                dataframe.at[dataframe.index[row_position], "enter_short"] = 1
            dataframe.at[dataframe.index[row_position], "enter_tag"] = tag
        return dataframe

    def _append_attribution(self, pair: str, *attribution_context) -> None:
        attribution_context = _AttributionContext(*attribution_context)
        freqai = attribution_context.freqai
        llm = attribution_context.llm
        decision = attribution_context.decision
        long_path_score = self._safe_float(
            decision.details.get("long_path_score"), freqai.long_edge
        )
        short_path_score = self._safe_float(
            decision.details.get("short_path_score"), freqai.short_edge
        )
        side = decision.direction or ("long" if long_path_score >= short_path_score else "short")
        reference_time = self._as_utc(attribution_context.row.get("date"))
        feature_params = self.config.get("freqai", {}).get("feature_parameters", {})
        detail_tf = self._label_detail_timeframe(feature_params)
        detail_df = self._get_detail_dataframe(pair, detail_tf) if detail_tf else None
        realized = SimpleAttributionWriter.realized_metrics(
            attribution_context.dataframe,
            attribution_context.row_position,
            attribution_context.price,
            side,
            max_hold_candles=decision.max_hold_candles,
            detail_dataframe=detail_df,
            timeframe_minutes=timeframe_to_minutes(self.timeframe),
            detail_timeframe=detail_tf,
            detail_timeframe_minutes=timeframe_to_minutes(detail_tf) if detail_tf else 0,
        )
        payload = {
            "pair": pair,
            "reference_time": reference_time.isoformat() if reference_time else "",
            "side": side,
            "price": attribution_context.price,
            "decision": "emitted" if decision.should_enter else "rejected",
            "reason": decision.reason,
            "freqai_available": freqai.available,
            "freqai_reason": freqai.reason,
            "freqai_confidence": freqai.confidence,
            "freqai_dir_class": freqai.dir_class,
            "freqai_long_probability": freqai.long_probability,
            "freqai_flat_probability": freqai.flat_probability,
            "freqai_short_probability": freqai.short_probability,
            "freqai_long_edge": freqai.long_edge,
            "freqai_short_edge": freqai.short_edge,
            "freqai_long_edge_lower": freqai.long_edge_lower,
            "freqai_short_edge_lower": freqai.short_edge_lower,
            "freqai_long_edge_upper": freqai.long_edge_upper,
            "freqai_short_edge_upper": freqai.short_edge_upper,
            "freqai_edge_gap": abs(freqai.long_edge - freqai.short_edge),
            "freqai_level_event_class": freqai.level_event_class,
            "freqai_upside_room_pct": freqai.upside_room_pct,
            "freqai_downside_room_pct": freqai.downside_room_pct,
            "freqai_upside_room_atr": freqai.upside_room_atr,
            "freqai_downside_room_atr": freqai.downside_room_atr,
            "freqai_strong_upside_room_atr": freqai.strong_upside_room_atr,
            "freqai_strong_downside_room_atr": freqai.strong_downside_room_atr,
            "freqai_strong_resistance_strength": freqai.strong_resistance_strength,
            "freqai_strong_support_strength": freqai.strong_support_strength,
            "freqai_long_peak_profit": freqai.long_peak_profit,
            "freqai_short_peak_profit": freqai.short_peak_profit,
            "freqai_long_pre_profit_drawdown": freqai.long_pre_profit_drawdown,
            "freqai_short_pre_profit_drawdown": freqai.short_pre_profit_drawdown,
            "freqai_long_post_profit_drawdown": freqai.long_post_profit_drawdown,
            "freqai_short_post_profit_drawdown": freqai.short_post_profit_drawdown,
            "freqai_long_early_fail_risk": freqai.long_early_fail_risk,
            "freqai_short_early_fail_risk": freqai.short_early_fail_risk,
            "freqai_long_path_score": long_path_score,
            "freqai_short_path_score": short_path_score,
            "freqai_selected_path_score": decision.details.get(
                "selected_path_score", decision.path_score
            ),
            "freqai_opposite_path_score": decision.details.get(
                "opposite_path_score", decision.opposite_path_score
            ),
            "freqai_path_score_gap": decision.details.get(
                "path_score_gap", decision.path_score_gap
            ),
            "freqai_selected_peak_profit": decision.details.get(
                "selected_peak_profit", decision.peak_profit
            ),
            "freqai_selected_pre_profit_drawdown": decision.details.get(
                "selected_pre_profit_drawdown",
                decision.pre_profit_drawdown,
            ),
            "freqai_selected_post_profit_drawdown": decision.details.get(
                "selected_post_profit_drawdown",
                decision.post_profit_drawdown,
            ),
            "freqai_selected_reward_quality": decision.details.get("selected_reward_quality", ""),
            "freqai_selected_risk_quality": decision.details.get("selected_risk_quality", ""),
            "candidate_side": decision.details.get("candidate_side", side),
            "candidate_dir_class": decision.details.get("candidate_dir_class", freqai.dir_class),
            "candidate_edge": decision.details.get("candidate_edge", decision.edge),
            "candidate_opposite_edge": decision.details.get(
                "candidate_opposite_edge", decision.opposite_edge
            ),
            "candidate_edge_gap": decision.details.get("candidate_edge_gap", decision.edge_gap),
            "candidate_path_score": decision.details.get(
                "candidate_path_score", decision.path_score
            ),
            "candidate_opposite_path_score": decision.details.get(
                "candidate_opposite_path_score",
                decision.opposite_path_score,
            ),
            "candidate_path_score_gap": decision.details.get(
                "candidate_path_score_gap", decision.path_score_gap
            ),
            "candidate_peak_profit": decision.details.get(
                "candidate_peak_profit", decision.peak_profit
            ),
            "candidate_pre_profit_drawdown": decision.details.get(
                "candidate_pre_profit_drawdown",
                decision.pre_profit_drawdown,
            ),
            "candidate_post_profit_drawdown": decision.details.get(
                "candidate_post_profit_drawdown",
                decision.post_profit_drawdown,
            ),
            "candidate_early_fail_risk": decision.details.get("candidate_early_fail_risk", ""),
            "selected_early_fail_risk": decision.details.get("selected_early_fail_risk", ""),
            "early_fail_stake_multiplier": decision.details.get("early_fail_stake_multiplier", ""),
            "early_fail_leverage_multiplier": decision.details.get(
                "early_fail_leverage_multiplier", ""
            ),
            "candidate_reward_quality": decision.details.get("candidate_reward_quality", ""),
            "candidate_risk_quality": decision.details.get("candidate_risk_quality", ""),
            "dir_enter_threshold": decision.details.get("dir_enter_threshold", ""),
            "path_gap_threshold": decision.details.get("path_gap_threshold", ""),
            "dir_class_pass": decision.details.get("dir_class_pass", ""),
            "edge_min_pass": decision.details.get("edge_min_pass", ""),
            "edge_gap_pass": decision.details.get("edge_gap_pass", ""),
            "risk_adjusted_edge_pass": decision.details.get("risk_adjusted_edge_pass", ""),
            "strong_level_room_pass": decision.details.get("strong_level_room_pass", ""),
            "quality_multiplier": decision.details.get("quality_multiplier", ""),
            "leverage_reward_factor": decision.details.get("leverage_reward_factor", ""),
            "leverage_drawdown_factor": decision.details.get("leverage_drawdown_factor", ""),
            "llm_available": llm.available,
            "llm_direction_bias": llm.direction_bias,
            "llm_confidence": llm.confidence,
            "llm_event_risk": llm.event_risk,
            "llm_avoid_trade": llm.avoid_trade,
            "llm_alignment": decision.llm_alignment,
            "stake_ratio": decision.stake_ratio,
            "leverage": decision.leverage,
            "stop_loss_pct": decision.stop_loss_pct,
            "take_profit_pct": decision.take_profit_pct,
            "max_hold_candles": decision.max_hold_candles,
        }
        payload.update(realized)
        record = AttributionRecord(**payload)
        self.strategy_collaborators.attribution_writer.append(record)
        if self.strategy_collaborators.adaptive_manager.enabled:
            self.strategy_collaborators.adaptive_manager.record_signal(record)
            self._refresh_adaptive_params(pair)

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Disable indicator exits because deterministic callbacks own exit decisions."""
        pair = str(metadata.get("pair", "") or "")
        if pair:
            logger.debug("[%s] indicator-based exits are disabled", pair)
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    # ---------------------------------------------------------------------
    # Exits, stoploss, stake, leverage
    # ---------------------------------------------------------------------

    def _current_row(self, pair: str, current_time: datetime) -> pd.Series | None:
        if self.dp is None:
            return None
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            return None
        if dataframe is None or dataframe.empty:
            return None
        if "date" not in dataframe:
            return dataframe.iloc[-1]
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        current_ts = pd.Timestamp(current_time)
        if current_ts.tzinfo is None:
            current_ts = current_ts.tz_localize("UTC")
        else:
            current_ts = current_ts.tz_convert("UTC")
        eligible = dataframe.loc[dates <= current_ts]
        if eligible.empty:
            return None
        return eligible.iloc[-1]

    def _price_profit(self, trade, current_rate: float) -> float:
        open_rate = self._safe_float(getattr(trade, "open_rate", 0.0), 0.0)
        if open_rate <= 0 or current_rate <= 0:
            return 0.0
        if bool(getattr(trade, "is_short", False)):
            return open_rate / current_rate - 1.0
        return current_rate / open_rate - 1.0

    def _age_candles(self, trade, current_time: datetime) -> int:
        open_time = self._as_utc(
            getattr(trade, "open_date_utc", None) or getattr(trade, "open_date", None)
        )
        if open_time is None:
            return 0
        now = current_time if current_time.tzinfo is not None else current_time.replace(tzinfo=UTC)
        minutes = max((now - open_time).total_seconds() / 60.0, 0.0)
        tf_minutes = max(1, timeframe_to_minutes(self.timeframe))
        return int(minutes // tf_minutes)

    @staticmethod
    def _get_trade_custom_float(trade, key: str, default: float = 0.0) -> float:
        try:
            value = trade.get_custom_data(key, default=default)
            return float(value if value is not None else default)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            return default

    @staticmethod
    def _set_trade_custom_data(trade, key: str, value: float) -> None:
        try:
            trade.set_custom_data(key, float(value))
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            logger.debug("Unable to persist trade custom data %s: %s", key, exc)

    # Entry-snapshot keys persisted to trade custom_data on entry fill (P3-15).
    _ENTRY_SNAPSHOT_KEYS = (
        ("entry_edge", "e"),
        ("entry_path_score", "ps"),
        ("entry_peak_profit", "pk"),
        ("entry_pre_profit_drawdown", "prd"),
        ("entry_post_profit_drawdown", "pod"),
    )

    def _persist_entry_snapshot(self, trade, entry_tag: str | None) -> None:
        """Persist the entry decision snapshot to custom_data so exits no longer
        depend on re-parsing the entry_tag string each call."""
        if not entry_tag:
            return
        for custom_key, tag_key in self._ENTRY_SNAPSHOT_KEYS:
            existing = self._get_trade_custom_float(trade, custom_key, float("nan"))
            if np.isnan(existing):
                self._set_trade_custom_data(
                    trade, custom_key, self._tag_float(entry_tag, tag_key, 0.0)
                )

    def _entry_snapshot_float(
        self,
        trade,
        snapshot_keys: tuple[str, str],
        entry_tag: str | None,
        default: float = 0.0,
    ) -> float:
        """Read the canonical persisted custom-data entry snapshot."""
        del entry_tag
        custom_key, _tag_key = snapshot_keys
        return self._get_trade_custom_float(trade, custom_key, default)

    def _runtime_round_trip_cost(self, pair: str) -> float:
        feature_params = self.config.get("freqai", {}).get("feature_parameters", {})
        return self._configured_round_trip_cost(pair, feature_params)

    def _decision_stop_loss_pct(self) -> float:
        """Read the effective stop loss through the decision engine's public config."""
        return max(
            float(self.strategy_collaborators.decision_engine.config.risk.stop_loss_pct), 0.0005
        )

    def _update_trade_peak_profit(self, trade, price_profit: float) -> float:
        peak_key = "simple_trailing_peak_price_profit"
        peak_profit = max(
            self._get_trade_custom_float(trade, peak_key, price_profit),
            price_profit,
        )
        self._set_trade_custom_data(trade, peak_key, peak_profit)
        return peak_profit

    def _check_trailing_take_profit(
        self,
        pair: str,
        price_profit: float,
        current_profit: float,
        peak_profit: float,
    ) -> str | None:
        """Price-level trailing take-profit with net-profit/cost buffer protection."""
        exit_cfg = self.strategy_collaborators.simple_config.exit
        adaptive = (
            self.strategy_collaborators.adaptive_manager.current_parameters_dict()
            if self.strategy_collaborators.adaptive_manager.enabled
            else {}
        )
        trailing_activation = self._safe_float(
            adaptive.get("trailing_activation_pct"), exit_cfg.trailing_activation_pct
        )
        trailing_distance = self._safe_float(
            adaptive.get("trailing_distance_pct"), exit_cfg.trailing_distance_pct
        )
        if not getattr(exit_cfg, "trailing_enabled", False):
            return None
        if peak_profit < trailing_activation:
            return None
        round_trip_cost = self._runtime_round_trip_cost(pair)
        if current_profit <= 0 and price_profit <= round_trip_cost:
            return None
        if 0 < price_profit <= peak_profit - trailing_distance:
            return "simple_trailing_take_profit"
        return None

    def custom_exit(self, pair: str, trade, *args, **kwargs) -> str | None:
        """Return a deterministic callback exit reason for the current trade state."""
        values = self._callback_values(
            args,
            kwargs,
            ("current_time", "current_rate", "current_profit"),
            (datetime.now(UTC), 0.0, 0.0),
        )
        values["current_rate"] = self._safe_float(values["current_rate"], 0.0)
        values["current_profit"] = self._safe_float(values["current_profit"], 0.0)
        side = "short" if bool(getattr(trade, "is_short", False)) else "long"
        self._refresh_adaptive_params(pair)
        price_profit = self._price_profit(trade, values["current_rate"])
        peak_profit = self._update_trade_peak_profit(trade, price_profit)
        trailing_reason = self._check_trailing_take_profit(
            pair, price_profit, values["current_profit"], peak_profit
        )
        if trailing_reason:
            return trailing_reason

        age_candles = self._age_candles(trade, values["current_time"])
        prediction = None
        row = self._current_row(pair, values["current_time"])
        if row is not None:
            prediction = self._extract_freqai_prediction(row, pair)
        # P4-17: make the "no usable prediction" path explicit instead of silently
        # degrading to a price-only exit.
        if prediction is None or not prediction.available:
            reason = (
                "row_unavailable" if row is None else getattr(prediction, "reason", "unavailable")
            )
            logger.debug(
                "[%s] custom_exit without FreqAI prediction (%s); using price/path-only exits.",
                pair,
                reason,
            )
            prediction = None
        entry_tag = getattr(trade, "enter_tag", None)
        return self.strategy_collaborators.decision_engine.decide_exit(
            side=side,
            price_profit=price_profit,
            age_candles=age_candles,
            freqai=prediction,
            peak_profit=peak_profit,
            entry_peak_profit=self._entry_snapshot_float(
                trade, ("entry_peak_profit", "pk"), entry_tag
            ),
            entry_pre_profit_drawdown=self._entry_snapshot_float(
                trade, ("entry_pre_profit_drawdown", "prd"), entry_tag
            ),
            entry_post_profit_drawdown=self._entry_snapshot_float(
                trade, ("entry_post_profit_drawdown", "pod"), entry_tag
            ),
            entry_path_score=self._entry_snapshot_float(
                trade, ("entry_path_score", "ps"), entry_tag
            ),
            current_profit=values["current_profit"],
            round_trip_cost=self._runtime_round_trip_cost(pair),
        )

    def order_filled(self, pair: str, trade, order, current_time: datetime, **kwargs) -> None:
        """Persist entry snapshots and forward completed trades to adaptive analysis."""
        if kwargs:
            logger.debug("[%s] order-filled callback extras: %s", pair, sorted(kwargs))
        order_side = str(getattr(order, "ft_order_side", "") or "")
        entry_side = str(getattr(trade, "entry_side", "") or "")
        entry_tag = getattr(trade, "enter_tag", None)
        # On entry fill: persist the decision snapshot to custom_data (P3-15).
        if order_side == entry_side:
            self._persist_entry_snapshot(trade, entry_tag)
            return
        if not self.strategy_collaborators.adaptive_manager.enabled:
            return
        try:
            close_rate = self._safe_float(
                getattr(order, "safe_price", None), 0.0
            ) or self._safe_float(getattr(trade, "close_rate", None), 0.0)
            profit_ratio = self._safe_float(getattr(trade, "close_profit", None), 0.0)
            if close_rate > 0 and profit_ratio == 0.0:
                try:
                    profit_ratio = float(trade.calc_profit_ratio(close_rate))
                except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
                    profit_ratio = 0.0
            payload = {
                "trade_id": getattr(trade, "id", ""),
                "pair": pair,
                "side": "short" if bool(getattr(trade, "is_short", False)) else "long",
                "open_time": str(
                    getattr(trade, "open_date_utc", None) or getattr(trade, "open_date", "")
                ),
                "close_time": current_time.isoformat()
                if hasattr(current_time, "isoformat")
                else str(current_time),
                "exit_reason": getattr(trade, "exit_reason", "") or order_side,
                "profit_ratio": profit_ratio,
                "profit_abs": self._safe_float(getattr(trade, "close_profit_abs", None), 0.0),
                "leverage": self._safe_float(getattr(trade, "leverage", None), 1.0),
                "stake_amount": self._safe_float(getattr(trade, "stake_amount", None), 0.0),
                "entry_tag": entry_tag or "",
                "entry_edge": self._entry_snapshot_float(trade, ("entry_edge", "e"), entry_tag),
                "entry_path_score": self._entry_snapshot_float(
                    trade, ("entry_path_score", "ps"), entry_tag
                ),
                "entry_peak_profit": self._entry_snapshot_float(
                    trade, ("entry_peak_profit", "pk"), entry_tag
                ),
                "entry_pre_profit_drawdown": self._entry_snapshot_float(
                    trade, ("entry_pre_profit_drawdown", "prd"), entry_tag
                ),
                "entry_post_profit_drawdown": self._entry_snapshot_float(
                    trade, ("entry_post_profit_drawdown", "pod"), entry_tag
                ),
            }
            self.strategy_collaborators.adaptive_manager.record_trade(payload)
            performance = self.strategy_collaborators.performance_tracker
            if performance is not None:
                open_date = getattr(trade, "open_date_utc", None) or getattr(
                    trade, "open_date", None
                )
                if isinstance(open_date, datetime):
                    if open_date.tzinfo is None:
                        open_date = open_date.replace(tzinfo=UTC)
                    close_date = current_time
                    if close_date.tzinfo is None:
                        close_date = close_date.replace(tzinfo=UTC)
                    duration_hours = max(0.0, (close_date - open_date).total_seconds() / 3600)
                else:
                    duration_hours = 0.0
                performance.record_trade(
                    TradeRecord(
                        pair=pair,
                        profit_ratio=float(payload["profit_ratio"]),
                        duration_hours=duration_hours,
                        entry_reason=str(payload["entry_tag"]),
                        exit_reason=str(payload["exit_reason"]),
                    )
                )
            self._refresh_adaptive_params(pair)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            logger.debug("Adaptive trade recording failed: %s", exc)

    @staticmethod
    def _portfolio_open_trades() -> list:
        """Load open trades without constraining the strategy callback signature."""
        persistence = __import__("freqtrade.persistence", fromlist=["Trade"])
        return persistence.Trade.get_open_trades()

    def _same_direction_allowed(self, pair: str, side: str, open_trades: list) -> bool:
        if self.strategy_collaborators.execution.max_same_direction_positions <= 0:
            return True
        is_short = side.lower() == "short"
        same_direction = sum(
            bool(getattr(trade, "is_short", False)) == is_short for trade in open_trades
        )
        if same_direction < self.strategy_collaborators.execution.max_same_direction_positions:
            return True
        logger.info(
            "[%s] entry blocked: same-direction positions %s >= cap %s",
            pair,
            same_direction,
            self.strategy_collaborators.execution.max_same_direction_positions,
        )
        return False

    def _wallet_total_for_entry(self, pair: str) -> float | None:
        try:
            total = float(self.wallets.get_total_stake_amount()) if self.wallets else 0.0
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            logger.error("[%s] entry blocked: failed to read wallet exposure: %s", pair, exc)
            return None
        if total <= 0:
            logger.error("[%s] entry blocked: wallet exposure denominator is unavailable", pair)
            return None
        return total

    def _gross_exposure_allowed(
        self, pair: str, amount: float, rate: float, open_trades: list
    ) -> bool:
        if self.strategy_collaborators.execution.max_gross_exposure <= 0:
            return True
        total = self._wallet_total_for_entry(pair)
        if total is None:
            return False
        gross = sum(
            self._safe_float(getattr(trade, "stake_amount", 0.0), 0.0)
            * max(self._safe_float(getattr(trade, "leverage", 1.0), 1.0), 1.0)
            for trade in open_trades
        )
        new_notional = max(0.0, self._safe_float(amount, 0.0) * self._safe_float(rate, 0.0))
        exposure = (gross + new_notional) / total
        if exposure <= self.strategy_collaborators.execution.max_gross_exposure:
            return True
        logger.info(
            "[%s] entry blocked: gross exposure %.2f%% would exceed cap %.0f%%",
            pair,
            exposure * 100.0,
            self.strategy_collaborators.execution.max_gross_exposure * 100.0,
        )
        return False

    def confirm_trade_entry(self, pair: str, *args, **kwargs) -> bool:
        """Apply portfolio concentration and gross-exposure limits."""
        values = self._callback_values(
            args,
            kwargs,
            (
                "order_type",
                "amount",
                "rate",
                "time_in_force",
                "current_time",
                "entry_tag",
                "side",
            ),
            ("market", 0.0, 0.0, "gtc", datetime.now(UTC), None, "long"),
        )
        logger.debug(
            "[%s] validating %s/%s entry at %s with tag %s",
            pair,
            values["order_type"],
            values["time_in_force"],
            values["current_time"],
            values["entry_tag"],
        )
        if not self.strategy_collaborators.execution.portfolio_enabled:
            return True
        if (
            self.strategy_collaborators.execution.max_same_direction_positions <= 0
            and self.strategy_collaborators.execution.max_gross_exposure <= 0
        ):
            return True
        try:
            open_trades = self._portfolio_open_trades()
        except (AttributeError, ImportError, RuntimeError) as exc:
            logger.error(
                "[%s] entry blocked: unable to validate open-trade exposure: %s", pair, exc
            )
            return False
        side = str(values["side"])
        direction_allowed = self._same_direction_allowed(pair, side, open_trades)
        if not direction_allowed:
            return False
        return self._gross_exposure_allowed(
            pair,
            self._safe_float(values["amount"], 0.0),
            self._safe_float(values["rate"], 0.0),
            open_trades,
        )

    @staticmethod
    def _callback_values(
        positional: tuple, named: dict[str, Any], names: tuple[str, ...], defaults: tuple
    ) -> dict[str, Any]:
        """Normalize positional and keyword callback payloads without narrowing Freqtrade APIs."""
        return {
            name: positional[index] if index < len(positional) else named.get(name, defaults[index])
            for index, name in enumerate(names)
        }

    def custom_stoploss(self, pair: str, trade, *args, **kwargs) -> float:
        """Return the configured account-aware stop loss for a live trade."""
        values = self._callback_values(
            args,
            kwargs,
            ("current_time", "current_rate", "current_profit", "after_fill"),
            (datetime.now(UTC), 0.0, 0.0, False),
        )
        logger.debug(
            "[%s] stoploss at %s rate %.8f after_fill=%s",
            pair,
            values["current_time"],
            self._safe_float(values["current_rate"], 0.0),
            bool(values["after_fill"]),
        )
        self._refresh_adaptive_params(pair)
        leverage = max(
            1.0,
            self._safe_float(
                getattr(trade, "leverage", None),
                self.strategy_collaborators.simple_config.risk.leverage,
            ),
        )
        effective_stop = self._decision_stop_loss_pct()
        current_profit = self._safe_float(values["current_profit"], 0.0)
        return stoploss_from_open(
            -effective_stop * leverage,
            current_profit,
            is_short=bool(getattr(trade, "is_short", False)),
            leverage=leverage,
        )

    def custom_stake_amount(self, pair: str, *args, **kwargs) -> float:
        """Size an entry while respecting wallet balance and exchange stake limits."""
        values = self._callback_values(
            args,
            kwargs,
            (
                "current_time",
                "current_rate",
                "proposed_stake",
                "min_stake",
                "max_stake",
                "leverage",
                "entry_tag",
                "side",
            ),
            (datetime.now(UTC), 0.0, 0.0, None, 0.0, 1.0, None, "long"),
        )
        logger.debug(
            "[%s] sizing %s entry at %s rate %.8f with leverage %.3f",
            pair,
            values["side"],
            values["current_time"],
            self._safe_float(values["current_rate"], 0.0),
            self._safe_float(values["leverage"], 1.0),
        )
        self._refresh_adaptive_params(pair)
        stake_ratio = self._tag_float(
            values["entry_tag"], "sr", self.strategy_collaborators.simple_config.risk.stake_ratio
        )
        try:
            total = float(self.wallets.get_total_stake_amount()) if self.wallets else 0.0
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            logger.debug("[%s] Wallet balance unavailable for stake sizing: %s", pair, exc)
            total = 0.0
        proposed = self._safe_float(values["proposed_stake"], 0.0)
        stake = total * self._clip(stake_ratio, 0.0, 1.0) if total > 0 else proposed
        max_stake = self._safe_float(values["max_stake"], 0.0)
        min_stake = self._safe_float(values["min_stake"], 0.0)
        if max_stake > 0:
            stake = min(stake, max_stake)
        return 0.0 if min_stake > 0 and stake < min_stake else max(stake, 0.0)

    def leverage(self, pair: str, *args, **kwargs) -> float:
        """Select leverage from the entry snapshot and account-loss constraints."""
        values = self._callback_values(
            args,
            kwargs,
            (
                "current_time",
                "current_rate",
                "proposed_leverage",
                "max_leverage",
                "entry_tag",
                "side",
            ),
            (datetime.now(UTC), 0.0, 1.0, 1.0, None, "long"),
        )
        logger.debug(
            "[%s] selecting leverage for %s entry at %s rate %.8f",
            pair,
            values["side"],
            values["current_time"],
            self._safe_float(values["current_rate"], 0.0),
        )
        self._refresh_adaptive_params(pair)
        proposed = self._safe_float(values["proposed_leverage"], 1.0)
        selected = self._tag_float(values["entry_tag"], "lv", proposed)
        max_leverage = self._safe_float(values["max_leverage"], selected)
        cap = min(
            max_leverage or selected, self.strategy_collaborators.simple_config.risk.max_leverage
        )
        effective_stop = self._decision_stop_loss_pct()
        account_loss_cap = (
            self.strategy_collaborators.simple_config.risk.max_account_loss_per_trade
            / max(effective_stop, 1e-9)
        )
        return max(1.0, min(selected, cap, account_loss_cap))

    def adjust_trade_position(self, trade, *args, **kwargs) -> float | None:
        """Keep position adjustment disabled while accepting the complete callback payload."""
        values = self._callback_values(
            args,
            kwargs,
            (
                "current_time",
                "current_rate",
                "current_profit",
                "min_stake",
                "max_stake",
                "current_entry_rate",
                "current_exit_rate",
                "current_entry_profit",
                "current_exit_profit",
            ),
            (datetime.now(UTC), 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        logger.debug(
            "Position adjustment disabled for trade %s at %s: rate %.8f, profit %.6f, "
            "stake bounds %s..%.8f, entry %.8f/%.6f, exit %.8f/%.6f",
            getattr(trade, "id", ""),
            values["current_time"],
            self._safe_float(values["current_rate"], 0.0),
            self._safe_float(values["current_profit"], 0.0),
            values["min_stake"],
            self._safe_float(values["max_stake"], 0.0),
            self._safe_float(values["current_entry_rate"], 0.0),
            self._safe_float(values["current_entry_profit"], 0.0),
            self._safe_float(values["current_exit_rate"], 0.0),
            self._safe_float(values["current_exit_profit"], 0.0),
        )
