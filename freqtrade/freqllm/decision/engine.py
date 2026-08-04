"""Minimal FreqAI + LLM decision engine for short-horizon trading."""

import os
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any


def _env(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    if value.startswith("$"):
        return os.environ.get(value[1:], "")
    return value


def _float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


class _ValueBundle:
    _defaults: dict[str, Any] = {}

    def __init__(self, *args: Any, **values: Any) -> None:
        names = tuple(self._defaults)
        if len(args) > len(names):
            raise TypeError(f"expected at most {len(names)} arguments, got {len(args)}")
        duplicates = set(names[: len(args)]).intersection(values)
        unknown = set(values).difference(names)
        if duplicates:
            raise TypeError(f"got multiple values for argument '{min(duplicates)}'")
        if unknown:
            raise TypeError(f"got an unexpected keyword argument '{min(unknown)}'")
        supplied = dict(zip(names, args, strict=False))
        supplied.update(values)
        stored = {
            name: supplied.get(name, default() if callable(default) else default)
            for name, default in self._defaults.items()
        }
        object.__setattr__(self, "_values", stored)

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        values = object.__getattribute__(self, "_values")
        if name in values:
            values[name] = value
        else:
            object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        values = ", ".join(f"{name}={value!r}" for name, value in self._values.items())
        return f"{type(self).__name__}({values})"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self._values == getattr(other, "_values", None)

    def field_items(self):
        """Return the ordered fields stored by this value bundle."""
        return self._values.items()


def _bundle_defaults(names: str, values: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(names.split(), values, strict=True))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(init=False, repr=False, eq=False)
class SimpleSignalConfig(_ValueBundle):
    """Entry-signal, level, and optional LLM gate configuration."""

    _defaults = _bundle_defaults(
        "min_confidence min_edge min_edge_gap max_abs_prediction dir_enter_threshold "
        "dir_reverse_threshold require_edge_positive min_level_room_atr level_room_buffer_pct "
        "level_event_conflict_threshold strong_level_room_atr min_level_strength edge_q50_weight "
        "edge_q80_weight edge_q20_weight llm_event_risk_block llm_conflict_blocks "
        "llm_conflict_confidence require_llm neutral_size_multiplier unavailable_size_multiplier "
        "aligned_size_multiplier",
        (
            0.45,
            0.001,
            0.0005,
            0.10,
            0.20,
            0.25,
            True,
            0.35,
            0.0,
            0.35,
            0.35,
            0.0,
            0.70,
            0.20,
            0.10,
            0.80,
            True,
            0.60,
            False,
            1.0,
            0.70,
            1.0,
        ),
    )


@dataclass(init=False, repr=False, eq=False)
class SimpleRiskConfig(_ValueBundle):
    """Position sizing, leverage, and account-loss configuration."""

    _defaults = _bundle_defaults(
        "stake_ratio leverage max_leverage stop_loss_pct take_profit_pct "
        "max_account_loss_per_trade",
        (0.25, 10.0, 20.0, 0.010, 0.005, 0.03),
    )


@dataclass(init=False, repr=False, eq=False)
class SimpleExitConfig(_ValueBundle):
    """Time, signal, retention, and predicted-path exit configuration."""

    _defaults = _bundle_defaults(
        "max_hold_candles exit_on_reverse_signal exit_on_edge_decay reverse_edge_gap "
        "edge_decay_threshold trailing_enabled trailing_activation_pct trailing_distance_pct "
        "edge_decay_profit_retention edge_decay_min_profit_pct recovery_band_mult "
        "recovery_sl_floor_ratio peak_activation_ratio post_drawdown_band_mult "
        "path_collapse_ratio expiry_decay_ratio",
        (
            4,
            True,
            True,
            0.001,
            -0.0005,
            True,
            0.003,
            0.0015,
            0.75,
            0.001,
            1.20,
            0.25,
            0.70,
            1.20,
            0.20,
            0.50,
        ),
    )


@dataclass(init=False, repr=False, eq=False)
class SimpleAttributionConfig(_ValueBundle):
    """Attribution output switch and destination configuration."""

    _defaults = {"enabled": True, "directory": "user_data/backtest_results"}


def _config_values(
    defaults: _ValueBundle,
    raw: dict[str, Any],
    bool_fields: frozenset[str] = frozenset(),
    int_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, default in defaults.field_items():
        parser = _bool if name in bool_fields else _int if name in int_fields else _float
        values[name] = parser(raw.get(name), default)
    return values


@dataclass
class SimpleStrategyConfig:
    """Aggregate configuration consumed by the deterministic decision engine."""

    signal: SimpleSignalConfig = field(default_factory=SimpleSignalConfig)
    risk: SimpleRiskConfig = field(default_factory=SimpleRiskConfig)
    exit: SimpleExitConfig = field(default_factory=SimpleExitConfig)
    attribution: SimpleAttributionConfig = field(default_factory=SimpleAttributionConfig)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SimpleStrategyConfig":
        """Build the strategy config bundle from the freqtrade config dict."""
        raw = _mapping(config.get("llm_strategy", {})) if isinstance(config, dict) else {}
        if "exit" in raw:
            raise ValueError("Removed configuration field: llm_strategy.exit")
        signal_raw = _mapping(raw.get("signal"))
        risk_raw = _mapping(raw.get("risk"))
        exit_raw = _mapping(raw.get("simple_exit"))
        attr_raw = _mapping(raw.get("attribution"))
        signal = SimpleSignalConfig(
            **_config_values(
                SimpleSignalConfig(),
                signal_raw,
                frozenset({"require_edge_positive", "llm_conflict_blocks", "require_llm"}),
            )
        )
        risk = SimpleRiskConfig(**_config_values(SimpleRiskConfig(), risk_raw))
        exit_values = _config_values(
            SimpleExitConfig(),
            exit_raw,
            frozenset({"exit_on_reverse_signal", "exit_on_edge_decay", "trailing_enabled"}),
            frozenset({"max_hold_candles"}),
        )
        exit_values["max_hold_candles"] = max(1, exit_values["max_hold_candles"])
        attr_defaults = SimpleAttributionConfig()
        directory = _env(attr_raw.get("directory", attr_defaults.directory))
        attribution = SimpleAttributionConfig(
            enabled=_bool(attr_raw.get("enabled"), attr_defaults.enabled),
            directory=str(directory or attr_defaults.directory),
        )
        result = cls(signal, risk, SimpleExitConfig(**exit_values), attribution)
        result._validate()
        return result

    def _validate(self) -> None:
        signal = self.signal
        signal.min_confidence = _clip(signal.min_confidence, 0.0, 1.0)
        nonnegative = (
            "min_level_room_atr",
            "level_room_buffer_pct",
            "strong_level_room_atr",
            "min_level_strength",
        )
        for name in nonnegative:
            setattr(signal, name, max(getattr(signal, name), 0.0))
        signal.level_event_conflict_threshold = _clip(
            signal.level_event_conflict_threshold, 0.0, 3.0
        )
        bounded = (
            "edge_q50_weight",
            "edge_q80_weight",
            "edge_q20_weight",
            "llm_event_risk_block",
            "llm_conflict_confidence",
            "neutral_size_multiplier",
            "unavailable_size_multiplier",
        )
        for name in bounded:
            setattr(signal, name, _clip(getattr(signal, name), 0.0, 1.0))
        signal.aligned_size_multiplier = _clip(signal.aligned_size_multiplier, 0.0, 1.5)
        self.risk.stake_ratio = _clip(self.risk.stake_ratio, 0.0, 1.0)
        self.risk.leverage = max(self.risk.leverage, 1.0)
        self.risk.max_leverage = max(self.risk.max_leverage, 1.0)
        for name in ("stop_loss_pct", "take_profit_pct", "max_account_loss_per_trade"):
            setattr(self.risk, name, max(getattr(self.risk, name), 0.0005))
        self.exit.trailing_activation_pct = max(self.exit.trailing_activation_pct, 0.0005)
        self.exit.trailing_distance_pct = max(self.exit.trailing_distance_pct, 0.0005)


@dataclass(init=False, repr=False, eq=False)
class FreqAIPrediction(_ValueBundle):
    """Normalized FreqAI prediction and path-quality features for one candle."""

    _defaults = _bundle_defaults(
        "available reason confidence dir_class long_probability flat_probability short_probability "
        "long_edge short_edge long_edge_lower short_edge_lower long_edge_upper "
        "short_edge_upper long_peak_profit short_peak_profit long_pre_profit_drawdown "
        "short_pre_profit_drawdown long_post_profit_drawdown short_post_profit_drawdown "
        "long_early_fail_risk short_early_fail_risk level_event_class upside_room_pct "
        "downside_room_pct upside_room_atr downside_room_atr strong_upside_room_atr "
        "strong_downside_room_atr strong_resistance_strength strong_support_strength "
        "round_trip_cost do_predict di_value raw",
        (
            False,
            "ok",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            None,
            None,
            dict,
        ),
    )


@dataclass(init=False, repr=False, eq=False)
class LLMView(_ValueBundle):
    """Normalized, bounded view of optional LLM trading advice."""

    _defaults = _bundle_defaults(
        "available direction_bias confidence event_risk avoid_trade leverage_cap_multiplier "
        "reason raw",
        (False, "neutral", 0.0, 0.0, False, 1.0, "", dict),
    )

    @classmethod
    def from_advice(cls, advice: dict[str, Any] | None) -> "LLMView":
        """Create a view from raw LLM advice, tolerating missing or invalid fields."""
        if not isinstance(advice, dict):
            return cls(available=False, reason="llm_unavailable")
        direction = str(advice.get("direction_bias") or "neutral").lower()
        if direction not in {"long", "short", "neutral"}:
            direction = "neutral"
        return cls(
            available=_bool(advice.get("available"), True),
            direction_bias=direction,
            confidence=_clip(_float(advice.get("confidence"), 0.0), 0.0, 1.0),
            event_risk=_clip(_float(advice.get("event_risk"), 0.0), 0.0, 1.0),
            avoid_trade=_bool(advice.get("avoid_trade"), False),
            leverage_cap_multiplier=_clip(
                _float(advice.get("leverage_cap_multiplier"), 1.0), 0.0, 1.0
            ),
            reason=str(advice.get("reason") or ""),
            raw=dict(advice),
        )


@dataclass(init=False, repr=False, eq=False)
class SimpleDecision(_ValueBundle):
    """Deterministic entry decision, sizing, path metrics, and attribution details."""

    _defaults = _bundle_defaults(
        "action reason direction stake_ratio leverage stop_loss_pct take_profit_pct "
        "max_hold_candles size_multiplier edge opposite_edge edge_gap path_score "
        "opposite_path_score path_score_gap peak_profit pre_profit_drawdown "
        "post_profit_drawdown llm_alignment details",
        (
            "reject",
            "",
            None,
            0.0,
            1.0,
            0.0,
            0.0,
            1,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            "unavailable",
            dict,
        ),
    )

    @property
    def should_enter(self) -> bool:
        """Return whether this decision authorizes a directional entry."""
        return self.action == "enter" and self.direction in {"long", "short"}


_EntryContext = namedtuple(
    "_EntryContext", "direction selected opposite candidate_direction candidate candidate_opposite"
)
_LLMAdjustment = namedtuple("_LLMAdjustment", "rejection size_multiplier alignment cap_influence")
_Sizing = namedtuple(
    "_Sizing",
    "size_multiplier stake_ratio leverage quality_multiplier "
    "early_stake_multiplier early_leverage_multiplier",
)
_ExitPosition = namedtuple(
    "_ExitPosition", "side price_profit age_candles peak_profit current_profit round_trip_cost"
)
_EntrySnapshot = namedtuple(
    "_EntrySnapshot", "peak_profit pre_profit_drawdown post_profit_drawdown path_score"
)


@dataclass
class _ExitEvaluation:
    reason: str | None = None
    terminal: bool = False
    current_path_score: float | None = None


def _path_gap(selected: dict[str, float], opposite: dict[str, float]) -> float:
    return selected["path_score"] - opposite["path_score"]


def _profitable(position: _ExitPosition) -> bool:
    return position.current_profit > 0 or position.price_profit > max(position.round_trip_cost, 0.0)


class SimpleDecisionEngine:
    """Apply deterministic entry, sizing, risk, and exit rules to model views."""

    PEAK_TARGET_TP_MULTIPLIER = 0.33
    PATH_ENTRY_THRESHOLD_MULTIPLIER = 0.60

    def __init__(self, config: SimpleStrategyConfig, adaptive_params: dict[str, Any] | None = None):
        self.config = config
        self.adaptive_params = adaptive_params or {}

    def set_adaptive_params(self, adaptive_params: dict[str, Any] | None) -> None:
        """Update the adaptive parameter overlay used by the decision gates."""
        self.adaptive_params = adaptive_params or {}

    def _adaptive_float(self, key: str, default: float) -> float:
        value = self.adaptive_params.get(key, default)
        return _float(value, default)

    def _adaptive_int(self, key: str, default: int) -> int:
        value = self.adaptive_params.get(key, default)
        return _int(value, default)

    def _target_peak(self) -> float:
        multiplier = self._adaptive_float(
            "target_peak_tp_multiplier", self.PEAK_TARGET_TP_MULTIPLIER
        )
        return max(
            self.config.signal.min_edge * 4.0,
            self.config.risk.take_profit_pct * multiplier,
            1e-6,
        )

    def _path_entry_threshold(self) -> float:
        multiplier = self._adaptive_float(
            "path_entry_threshold_multiplier", self.PATH_ENTRY_THRESHOLD_MULTIPLIER
        )
        return max(self.config.signal.min_edge * multiplier, 1e-6)

    def _path_gap_threshold(self) -> float:
        return max(
            self.config.signal.min_edge_gap
            * self._adaptive_float("path_gap_threshold_multiplier", 1.0),
            1e-9,
        )

    def _effective_stop_loss_pct(self) -> float:
        return max(self.config.risk.stop_loss_pct, 0.0005)

    def _effective_take_profit_pct(self) -> float:
        return max(
            self.config.risk.take_profit_pct * self._adaptive_float("take_profit_multiplier", 1.0),
            0.0005,
        )

    def _effective_max_hold(self) -> int:
        config_max = max(1, int(self.config.exit.max_hold_candles))
        return max(1, min(self._adaptive_int("max_hold_candles", config_max), config_max))

    def _side_components(self, prediction: FreqAIPrediction, side: str) -> dict[str, float]:
        prefix = "short" if side == "short" else "long"
        edge = getattr(prediction, f"{prefix}_edge")
        edge_lower = getattr(prediction, f"{prefix}_edge_lower")
        edge_upper = getattr(prediction, f"{prefix}_edge_upper")
        pre_drawdown = getattr(prediction, f"{prefix}_pre_profit_drawdown")
        probability = getattr(prediction, f"{prefix}_probability")
        signal = self.config.signal
        quantile_score = (
            signal.edge_q50_weight * edge
            + signal.edge_q80_weight * edge_upper
            + signal.edge_q20_weight * edge_lower
        )
        return {
            "edge": edge,
            "edge_lower": edge_lower,
            "edge_upper": edge_upper,
            "quantile_score": quantile_score,
            "risk_adjusted_edge": probability * quantile_score,
            "direction_probability": probability,
            "peak_profit": max(getattr(prediction, f"{prefix}_peak_profit"), 0.0),
            "pre_profit_drawdown": pre_drawdown,
            "pre_drawdown_abs": abs(min(pre_drawdown, 0.0)),
            "post_profit_drawdown": max(getattr(prediction, f"{prefix}_post_profit_drawdown"), 0.0),
            "early_fail_risk": _clip(getattr(prediction, f"{prefix}_early_fail_risk"), 0.0, 1.0),
        }

    def _path_score(self, components: dict[str, float]) -> float:
        stop_loss = self._effective_stop_loss_pct()
        target_peak = self._target_peak()
        reward_quality = _clip(
            components["peak_profit"] / target_peak,
            self._adaptive_float("reward_quality_floor", 0.50),
            self._adaptive_float("reward_quality_cap", 1.20),
        )
        pre_free = stop_loss * self._adaptive_float("pre_drawdown_free_ratio", 0.0)
        effective_pre_drawdown = max(0.0, components["pre_drawdown_abs"] - pre_free)
        risk_quality = _clip(
            1.0
            - self._adaptive_float("pre_drawdown_weight", 1.0)
            * effective_pre_drawdown
            / max(stop_loss, 1e-9)
            - self._adaptive_float("post_drawdown_weight", 0.5)
            * components["post_profit_drawdown"]
            / max(stop_loss, 1e-9),
            self._adaptive_float("risk_quality_floor", 0.25),
            self._adaptive_float("risk_quality_cap", 1.00),
        )
        components.update(
            effective_pre_drawdown=effective_pre_drawdown,
            reward_quality=reward_quality,
            risk_quality=risk_quality,
            target_peak=target_peak,
        )
        return components["risk_adjusted_edge"] * reward_quality * risk_quality

    @staticmethod
    def _prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
        return {f"{prefix}_{key}": value for key, value in values.items()}

    @staticmethod
    def _base_details(prediction: FreqAIPrediction, llm: LLMView) -> dict[str, Any]:
        prediction_names = (
            "reason",
            "confidence",
            "dir_class",
            "long_probability",
            "flat_probability",
            "short_probability",
            "long_edge",
            "short_edge",
            "long_edge_lower",
            "short_edge_lower",
            "long_edge_upper",
            "short_edge_upper",
            "long_peak_profit",
            "short_peak_profit",
            "long_pre_profit_drawdown",
            "short_pre_profit_drawdown",
            "long_post_profit_drawdown",
            "short_post_profit_drawdown",
            "long_early_fail_risk",
            "short_early_fail_risk",
        )
        details = {f"freqai_{name}": getattr(prediction, name) for name in prediction_names}
        details.update(
            llm_available=llm.available,
            llm_direction_bias=llm.direction_bias,
            llm_confidence=llm.confidence,
            llm_event_risk=llm.event_risk,
            llm_avoid_trade=llm.avoid_trade,
        )
        return details

    def _prediction_rejection(
        self, prediction: FreqAIPrediction, details: dict[str, Any]
    ) -> SimpleDecision | None:
        if not prediction.available:
            return SimpleDecision("reject", f"freqai_{prediction.reason}", details=details)
        minimum = _clip(
            self.config.signal.min_confidence + self._adaptive_float("min_confidence_delta", 0.0),
            0.0,
            1.0,
        )
        if prediction.confidence < minimum:
            details["effective_min_confidence"] = minimum
            return SimpleDecision("reject", "freqai_confidence_low", details=details)
        bounded_names = (
            "long_edge",
            "short_edge",
            "long_edge_lower",
            "short_edge_lower",
            "long_edge_upper",
            "short_edge_upper",
            "long_peak_profit",
            "short_peak_profit",
            "long_pre_profit_drawdown",
            "short_pre_profit_drawdown",
            "long_post_profit_drawdown",
            "short_post_profit_drawdown",
        )
        if any(
            abs(getattr(prediction, name)) > self.config.signal.max_abs_prediction
            for name in bounded_names
        ):
            return SimpleDecision("reject", "freqai_prediction_outlier", details=details)
        return None

    def _entry_context(
        self, prediction: FreqAIPrediction, details: dict[str, Any]
    ) -> _EntryContext:
        long_path = self._side_components(prediction, "long")
        short_path = self._side_components(prediction, "short")
        long_path["path_score"] = self._path_score(long_path)
        short_path["path_score"] = self._path_score(short_path)
        details.update(self._prefixed("long", long_path))
        details.update(self._prefixed("short", short_path))
        threshold = max(
            self._adaptive_float("dir_enter_threshold", self.config.signal.dir_enter_threshold),
            1e-6,
        )
        if prediction.dir_class > 0 or (
            prediction.dir_class == 0 and long_path["path_score"] >= short_path["path_score"]
        ):
            candidate_direction, candidate, candidate_opposite = "long", long_path, short_path
        else:
            candidate_direction, candidate, candidate_opposite = "short", short_path, long_path
        direction = None
        selected, opposite = long_path, short_path
        if prediction.dir_class >= threshold:
            direction = "long"
        elif prediction.dir_class <= -threshold:
            direction, selected, opposite = "short", short_path, long_path
        context = _EntryContext(
            direction, selected, opposite, candidate_direction, candidate, candidate_opposite
        )
        self._update_candidate_details(details, prediction, context, threshold)
        return context

    def _update_candidate_details(
        self,
        details: dict[str, Any],
        prediction: FreqAIPrediction,
        context: _EntryContext,
        threshold: float,
    ) -> None:
        candidate, opposite = context.candidate, context.candidate_opposite
        fields = (
            "edge",
            "path_score",
            "peak_profit",
            "pre_profit_drawdown",
            "pre_drawdown_abs",
            "post_profit_drawdown",
            "early_fail_risk",
            "reward_quality",
            "risk_quality",
        )
        details.update({f"candidate_{name}": candidate[name] for name in fields})
        details.update(
            candidate_side=context.candidate_direction,
            candidate_dir_class=prediction.dir_class,
            candidate_opposite_edge=opposite["edge"],
            candidate_edge_gap=candidate["edge"] - opposite["edge"],
            candidate_opposite_path_score=opposite["path_score"],
            candidate_path_score_gap=_path_gap(candidate, opposite),
            dir_enter_threshold=threshold,
            path_gap_threshold=self._path_gap_threshold(),
            dir_class_pass=abs(prediction.dir_class) >= threshold,
            edge_positive_pass=candidate["edge"] > 0.0,
        )

    def _level_gate_values(
        self, prediction: FreqAIPrediction, direction: str
    ) -> tuple[float, float, float, float, bool]:
        suffix = "upside" if direction == "long" else "downside"
        room_pct = getattr(prediction, f"{suffix}_room_pct")
        room_atr = getattr(prediction, f"{suffix}_room_atr")
        strong_room = getattr(prediction, f"strong_{suffix}_room_atr")
        strength_name = (
            "strong_resistance_strength" if direction == "long" else "strong_support_strength"
        )
        event_sign = -1 if direction == "long" else 1
        event_conflict = (
            event_sign * prediction.level_event_class
            >= self.config.signal.level_event_conflict_threshold
        )
        return room_pct, room_atr, strong_room, getattr(prediction, strength_name), event_conflict

    def _entry_gate_rejection(
        self,
        prediction: FreqAIPrediction,
        context: _EntryContext,
        details: dict[str, Any],
    ) -> str | None:
        if context.direction is None:
            return "no_directional_signal"
        selected, signal = context.selected, self.config.signal
        room_pct, room_atr, strong_room, level_strength, event_conflict = self._level_gate_values(
            prediction, context.direction
        )
        required_room = prediction.round_trip_cost + signal.min_edge + signal.level_room_buffer_pct
        strong_conflict = level_strength >= signal.min_level_strength
        strong_conflict &= strong_room < signal.strong_level_room_atr
        edge_gap = selected["edge"] - context.opposite["edge"]
        details.update(
            selected_room_pct=room_pct,
            selected_room_atr=room_atr,
            selected_strong_room_atr=strong_room,
            selected_strong_level_strength=level_strength,
            required_room_pct=required_room,
            level_event_class=prediction.level_event_class,
            edge_min_pass=selected["edge"] >= signal.min_edge,
            edge_gap_pass=edge_gap >= self._path_gap_threshold(),
            risk_adjusted_edge_pass=selected["risk_adjusted_edge"] > 0.0,
            strong_level_room_pass=not strong_conflict,
        )
        checks = (
            (signal.require_edge_positive and selected["edge"] <= 0.0, "edge_not_positive"),
            (selected["edge"] < signal.min_edge, "edge_below_minimum"),
            (edge_gap < self._path_gap_threshold(), "edge_gap_below_minimum"),
            (selected["risk_adjusted_edge"] <= 0.0, "risk_adjusted_edge_not_positive"),
            (room_pct < required_room, "key_level_room_insufficient"),
            (room_atr < signal.min_level_room_atr, "key_level_atr_room_insufficient"),
            (strong_conflict, "strong_key_level_room_insufficient"),
            (event_conflict, "key_level_event_conflict"),
        )
        return next((reason for failed, reason in checks if failed), None)

    @staticmethod
    def _path_rejection(
        reason: str, context: _EntryContext, details: dict[str, Any], *, selected: bool = False
    ) -> SimpleDecision:
        path = context.selected if selected else context.candidate
        opposite = context.opposite if selected else context.candidate_opposite
        direction = (
            context.direction if selected else context.direction or context.candidate_direction
        )
        return SimpleDecision(
            "reject",
            reason,
            direction=direction,
            edge=path["edge"],
            opposite_edge=opposite["edge"],
            edge_gap=path["edge"] - opposite["edge"],
            path_score=path["path_score"],
            opposite_path_score=opposite["path_score"],
            path_score_gap=path["path_score"] - opposite["path_score"],
            peak_profit=path["peak_profit"],
            pre_profit_drawdown=path["pre_profit_drawdown"],
            post_profit_drawdown=path["post_profit_drawdown"],
            details=details,
        )

    def _llm_adjustment(
        self, direction: str, llm: LLMView, details: dict[str, Any]
    ) -> _LLMAdjustment:
        signal = self.config.signal
        cap_influence = _clip(self._adaptive_float("llm_leverage_cap_influence", 1.0), 0.0, 1.0)
        if not llm.available:
            rejection = None
            if signal.require_llm:
                rejection = SimpleDecision(
                    "reject", "llm_required_unavailable", direction=direction, details=details
                )
            return _LLMAdjustment(
                rejection, signal.unavailable_size_multiplier, "unavailable", cap_influence
            )
        if llm.avoid_trade:
            rejection = SimpleDecision(
                "reject", "llm_avoid_trade", direction=direction, details=details
            )
            return _LLMAdjustment(rejection, 1.0, "unavailable", cap_influence)
        event_block = _clip(
            signal.llm_event_risk_block * self._adaptive_float("llm_event_risk_block_mult", 1.0),
            0.0,
            1.0,
        )
        if llm.event_risk >= event_block:
            rejection = SimpleDecision(
                "reject", "llm_event_risk_block", direction=direction, details=details
            )
            return _LLMAdjustment(rejection, 1.0, "unavailable", cap_influence)
        neutral_multiplier = signal.neutral_size_multiplier * self._adaptive_float(
            "llm_neutral_size_mult", 1.0
        )
        if llm.direction_bias == direction:
            return _LLMAdjustment(None, signal.aligned_size_multiplier, "aligned", cap_influence)
        if llm.direction_bias not in {"long", "short"}:
            return _LLMAdjustment(None, neutral_multiplier, "neutral", cap_influence)
        conflict_threshold = _clip(
            signal.llm_conflict_confidence
            * self._adaptive_float("llm_conflict_confidence_mult", 1.0),
            0.0,
            1.0,
        )
        rejection = None
        if signal.llm_conflict_blocks and llm.confidence >= conflict_threshold:
            rejection = SimpleDecision(
                "reject", "llm_direction_conflict", direction=direction, details=details
            )
        return _LLMAdjustment(rejection, neutral_multiplier, "conflict", cap_influence)

    def _position_sizing(
        self,
        selected: dict[str, float],
        llm: LLMView,
        adjustment: _LLMAdjustment,
    ) -> _Sizing:
        early_risk = _clip(selected.get("early_fail_risk", 0.0), 0.0, 1.0)
        quality = _clip(
            selected["path_score"] / self._target_peak(),
            self._adaptive_float("stake_quality_min_multiplier", 0.35),
            self._adaptive_float("stake_quality_max_multiplier", 1.20),
        )
        early_stake = _clip(
            1.0 - self._adaptive_float("early_fail_stake_weight", 0.70) * early_risk,
            self._adaptive_float("early_fail_stake_floor", 0.50),
            1.0,
        )
        size_multiplier = adjustment.size_multiplier * quality * early_stake
        reward = min(selected["reward_quality"], self._adaptive_float("leverage_reward_cap", 1.20))
        early_leverage = _clip(
            1.0 - self._adaptive_float("early_fail_leverage_weight", 0.35) * early_risk,
            self._adaptive_float("early_fail_leverage_floor", 0.70),
            1.0,
        )
        risk = self.config.risk
        leverage = min(risk.leverage, risk.max_leverage) * reward * selected["risk_quality"]
        leverage *= early_leverage
        if llm.available:
            effective_cap = 1.0 - adjustment.cap_influence * (1.0 - llm.leverage_cap_multiplier)
            leverage *= _clip(effective_cap, 0.0, 1.0)
        leverage = _clip(leverage, 1.0, risk.max_leverage)
        account_cap = risk.max_account_loss_per_trade / max(self._effective_stop_loss_pct(), 1e-9)
        leverage = max(1.0, min(leverage, account_cap))
        stake_ratio = _clip(risk.stake_ratio * size_multiplier, 0.0, 1.0)
        return _Sizing(size_multiplier, stake_ratio, leverage, quality, early_stake, early_leverage)

    @staticmethod
    def _update_entry_details(
        details: dict[str, Any], context: _EntryContext, sizing: _Sizing
    ) -> None:
        selected, opposite = context.selected, context.opposite
        fields = (
            "edge",
            "path_score",
            "peak_profit",
            "pre_profit_drawdown",
            "pre_drawdown_abs",
            "post_profit_drawdown",
            "early_fail_risk",
            "reward_quality",
            "risk_quality",
        )
        details.update({f"selected_{name}": selected[name] for name in fields})
        details.update(
            opposite_edge=opposite["edge"],
            edge_gap=selected["edge"] - opposite["edge"],
            opposite_path_score=opposite["path_score"],
            path_score_gap=_path_gap(selected, opposite),
            dir_class_pass=True,
            edge_positive_pass=selected["edge"] > 0.0,
            quality_multiplier=sizing.quality_multiplier,
            early_fail_stake_multiplier=sizing.early_stake_multiplier,
            early_fail_leverage_multiplier=sizing.early_leverage_multiplier,
            leverage_reward_factor=min(selected["reward_quality"], 1.20),
            leverage_drawdown_factor=selected["risk_quality"],
        )

    def decide_entry(self, freqai: FreqAIPrediction, llm: LLMView) -> SimpleDecision:
        """Evaluate entry gates and return a rejection or fully sized entry decision."""
        details = self._base_details(freqai, llm)
        rejection = self._prediction_rejection(freqai, details)
        if rejection is not None:
            return rejection
        context = self._entry_context(freqai, details)
        reason = self._entry_gate_rejection(freqai, context, details)
        if reason is not None:
            return self._path_rejection(reason, context, details)
        early_risk = _clip(context.selected.get("early_fail_risk", 0.0), 0.0, 1.0)
        early_limit = self._adaptive_float("early_fail_block_threshold", 0.90)
        if early_risk >= early_limit:
            details["early_fail_block_threshold"] = early_limit
            return self._path_rejection("early_fail_risk_high", context, details, selected=True)
        direction = context.direction or context.candidate_direction
        adjustment = self._llm_adjustment(direction, llm, details)
        if adjustment.rejection is not None:
            return adjustment.rejection
        sizing = self._position_sizing(context.selected, llm, adjustment)
        self._update_entry_details(details, context, sizing)
        selected, opposite = context.selected, context.opposite
        return SimpleDecision(
            action="enter",
            reason="simple_dir_class_entry",
            direction=direction,
            stake_ratio=sizing.stake_ratio,
            leverage=sizing.leverage,
            stop_loss_pct=self._effective_stop_loss_pct(),
            take_profit_pct=self._effective_take_profit_pct(),
            max_hold_candles=self._effective_max_hold(),
            size_multiplier=sizing.size_multiplier,
            edge=selected["edge"],
            opposite_edge=opposite["edge"],
            edge_gap=selected["edge"] - opposite["edge"],
            path_score=selected["path_score"],
            opposite_path_score=opposite["path_score"],
            path_score_gap=_path_gap(context.selected, context.opposite),
            peak_profit=selected["peak_profit"],
            pre_profit_drawdown=selected["pre_profit_drawdown"],
            post_profit_drawdown=selected["post_profit_drawdown"],
            llm_alignment=adjustment.alignment,
            details=details,
        )

    @staticmethod
    def _exit_position(side: str, price_profit: float, context: dict[str, Any]) -> _ExitPosition:
        floats = ("peak_profit", "current_profit", "round_trip_cost")
        values = (_float(context.get(name), 0.0) for name in floats)
        return _ExitPosition(side, price_profit, _int(context.get("age_candles"), 0), *values)

    @staticmethod
    def _entry_snapshot(context: dict[str, Any]) -> _EntrySnapshot:
        names = (
            "entry_peak_profit",
            "entry_pre_profit_drawdown",
            "entry_post_profit_drawdown",
            "entry_path_score",
        )
        return _EntrySnapshot(*(_float(context.get(name), 0.0) for name in names))

    def _initial_exit(self, position: _ExitPosition, entry: _EntrySnapshot) -> _ExitEvaluation:
        exit_cfg = self.config.exit
        take_profit = position.price_profit >= self._effective_take_profit_pct()
        if not exit_cfg.trailing_enabled and take_profit:
            return _ExitEvaluation("simple_take_profit", True)
        stop_loss = self._effective_stop_loss_pct()
        if position.price_profit <= -stop_loss:
            return _ExitEvaluation("simple_stop_loss", True)
        pre_drawdown = abs(entry.pre_profit_drawdown)
        recovery_band = self._adaptive_float("recovery_band_mult", exit_cfg.recovery_band_mult)
        recovery_floor = self._adaptive_float(
            "recovery_sl_floor_ratio", exit_cfg.recovery_sl_floor_ratio
        )
        allowed = max(pre_drawdown * recovery_band, stop_loss * recovery_floor)
        recovery_failed = position.age_candles >= 1 and pre_drawdown > 0
        if recovery_failed and position.price_profit <= -allowed:
            return _ExitEvaluation("simple_path_recovery_failed", True)
        max_hold = self._effective_max_hold()
        no_progress_age = self._adaptive_int("no_progress_age", max(3, min(max_hold, 6)))
        minimum_progress = max(position.round_trip_cost, 0.0)
        minimum_progress += exit_cfg.edge_decay_min_profit_pct * self._adaptive_float(
            "no_progress_min_profit_multiplier", 1.0
        )
        no_progress = position.age_candles >= no_progress_age and position.price_profit <= 0
        if no_progress and position.peak_profit < minimum_progress:
            return _ExitEvaluation("simple_no_progress_exit", True)
        return _ExitEvaluation()

    @staticmethod
    def _profit_guarded(reason: str, position: _ExitPosition) -> _ExitEvaluation:
        return _ExitEvaluation(reason if _profitable(position) else None, True)

    def _predicted_path_exit(
        self, position: _ExitPosition, entry: _EntrySnapshot
    ) -> _ExitEvaluation:
        exit_cfg = self.config.exit
        retention = self._adaptive_float(
            "edge_decay_profit_retention", exit_cfg.edge_decay_profit_retention
        )
        peak_ratio = self._adaptive_float("peak_activation_ratio", exit_cfg.peak_activation_ratio)
        activation = max(exit_cfg.edge_decay_min_profit_pct, entry.peak_profit * peak_ratio)
        retained = position.peak_profit * retention
        peak_retraced = entry.peak_profit > 0 and position.peak_profit >= activation
        if peak_retraced and 0 < position.price_profit <= retained:
            return self._profit_guarded("simple_predicted_peak_retention", position)
        post_band = self._adaptive_float(
            "post_drawdown_band_mult", exit_cfg.post_drawdown_band_mult
        )
        distance = self._adaptive_float("trailing_distance_pct", exit_cfg.trailing_distance_pct)
        actual_drawdown = max(0.0, position.peak_profit - position.price_profit)
        maximum = max(entry.post_profit_drawdown * post_band, distance * post_band)
        post_retraced = entry.post_profit_drawdown > 0 and position.peak_profit > 0
        if post_retraced and position.price_profit > 0 and actual_drawdown >= maximum:
            return self._profit_guarded("simple_path_post_drawdown_broken", position)
        return _ExitEvaluation()

    def _reverse_exit(
        self,
        position: _ExitPosition,
        prediction: FreqAIPrediction,
        current: dict[str, float],
        opposite: dict[str, float],
    ) -> str | None:
        if not self.config.exit.exit_on_reverse_signal:
            return None
        threshold = max(
            self._adaptive_float("dir_reverse_threshold", self.config.signal.dir_reverse_threshold),
            1e-6,
        )
        direction_flipped = (position.side == "long" and prediction.dir_class <= -threshold) or (
            position.side == "short" and prediction.dir_class >= threshold
        )
        reverse_gap = self._adaptive_float("reverse_edge_gap", self.config.exit.reverse_edge_gap)
        path_flipped = opposite["path_score"] > self._path_entry_threshold()
        path_flipped &= opposite["path_score"] - current["path_score"] >= reverse_gap
        return "simple_reverse_signal" if direction_flipped or path_flipped else None

    def _retention_exit(
        self, position: _ExitPosition, path_score: float, loss_reason: str, profit_reason: str
    ) -> _ExitEvaluation:
        retention = self._adaptive_float(
            "edge_decay_profit_retention", self.config.exit.edge_decay_profit_retention
        )
        if position.price_profit > position.peak_profit * retention:
            return _ExitEvaluation(terminal=True, current_path_score=path_score)
        if position.price_profit <= 0:
            return _ExitEvaluation(loss_reason, True, path_score)
        result = self._profit_guarded(profit_reason, position)
        result.current_path_score = path_score
        return result

    def _collapse_exit(
        self, position: _ExitPosition, entry: _EntrySnapshot, current_path_score: float
    ) -> _ExitEvaluation:
        exit_cfg = self.config.exit
        ratio = self._adaptive_float("path_collapse_ratio", exit_cfg.path_collapse_ratio)
        collapsed = entry.path_score > 0 and current_path_score <= entry.path_score * ratio
        collapsed &= position.peak_profit >= exit_cfg.edge_decay_min_profit_pct
        if not collapsed:
            return _ExitEvaluation(current_path_score=current_path_score)
        return self._retention_exit(
            position,
            current_path_score,
            "simple_path_score_collapse",
            "simple_path_score_collapse_retention",
        )

    def _decay_exit(self, position: _ExitPosition, current_path_score: float) -> _ExitEvaluation:
        exit_cfg = self.config.exit
        decayed = exit_cfg.exit_on_edge_decay
        decayed &= current_path_score <= exit_cfg.edge_decay_threshold
        if not decayed:
            return _ExitEvaluation(current_path_score=current_path_score)
        if position.peak_profit < exit_cfg.edge_decay_min_profit_pct:
            return _ExitEvaluation("simple_path_decay", True, current_path_score)
        return self._retention_exit(
            position,
            current_path_score,
            "simple_path_decay",
            "simple_path_decay_profit_retention",
        )

    def _prediction_exit(
        self,
        position: _ExitPosition,
        entry: _EntrySnapshot,
        prediction: FreqAIPrediction | None,
    ) -> _ExitEvaluation:
        if prediction is None or not prediction.available:
            return _ExitEvaluation()
        current = self._side_components(prediction, position.side)
        early_limit = self._adaptive_float("early_fail_exit_threshold", 0.80)
        early_loss = max(position.round_trip_cost * 0.5, self.config.exit.edge_decay_min_profit_pct)
        early_failure = current.get("early_fail_risk", 0.0) >= early_limit
        if early_failure and position.price_profit <= -early_loss:
            return _ExitEvaluation("simple_early_fail_risk", True)
        opposite_side = "long" if position.side == "short" else "short"
        opposite = self._side_components(prediction, opposite_side)
        current["path_score"] = self._path_score(current)
        opposite["path_score"] = self._path_score(opposite)
        reverse = self._reverse_exit(position, prediction, current, opposite)
        if reverse is not None:
            return _ExitEvaluation(reverse, True, current["path_score"])
        collapse = self._collapse_exit(position, entry, current["path_score"])
        return collapse if collapse.terminal else self._decay_exit(position, current["path_score"])

    def _expiry_exit(
        self,
        position: _ExitPosition,
        entry: _EntrySnapshot,
        current_path_score: float | None,
    ) -> str | None:
        if position.age_candles < self._effective_max_hold():
            return None
        if position.price_profit > 0 and _profitable(position):
            return "simple_prediction_expired_profit"
        ratio = self._adaptive_float("expiry_decay_ratio", self.config.exit.expiry_decay_ratio)
        path_decayed = current_path_score is not None and entry.path_score > 0
        if path_decayed and current_path_score <= entry.path_score * ratio:
            return "simple_prediction_expired_path_decay"
        return "simple_prediction_expired"

    def decide_exit(self, *, side: str, price_profit: float, **context: Any) -> str | None:
        """Return the first matching deterministic exit reason, or ``None`` to hold."""
        position = self._exit_position(side, price_profit, context)
        entry = self._entry_snapshot(context)
        initial = self._initial_exit(position, entry)
        if initial.terminal:
            return initial.reason
        predicted_path = self._predicted_path_exit(position, entry)
        if predicted_path.terminal:
            return predicted_path.reason
        prediction_exit = self._prediction_exit(position, entry, context.get("freqai"))
        if prediction_exit.terminal:
            return prediction_exit.reason
        return self._expiry_exit(position, entry, prediction_exit.current_path_score)
